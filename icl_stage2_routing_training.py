#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stage-2 training script: fine-tunes a *router* that decides whether a
given model can answer a query, using a projector that maps frozen
semantic-embedding vectors into the token-embedding space of a
base LLM.  DeepSpeed is used for efficient single- or multi-GPU
training.

Major components
----------------
1. Projector (linear or MLP)            – learned
2. Base LLM (Qwen 2.5-7B-Instruct)      – learned
3. Embedding model (Qwen3-Embedding-8B) – frozen
"""

# --------------------------------------------------------------------------- #
#                          Standard library / third-party                     #
# --------------------------------------------------------------------------- #
import sys
import os
import json
import math
import random
import string
import shutil
import argparse
from pathlib import Path
from time import sleep
from functools import partial
from collections import defaultdict
from contextlib import nullcontext
from typing import List, Dict, Any

import numpy as np
import tqdm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn, Tensor

import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig
from safetensors.torch import save_file as safe_save_file

from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import datasets
import deepspeed
try:
    import wandb
except ImportError:
    wandb = None
try:
    from peft import LoraConfig, PeftModel, get_peft_model
except ImportError:
    LoraConfig = None
    PeftModel = None
    get_peft_model = None

# --------------------------------------------------------------------------- #
#                        Local project-specific imports                       #
# --------------------------------------------------------------------------- #
from modeling_projector import (
    ProjectorConfig,
    LinearProjector,
    MLPProjector,
)

# --------------------------------------------------------------------------- #
#                              Global constants                               #
# --------------------------------------------------------------------------- #
OOD_TASKS = {"aime", "mmlu_cf", "agi_eval", "humaneval", "korbench"}

# --------------------------------------------------------------------------- #
#                        Argument-parser / CLI interface                      #
# --------------------------------------------------------------------------- #
def parse_args():
    """Parse all CLI arguments (DeepSpeed flags are injected automatically)."""
    parser = argparse.ArgumentParser(
        description="Train a projector-augmented router with DeepSpeed"
    )

    # Paths
    parser.add_argument(
        "--base_model_name_or_path",
        type=str,
        default="./checkpoints/icl_stage1_seed42/epoch_2/llm",
        help="Local path or HF hub ID of the base LLM.",
    )
    parser.add_argument(
        "--embed_model_name_or_path",
        type=str,
        default="/path/to/Qwen3-Embedding-8B",
        help="Local path or HF hub ID of the frozen embedding model.",
    )
    parser.add_argument(
        "--experts_information_file",
        type=str,
        default="./data/experts_information_500.json",
        help="JSON containing statistics for each expert model.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default='{"train": "./data/train_router.json",'
        ' "validation": "./data/test_router.json"}',
        help="JSON mapping dataset splits to files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints",
        help="Where checkpoints and logs will be written.",
    )

    # Training hyper-parameters
    parser.add_argument("--batch_size", type=int, default=4, help="Per-GPU micro-batch size.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of micro-batches to accumulate before each optimizer update.")
    parser.add_argument("--eval_steps", type=int, default=0,
                        help="Run validation every N optimizer update steps. Set 0 to disable during training.")
    parser.add_argument("--eval_max_batches", type=int, default=0,
                        help="Maximum validation batches per rank for periodic validation. Set 0 for full validation.")
    parser.add_argument("--offload_optimizer", choices=["none", "cpu"],
                        default=os.environ.get("DEEPSPEED_OFFLOAD_OPTIMIZER", "none"),
                        help="Set to cpu to use DeepSpeed CPU optimizer offload.")
    parser.add_argument("--zero_stage", type=int, default=2, choices=[0, 1, 2, 3],
                        help="DeepSpeed ZeRO stage. Use 0 for single-GPU full fine-tuning.")
    parser.add_argument("--max_length", type=int, default=1024, help="Max input length.")
    parser.add_argument("--proj_lr", type=float, default=1e-5, help="LR for projector params.")
    parser.add_argument("--llm_lr", type=float, default=2e-6, help="LR for LLM params.")
    parser.add_argument("--freeze_llm", action="store_true",
                        help="Freeze the router LLM and train only the projector.")
    parser.add_argument("--use_lora", action="store_true",
                        help="Train LoRA adapters for the router LLM instead of full LLM weights.")
    parser.add_argument("--lora_adapter_path", type=str, default=None,
                        help="Optional stage1 LoRA adapter path to load before stage2 training.")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank for router LLM adapter training.")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha for router LLM adapter training.")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout for router LLM adapter training.")
    parser.add_argument("--lora_target_modules", type=str,
                        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                        help="Comma-separated module names targeted by LoRA.")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Total epochs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--wandb_project", type=str, default=os.environ.get("WANDB_PROJECT", "C2C_IRL"),
                        help="Weights & Biases project. Use empty string to disable.")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Optional Weights & Biases run name.")
    parser.add_argument("--wandb_mode", type=str, default=os.environ.get("WANDB_MODE", "online"),
                        help="Weights & Biases mode: online, offline, or disabled.")
    parser.add_argument("--wandb_tags", type=str, default=os.environ.get("WANDB_TAGS", ""),
                        help="Comma-separated Weights & Biases tags.")

    # Projector / cache
    parser.add_argument(
        "--projector_path",
        type=str,
        default="./checkpoints/icl_stage1_seed42/epoch_2/projector",
        help="Optional pretrained projector checkpoint.",
    )
    parser.add_argument(
        "--cached_embedding_file",
        type=str,
        default="Qwen3-Embedding-8B_500_sft.pt",
        help="File used to cache expert embeddings.",
    )
    parser.add_argument(
        "--projector_type",
        type=str,
        default="nonlinear",
        choices=["linear", "nonlinear"],
        help="Projector architecture type.",
    )

    # Distributed
    parser.add_argument("--local_rank", type=int, default=-1, help="Passed by torch.distributed.")

    # Let DeepSpeed add its own flags (e.g. --deepspeed, --master_port …)
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()

# --------------------------------------------------------------------------- #
#                          Helper / utility functions                         #
# --------------------------------------------------------------------------- #
def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """
    Pool the last *valid* token for each sequence.

    Works for either left- or right-padded inputs.
    """
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    seq_len = attention_mask.sum(dim=1) - 1
    batch = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch, device=last_hidden_states.device), seq_len]


def load_json(path: str | Path) -> List[Dict[str, Any]]:
    """Simple JSON loader that always returns list."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def pad_batch(
    batch: Dict[str, List[Tensor]],
    pad_token_id: int,
    label_pad_token_id: int = -100,
) -> Dict[str, Tensor]:
    """
    Right-pad variable-length tensors inside `batch` so every sequence is
    the same length.  Works on input_ids, labels, and attention_mask.
    """
    max_len = max(t.size(1) for t in batch["input_ids"])
    padded_iids, padded_lbls, padded_masks = [], [], []

    for iids, lbls, mask in zip(batch["input_ids"], batch["labels"], batch["attention_mask"]):
        pad_len = max_len - iids.size(1)

        # input_ids
        pad_val = torch.full((iids.size(0), pad_len), pad_token_id, dtype=torch.long)
        padded_iids.append(torch.cat([iids, pad_val], dim=1) if pad_len > 0 else iids)

        # labels
        pad_val = torch.full((lbls.size(0), pad_len), label_pad_token_id, dtype=torch.long)
        padded_lbls.append(torch.cat([lbls, pad_val], dim=1) if pad_len > 0 else lbls)

        # attention_mask
        pad_val = torch.zeros((mask.size(0), pad_len), dtype=torch.long)
        padded_masks.append(torch.cat([mask, pad_val], dim=1) if pad_len > 0 else mask)

    batch["input_ids"] = torch.cat(padded_iids, dim=0)
    batch["labels"] = torch.cat(padded_lbls, dim=0)
    batch["attention_mask"] = torch.cat(padded_masks, dim=0)
    return batch


def compute_expert_text_mapping(queries: List[str]):
    """
    Build a mapping {text → unique_idx} and return the index dict plus
    the ordered list of unique texts.
    """
    text2idx, unique_texts = {}, []
    for q in queries:
        if q not in text2idx:
            text2idx[q] = len(unique_texts)
            unique_texts.append(q)
    return text2idx, unique_texts


def process_batched(
    queries,
    models,
    is_correct_direct_list,
    is_correct_list,
    tasks,
    indices,
    tokenizer,
    template_type,
    expert_info=None,
    expert_text_to_idx=None,
):
    """
    Build one training example per input record, returning a dict ready
    for the DataLoader.  All heavy-weight strings (expert texts) are
    represented only by their global indices to minimise memory.
    """
    def encode(text: str) -> Tensor:
        return tokenizer.encode(
            text,
            return_tensors="pt",
            truncation=False,
            padding=False,
            add_special_tokens=False,
        )

    batch = {
        "input_ids": [],
        "labels": [],
        "attention_mask": [],
        "text_indices": [],
        "is_correct_direct": [],
        "is_correct": [],
        "task": [],
        "index": [],
        "model": [],
        "query": [],
    }

    icl_pad = torch.tensor([[151662]], dtype=torch.long)  # placeholder token
    sep_single = encode("\n")
    sep_double = encode("\n\n")

    for query, model, is_cd, is_c, task, idx in zip(
        queries, models, is_correct_direct_list, is_correct_list, tasks, indices
    ):

        # Prompt header
        system_prompt = (
            "<|im_start|>system\n"
            "You are a helpful assistant. Your task is to judge whether a "
            "model can answer a query based on its historical performance "
            "across many queries: “Yes” means it answered correctly, "
            "“No” means it did not.<|im_end|>\n"
            "<|im_start|>user\n"
        )
        header = encode(system_prompt)

        # Build the performance table (“few-shot” evidence)
        perf_tokens = [encode("Model's Performance:\n")]
        text_indices = []

        for k, ex in enumerate(expert_info[model]):
            inp, lbl = ex["input"], ex["label"]
            text_indices.append(expert_text_to_idx[inp])

            postfix = encode(lbl)
            tok = torch.cat([icl_pad, postfix], dim=1)
            perf_tokens.append(tok)

        perf_tokens.append(sep_double)
        perf_concat = torch.cat(perf_tokens, dim=1)

        # Query and instruction
        query_tokens = encode(query)
        instr = encode(
            'Indicate whether the model is capable of answering this query '
            'by responding with "Yes" or "No".' + tokenizer.eos_token + "\n"
        )
        assistant_tag = encode("<|im_start|>assistant\n")
        gold = encode("Yes" if is_cd else "No")

        # Full input sequence
        input_ids = torch.cat(
            [
                header,
                perf_concat,
                query_tokens,
                sep_double,
                instr,
                assistant_tag,
                gold,
            ],
            dim=1,
        )

        # Compute label mask such that loss is restricted to gold answer
        prefix_len = (
            header.size(1)
            + perf_concat.size(1)
            + query_tokens.size(1)
            + sep_double.size(1)
            + instr.size(1)
            + assistant_tag.size(1)
        )
        loss_start = prefix_len - 1
        output_len = gold.size(1)

        labels = input_ids.clone()
        labels_shifted = torch.full_like(labels, -100)
        labels_shifted[:, :-1] = labels[:, 1:]
        labels_shifted[:, -1] = -100
        labels_shifted[:, :loss_start] = -100
        labels_shifted[:, loss_start + output_len :] = -100

        # Append to batch
        batch["input_ids"].append(input_ids)
        batch["labels"].append(labels_shifted)
        batch["text_indices"].append(text_indices)
        batch["attention_mask"].append(torch.ones_like(input_ids))
        batch["is_correct_direct"].append(is_cd)
        batch["is_correct"].append(is_c)
        batch["task"].append(task)
        batch["index"].append(idx)
        batch["model"].append(model)
        batch["query"].append(query)

    return batch


def custom_collate_fn(batch, tokenizer):
    """
    Collate the list-of-dicts produced by `process_batched` into a single
    padded batch suitable for training.
    """
    out = {}
    for key in batch[0]:
        if key in {"input_ids", "labels", "attention_mask", "text_indices"}:
            out[key] = [torch.tensor(b[key]) for b in batch]
        else:
            out[key] = [b[key] for b in batch]

    return pad_batch(out, pad_token_id=tokenizer.pad_token_id, label_pad_token_id=-100)


def get_rank() -> int:
    """Return the global rank (0 if not using distributed)."""
    return dist.get_rank() if dist.is_initialized() else 0


def maybe_init_wandb(args, save_key: str, extra_config=None):
    if get_rank() != 0 or not args.wandb_project or args.wandb_mode == "disabled":
        return None
    if wandb is None:
        print("[wandb] package is not installed; skipping wandb logging.")
        return None
    tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
    config = vars(args).copy()
    config["save_key"] = save_key
    if extra_config:
        config.update(extra_config)
    try:
        return wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or save_key,
            mode=args.wandb_mode,
            tags=tags,
            config=config,
        )
    except Exception as exc:
        print(f"[wandb] init failed; skipping wandb logging: {exc}")
        return None


def wandb_log(run, payload, step=None):
    if run is not None:
        wandb.log(payload, step=step)


def sanitize_metric_name(value: str) -> str:
    allowed = set(string.ascii_letters + string.digits + "_.-")
    return "".join(ch if ch in allowed else "_" for ch in str(value))[:120]


def yes_no_scores(logits: Tensor, labels: Tensor, yes_id: int, no_id: int) -> tuple[Tensor, Tensor]:
    mask = labels != -100
    ans_pos = mask.float().argmax(dim=1).long()
    b_idx = torch.arange(logits.size(0), device=logits.device)
    targets = (labels[b_idx, ans_pos] == yes_id).float()

    logit_pos = ans_pos
    if logits.size(1) != labels.size(1):
        offset = labels.size(1) - logits.size(1)
        if torch.any(logit_pos < offset):
            raise RuntimeError("Answer token is outside logits_to_keep window.")
        logit_pos = logit_pos - offset

    diff = logits[b_idx, logit_pos, yes_id] - logits[b_idx, logit_pos, no_id]
    return diff, targets


def validate_router(
    model_engine,
    val_dl,
    cached_embeds: Tensor,
    device: torch.device,
    yes_id: int,
    no_id: int,
    max_batches: int = 0,
) -> Dict[str, float] | None:
    model_engine.eval()
    loss_sum = torch.tensor(0.0, device=device)
    row_count = torch.tensor(0.0, device=device)
    correct_count = torch.tensor(0.0, device=device)
    pos_count = torch.tensor(0.0, device=device)
    local_rows = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dl):
            if max_batches > 0 and batch_idx >= max_batches:
                break

            flat_idx = [int(i) for sl in batch["text_indices"] for i in sl]
            embeds = cached_embeds[torch.tensor(flat_idx, device=device)]
            n = len(batch["text_indices"])
            m = len(batch["text_indices"][0])
            last_states = embeds.view(n, m, cached_embeds.shape[1])

            logits = model_engine(
                last_states,
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                logits_to_keep=2,
            )
            labels = batch["labels"].to(device)
            scores, targets = yes_no_scores(logits, labels, yes_id, no_id)

            loss_sum += F.binary_cross_entropy_with_logits(scores, targets, reduction="sum")
            row_count += targets.numel()
            correct_count += ((scores > 0.0) == targets.bool()).float().sum()
            pos_count += targets.sum()

            for i, score in enumerate(scores.float().cpu().tolist()):
                local_rows.append(
                    {
                        "query": batch["query"][i],
                        "task": batch["task"][i],
                        "index": int(batch["index"][i]),
                        "model": batch["model"][i],
                        "score": float(score),
                        "pred_yes": bool(score > 0.0),
                        "is_correct_direct": bool(batch["is_correct_direct"][i]),
                    }
                )

    totals = torch.stack([loss_sum, row_count, correct_count, pos_count])
    if dist.is_initialized():
        dist.all_reduce(totals)
        gathered_rows = [None] * dist.get_world_size()
        dist.all_gather_object(gathered_rows, local_rows)
        all_rows = [row for rows in gathered_rows for row in rows]
    else:
        all_rows = local_rows

    metrics = None
    if get_rank() == 0:
        metrics = {
            "stage2/val_loss": float(totals[0].item() / max(1.0, totals[1].item())),
            "stage2/val_row_binary_accuracy": float(totals[2].item() / max(1.0, totals[1].item())),
            "stage2/val_pos_rate": float(totals[3].item() / max(1.0, totals[1].item())),
            "stage2/val_rows": float(totals[1].item()),
        }

        grouped = defaultdict(list)
        for row in all_rows:
            grouped[(row["query"], row["index"])].append(row)
        selected = [max(rows, key=lambda row: row["score"]) for rows in grouped.values() if rows]
        metrics["stage2/val_selected_correct"] = (
            sum(row["is_correct_direct"] for row in selected) / max(1, len(selected))
        )
        metrics["stage2/val_oracle_possible"] = (
            sum(any(row["is_correct_direct"] for row in rows) for rows in grouped.values())
            / max(1, len(grouped))
        )
        metrics["stage2/val_queries"] = float(len(grouped))

        candidates = sorted({row["model"] for row in all_rows})
        for name in candidates:
            key = sanitize_metric_name(name)
            rows = [row for row in all_rows if row["model"] == name]
            chosen = [row for row in selected if row["model"] == name]
            metrics[f"stage2/val_static_acc/{key}"] = (
                sum(row["is_correct_direct"] for row in rows) / max(1, len(rows))
            )
            metrics[f"stage2/val_mean_score/{key}"] = (
                sum(row["score"] for row in rows) / max(1, len(rows))
            )
            metrics[f"stage2/val_selection_frac/{key}"] = len(chosen) / max(1, len(selected))
            metrics[f"stage2/val_selection_acc/{key}"] = (
                sum(row["is_correct_direct"] for row in chosen) / max(1, len(chosen))
            )
            metrics[f"stage2/val_selected_pred_yes_frac/{key}"] = (
                sum(row["pred_yes"] for row in chosen) / max(1, len(chosen))
            )

    model_engine.train()
    return metrics


def parse_lora_target_modules(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def save_model_and_configs(args, model_engine, key, stage: str):
    """
    Save projector + LLM weights along with tokenizer and configs.

    Only executed on rank-0 to avoid redundant I/O.
    """
    if get_rank() != 0:
        return

    ckpt_dir = Path(args.output_dir) / key / stage
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1) Projector weights
    proj_dir = ckpt_dir / "projector"
    proj_dir.mkdir(exist_ok=True)
    safe_save_file(
        model_engine.module.projector.state_dict(),
        proj_dir / "model.safetensors",
        metadata={"format": "pt"},
    )
    model_engine.module.projector.config.save_pretrained(proj_dir)
    print(f"[Checkpoint] Projector saved → {proj_dir}")

    # 2) LLM weights. Qwen ties embed_tokens.weight and lm_head.weight, so
    # saving the raw state_dict with safetensors.save_file fails.
    llm_dir = ckpt_dir / "llm"
    llm_dir.mkdir(exist_ok=True)
    model_engine.module.big_model.save_pretrained(
        llm_dir,
        safe_serialization=True,
    )
    print(f"[Checkpoint] LLM saved → {llm_dir}")

    # 3) Config + tokenizer
    AutoConfig.from_pretrained(args.base_model_name_or_path, trust_remote_code=True).save_pretrained(llm_dir)
    AutoTokenizer.from_pretrained(args.base_model_name_or_path, trust_remote_code=True).save_pretrained(llm_dir)
    with (llm_dir / "c2c_router_training_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "base_model_name_or_path": args.base_model_name_or_path,
                "use_lora": args.use_lora,
                "lora_adapter_path": args.lora_adapter_path,
                "lora_r": args.lora_r if args.use_lora else None,
                "lora_alpha": args.lora_alpha if args.use_lora else None,
                "lora_dropout": args.lora_dropout if args.use_lora else None,
                "lora_target_modules": parse_lora_target_modules(args.lora_target_modules)
                if args.use_lora else None,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 4) generation_config.json (not handled by save_pretrained)
    src = Path(args.base_model_name_or_path) / "generation_config.json"
    dst = llm_dir / "generation_config.json"
    if src.exists():
        shutil.copy(src, dst)
        print("[Checkpoint] generation_config.json copied.")

# --------------------------------------------------------------------------- #
#                          Composite model (projector + LLM)                  #
# --------------------------------------------------------------------------- #
class CompositeModel(nn.Module):
    """
    A thin wrapper that plugs the projector into the LLM by replacing
    placeholder tokens (ID 151662) with projector outputs before passing
    the modified embeddings into the LLM.
    """

    def __init__(self, projector, big_model):
        super().__init__()
        self.projector = projector
        self.big_model = big_model

    def forward(self, last_states_tensor, input_ids, attention_mask, **llm_kwargs):
        # Project into LLM embedding space
        last_states_tensor = last_states_tensor.to(next(self.projector.parameters()).dtype)
        projected = self.projector(last_states_tensor)

        # Embed tokens from the LLM’s own embedding layer
        embed_layer = self.big_model.get_input_embeddings()
        model_inp = embed_layer(input_ids)

        mask = input_ids.eq(151662)
        order = torch.cumsum(mask, dim=1) - 1
        rows = (
            torch.arange(input_ids.size(0), device=input_ids.device)
            .unsqueeze(1)
            .expand_as(input_ids)
        )

        replacement = torch.zeros_like(model_inp)
        replacement[mask] = projected[rows[mask], order[mask]]
        model_inp = torch.where(mask.unsqueeze(-1), replacement, model_inp)

        outputs = self.big_model(inputs_embeds=model_inp, attention_mask=attention_mask, **llm_kwargs)
        return outputs.logits

# --------------------------------------------------------------------------- #
#                                   Training                                  #
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()

    # Seed everything
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # ---------------- Distributed init ----------------
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        device = torch.device("cuda", args.local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- Load base LLM -------------------
    big_model = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if args.lora_adapter_path is not None:
        if PeftModel is None:
            raise ImportError("peft is required for --lora_adapter_path. Install it with `pip install peft`.")
        big_model = PeftModel.from_pretrained(
            big_model,
            args.lora_adapter_path,
            is_trainable=args.use_lora and not args.freeze_llm,
        )
    elif args.use_lora:
        if get_peft_model is None:
            raise ImportError("peft is required for --use_lora. Install it with `pip install peft`.")
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=parse_lora_target_modules(args.lora_target_modules),
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        big_model = get_peft_model(big_model, lora_cfg)
    if args.use_lora and hasattr(big_model, "enable_input_require_grads"):
        big_model.enable_input_require_grads()
    if args.use_lora and get_rank() == 0 and hasattr(big_model, "print_trainable_parameters"):
        big_model.print_trainable_parameters()
    big_model.config.use_cache = False
    big_model.gradient_checkpointing_enable()
    big_model.to(device)

    # ---------------- Tokenizer -----------------------
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------------- Embedding model -----------------
    # Frozen; only used to compute expert-text embeddings. If the cache is
    # complete, the cached tensor shape is enough to initialize the projector,
    # so do not download/load large embedding checkpoints.
    output_root = Path(args.output_dir)
    cache_path_for_shape = output_root / args.cached_embedding_file
    embed_model = None
    embed_tokenizer = None
    if cache_path_for_shape.exists():
        cached_for_shape = torch.load(cache_path_for_shape, map_location="cpu")
        in_features = int(cached_for_shape["embeddings"].shape[1])
        expansion_ratio = 1
        if get_rank() == 0:
            print(f"[Cache] Found embedding cache at {cache_path_for_shape}; skipping embed model load.")
    elif any(k in args.embed_model_name_or_path for k in ["stella_en_1.5B_v5", "bge"]):
        embed_model = SentenceTransformer(args.embed_model_name_or_path, trust_remote_code=True)
        embed_model.max_seq_length = args.max_length
        in_features = 2048 if "stella" in args.embed_model_name_or_path else embed_model.get_sentence_embedding_dimension()
        expansion_ratio = 1
    elif "NV-Embed-v2" in args.embed_model_name_or_path:
        embed_cfg = AutoConfig.from_pretrained(args.embed_model_name_or_path, trust_remote_code=True)
        embed_model = AutoModel.from_pretrained(
            args.embed_model_name_or_path, config=embed_cfg, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        in_features, expansion_ratio = 4096, 1
    elif "mxbai-embed-large-v1" in args.embed_model_name_or_path:
        in_features, expansion_ratio = 1024, 1
        embed_model = SentenceTransformer(
            args.embed_model_name_or_path, truncate_dim=in_features, trust_remote_code=True
        )
        embed_model.max_seq_length = 512
    elif any(k in args.embed_model_name_or_path for k in ["Qwen3-Embedding", "gte_Qwen2"]):
        embed_tokenizer = AutoTokenizer.from_pretrained(args.embed_model_name_or_path, padding_side="left")
        embed_model = AutoModel.from_pretrained(
            args.embed_model_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        in_features, expansion_ratio = embed_model.config.hidden_size, 1
    else:
        raise ValueError("Unsupported embedding model")

    if embed_model is not None:
        embed_model.to(device).eval()

    # ---------------- Projector -----------------------
    out_features = big_model.get_input_embeddings().weight.shape[1]
    proj_cfg = ProjectorConfig(in_features=in_features, out_features=out_features, expansion_ratio=expansion_ratio)
    Projector = LinearProjector if args.projector_type == "linear" else MLPProjector
    if args.projector_path is not None:
        projector = Projector.from_pretrained(args.projector_path, config=proj_cfg, dtype=torch.bfloat16)
    else:
        projector = Projector(config=proj_cfg, dtype=torch.bfloat16)
    projector.to(device)

    # Special-token IDs for binary classification
    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("No", add_special_tokens=False)[0]

    # ---------------- Datasets -----------------------
    data_files = json.loads(args.dataset_name)
    raw = datasets.load_dataset("json", data_files=data_files)

    # Expert information JSON
    with open(args.experts_information_file, "r", encoding="utf-8") as f:
        experts_information = json.load(f)

    # Build global set of expert texts (only need *one* expert list to get all texts)
    first_expert = next(iter(experts_information.values()))
    queries_unique = list(dict.fromkeys(ex["input"] for ex in first_expert))
    expert_text_to_idx, global_expert_texts = compute_expert_text_mapping(queries_unique)

    # Pre-process datasets
    ds_kwargs = dict(
        tokenizer=tokenizer,
        template_type="qwen",
        expert_info=experts_information,
        expert_text_to_idx=expert_text_to_idx,
    )
    train_ds = raw["train"].map(
        process_batched,
        fn_kwargs=ds_kwargs,
        batched=True,
        batch_size=args.batch_size,
        num_proc=32,
        input_columns=["query", "model", "is_correct_direct", "is_correct", "task", "index"],
        remove_columns=raw["train"].column_names,
    )
    val_ds = raw["validation"].map(
        process_batched,
        fn_kwargs=ds_kwargs,
        batched=True,
        batch_size=args.batch_size,
        num_proc=32,
        input_columns=["query", "model", "is_correct_direct", "is_correct", "task", "index"],
        remove_columns=raw["validation"].column_names,
    )

    collate_fn = partial(custom_collate_fn, tokenizer=tokenizer)
    if args.local_rank != -1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds, seed=args.seed, shuffle=True, drop_last=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds, shuffle=False, drop_last=False)
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, collate_fn=collate_fn, drop_last=True)
        val_dl   = torch.utils.data.DataLoader(val_ds,   batch_size=args.batch_size, sampler=val_sampler,   collate_fn=collate_fn, drop_last=False)
    else:
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  collate_fn=collate_fn, drop_last=True)
        val_dl   = torch.utils.data.DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)

    # ---------------- DeepSpeed config --------------
    total_micro_steps = len(train_dl) * args.num_train_epochs
    total_update_steps = math.ceil(len(train_dl) / args.gradient_accumulation_steps) * args.num_train_epochs
    warmup_steps = int(0.1 * total_update_steps)
    ds_cfg = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "weight_decay": 0.01,
                "torch_adam": True,
            },
        },
        "scheduler": {"type": "WarmupCosineLR", "params": {"warmup_num_steps": warmup_steps, "total_num_steps": total_update_steps}},
        "bf16": {"enabled": True},
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": int(args.zero_stage),
        },
    }
    if args.zero_stage > 0:
        ds_cfg["zero_optimization"].update({
            "allgather_partitions": True,
            "allgather_bucket_size": 5e7,
            "overlap_comm": False,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e7,
            "contiguous_gradients": True,
            "round_robin_gradients": False,
            "ignore_unused_parameters": True,
        })
    if args.offload_optimizer == "cpu":
        if args.zero_stage <= 0:
            raise ValueError("--offload_optimizer cpu requires --zero_stage > 0")
        ds_cfg["optimizer"]["params"].pop("torch_adam", None)
        ds_cfg["optimizer"]["params"]["fp32_optimizer_states"] = False
        ds_cfg["zero_optimization"]["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": False,
        }

    # Model + optimizer groups
    composite = CompositeModel(projector, big_model)
    if args.freeze_llm:
        for p in composite.big_model.parameters():
            p.requires_grad = False
    optim_groups = [
        {"params": composite.projector.parameters(), "lr": args.proj_lr, "weight_decay": 0.01, "name": "projector"},
    ]
    if not args.freeze_llm:
        llm_params = [p for p in composite.big_model.parameters() if p.requires_grad]
        if args.use_lora and not llm_params:
            raise RuntimeError("--use_lora was set, but no trainable LoRA parameters were found.")
        optim_groups.append(
            {"params": llm_params, "lr": args.llm_lr,  "weight_decay": 0.01, "name": "big_model"}
        )

    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        args=args, model=composite, model_parameters=optim_groups, config=ds_cfg
    )

    # ---------------- Embedding cache ---------------
    rank, world_size = get_rank(), (dist.get_world_size() if dist.is_initialized() else 1)
    if rank == 0:
        output_root.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()
    cache_path = output_root / args.cached_embedding_file

    # Load existing cache if available
    if cache_path.exists():
        cache = torch.load(cache_path, map_location="cpu")
        cached_texts, cached_embeds = cache["global_expert_texts"], cache["embeddings"]
        if rank == 0:
            print(f"[Cache] Loaded {len(cached_texts)} expert texts from cache.")
    else:
        cached_texts, cached_embeds = [], None

    # Determine which texts are new
    old_idx = {t: i for i, t in enumerate(cached_texts)}
    missing_texts, new_text_indices = [], []
    for text in global_expert_texts:
        if text in old_idx:
            new_text_indices.append(old_idx[text])
        else:
            new_text_indices.append(-1)
            missing_texts.append(text)

    if missing_texts and embed_model is None:
        raise RuntimeError(
            f"Embedding cache {cache_path} is incomplete ({len(missing_texts)} missing texts), "
            "but the embedding model was not loaded. Delete or complete the cache."
        )

    # Compute embeddings for new texts (distributed)
    if missing_texts:
        total_missing = len(missing_texts)
        slice_size = total_missing // world_size
        remainder = total_missing % world_size

        if rank < remainder:
            s, e = rank * (slice_size + 1), rank * (slice_size + 1) + slice_size + 1
        else:
            s = rank * slice_size + remainder
            e = s + slice_size
        local_missing = missing_texts[s:e]
        print(f"[Rank {rank}] Encoding {len(local_missing)} new expert texts.")

        with torch.no_grad():
            local_embeds = []
            for i in tqdm.tqdm(range(0, len(local_missing), 2), desc=f"Rank {rank}"):
                bt = local_missing[i : i + 2]
                if "NV-Embed-v2" in args.embed_model_name_or_path:
                    emb = embed_model.encode(bt, instruction="", max_length=args.max_length, convert_to_tensor=True)
                elif any(k in args.embed_model_name_or_path for k in ["Qwen3-Embedding", "gte_Qwen2"]):
                    batch_dict = embed_tokenizer(bt, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt").to(embed_model.device)
                    outputs = embed_model(**batch_dict)
                    emb = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
                else:
                    emb = embed_model.encode(bt, convert_to_tensor=True, normalize_embeddings=False)
                local_embeds.append(emb)
            local_embeds = torch.cat(local_embeds, dim=0).cpu()

        if dist.is_initialized():
            gathered = [None] * world_size
            dist.all_gather_object(gathered, local_embeds)
            if rank == 0:
                all_new_embeds = torch.cat(gathered, dim=0)
        else:
            all_new_embeds = local_embeds

        if rank == 0:
            new_ordered = []
            ptr = 0
            for idx in new_text_indices:
                if idx == -1:
                    new_ordered.append(all_new_embeds[ptr : ptr + 1])
                    ptr += 1
                else:
                    new_ordered.append(cached_embeds[idx : idx + 1])
            new_global_embeds = torch.cat(new_ordered, dim=0)
            torch.save({"global_expert_texts": global_expert_texts, "embeddings": new_global_embeds}, cache_path)
            print(f"[Cache] Updated cache saved → {cache_path}")
    else:
        if rank == 0:
            print("[Cache] No new expert texts.")

    if dist.is_initialized():
        dist.barrier()

    # Move cache to GPU (same dtype as projector)
    cached = torch.load(cache_path, map_location="cpu")
    cached_embeds = cached["embeddings"].to(device, dtype=next(projector.parameters()).dtype)

    # Free embedding model memory
    if embed_model is not None:
        embed_model.to("cpu")
        del embed_model

    # -------------------------------------------------------------------- #
    #                             Training loop                            #
    # -------------------------------------------------------------------- #
    total_steps = total_micro_steps
    pbar = tqdm.tqdm(total=total_steps)

    save_key = f"{args.projector_type}_router_" \
               f"projLR{args.proj_lr}_llmLR{args.llm_lr}_epochs{args.num_train_epochs}_" \
               f"{Path(args.embed_model_name_or_path).name}_seed{args.seed}"
    (Path(args.output_dir) / save_key).mkdir(parents=True, exist_ok=True)
    wandb_run = maybe_init_wandb(
        args,
        save_key,
        extra_config={
            "stage": "stage2_router",
            "train_micro_steps": total_micro_steps,
            "train_update_steps": total_update_steps,
            "train_rows": len(train_ds),
            "validation_rows": len(val_ds),
            "micro_batch_size_per_gpu": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": args.batch_size
            * args.gradient_accumulation_steps
            * (dist.get_world_size() if dist.is_initialized() else 1),
            "eval_steps": args.eval_steps,
            "eval_max_batches": args.eval_max_batches,
            "offload_optimizer": args.offload_optimizer,
            "zero_stage": args.zero_stage,
            "freeze_llm": args.freeze_llm,
            "use_lora": args.use_lora,
            "lora_adapter_path": args.lora_adapter_path,
            "lora_r": args.lora_r if args.use_lora else None,
            "lora_alpha": args.lora_alpha if args.use_lora else None,
            "lora_dropout": args.lora_dropout if args.use_lora else None,
            "lora_target_modules": parse_lora_target_modules(args.lora_target_modules)
            if args.use_lora else None,
        },
    )

    global_step = 0
    update_step = 0
    accum_loss_sum = 0.0
    accum_loss_count = 0
    loss_ema = None
    model_engine.train()
    for epoch in range(args.num_train_epochs):
        if args.local_rank != -1:
            train_dl.sampler.set_epoch(epoch)

        for batch in train_dl:
            global_step += 1

            with torch.no_grad():
                flat_idx = [int(i) for sl in batch["text_indices"] for i in sl]
                embeds = cached_embeds[torch.tensor(flat_idx, device=device)]
                n, m, d = len(batch["text_indices"]), len(batch["text_indices"][0]), cached_embeds.shape[1]
                last_states = embeds.view(n, m, d)

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model_engine(last_states, input_ids, attention_mask)

            # Binary-cross-entropy over the single answer token
            diff, targets = yes_no_scores(logits, labels, yes_id, no_id)
            loss = F.binary_cross_entropy_with_logits(diff, targets)
            pos_rate = targets.mean()

            model_engine.backward(loss)
            if (
                not torch.isfinite(loss)
                or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model_engine.module.parameters())
            ):
                if get_rank() == 0:
                    print(f"[Step {global_step}] Non-finite loss/grad – skipping batch.")
                model_engine.zero_grad()
                accum_loss_sum = 0.0
                accum_loss_count = 0
                continue

            is_update_step = model_engine.is_gradient_accumulation_boundary()
            model_engine.step()
            torch.cuda.empty_cache()

            # Logging (average across GPUs)
            with torch.no_grad():
                red_loss = loss.clone()
                red_pos_rate = pos_rate.clone()
                if dist.is_initialized():
                    dist.all_reduce(red_loss)
                    dist.all_reduce(red_pos_rate)
                    red_loss /= dist.get_world_size()
                    red_pos_rate /= dist.get_world_size()
            if get_rank() == 0:
                accum_loss_sum += float(red_loss.item())
                accum_loss_count += 1
                loss_ema = float(red_loss.item()) if loss_ema is None else 0.95 * loss_ema + 0.05 * float(red_loss.item())
            if is_update_step:
                update_step += 1
            should_log = is_update_step or global_step % 5 == 0
            if get_rank() == 0 and should_log:
                accum_loss_mean = accum_loss_sum / max(1, accum_loss_count)
                grad_norm = model_engine.get_global_grad_norm()
                grad_norm_value = float(grad_norm) if grad_norm is not None else float("nan")
                tqdm.tqdm.write(
                    f"Epoch {epoch} | Step {global_step} | Update {update_step} | "
                    f"Loss {red_loss.item():.4f} | Accum-loss {accum_loss_mean:.4f} | "
                    f"Grad-norm {grad_norm_value:.4f}"
                )
                wandb_log(
                    wandb_run,
                    {
                        "stage2/train_loss": float(red_loss.item()),
                        "stage2/train_loss_ema": float(loss_ema),
                        "stage2/accum_loss_mean": float(accum_loss_mean),
                        "stage2/batch_pos_rate": float(red_pos_rate.item()),
                        "stage2/grad_norm": grad_norm_value,
                        "stage2/epoch": epoch,
                        "stage2/progress": global_step / max(1, total_steps),
                        "stage2/update_step": update_step,
                    },
                    step=global_step,
                )
                if is_update_step:
                    accum_loss_sum = 0.0
                    accum_loss_count = 0

            if args.eval_steps > 0 and is_update_step and update_step % args.eval_steps == 0:
                if get_rank() == 0:
                    tqdm.tqdm.write(
                        f"Epoch {epoch} | Step {global_step} | Update {update_step} | "
                        "running validation"
                    )
                val_metrics = validate_router(
                    model_engine,
                    val_dl,
                    cached_embeds,
                    device,
                    yes_id,
                    no_id,
                    max_batches=args.eval_max_batches,
                )
                if get_rank() == 0 and val_metrics is not None:
                    tqdm.tqdm.write(
                        f"Epoch {epoch} | Step {global_step} | Update {update_step} | "
                        f"Val-loss {val_metrics['stage2/val_loss']:.4f} | "
                        f"Val-selected {val_metrics['stage2/val_selected_correct']:.4f} | "
                        f"Val-oracle {val_metrics['stage2/val_oracle_possible']:.4f}"
                    )
                    payload = dict(val_metrics)
                    payload["stage2/update_step"] = update_step
                    wandb_log(wandb_run, payload, step=global_step)

            # Periodic checkpoint
            if global_step % 500 == 0 and global_step >= 2000 and get_rank() == 0:
                save_model_and_configs(args, model_engine, save_key, f"step{global_step}")
                wandb_log(
                    wandb_run,
                    {"stage2/checkpoint_step": global_step},
                    step=global_step,
                )

            pbar.update(1)

    if get_rank() == 0:
        final_stage = f"final_step{global_step}"
        save_model_and_configs(args, model_engine, save_key, final_stage)
        wandb_log(
            wandb_run,
            {"stage2/final_checkpoint_step": global_step},
            step=global_step,
        )

    if wandb_run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
