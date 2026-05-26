#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train a projector that maps embedding-model vectors into the token-embedding
space of a large language model (LLM).  Uses DeepSpeed for efficient, possibly
distributed, training.  The script supports several projector architectures
(Linear, MLP, Q-Former) and multiple text-embedding back-ends.
"""

import sys
import os
import json
import copy
import shutil
import random
import time
import re
import math
from functools import partial

import numpy as np
import tqdm
import argparse

import torch
import torch.distributed as dist
from torch import nn, Tensor
import torch.nn.functional as F

import transformers
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    AutoConfig,
)

from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import datasets
import deepspeed
from safetensors.torch import save_file as safe_save_file
try:
    import wandb
except ImportError:
    wandb = None

from modeling_projector import (
    ProjectorConfig,
    LinearProjector,
    MLPProjector,
    QFormerProjector,
    QFormerProjectorConfig,
)


# --------------------------------------------------------------------------- #
#                        Command-line argument handling                        #
# --------------------------------------------------------------------------- #
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a projector with DeepSpeed")

    # Model paths
    parser.add_argument("--base_model_name_or_path", type=str,
                        default="/path/to/Qwen2.5-7B-Instruct",
                        help="Path or HF hub ID for the LLM.")
    parser.add_argument("--embed_model_name_or_path", type=str,
                        default="/path/to/Qwen3-8B-embedding",
                        help="Path or HF hub ID for the embedding model.")

    # Data
    parser.add_argument("--dataset_name", type=str,
                        default='{"train": "./data/question_train.json", '
                                '"validation": "./data/question_test.json"}',
                        help='JSON mapping of split names to data files.')
    parser.add_argument("--max_length", type=int, default=1024,
                        help="Max input length fed to the LLM.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Global per-step micro-batch size.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of micro-batches to accumulate before each optimizer update.")
    parser.add_argument("--offload_optimizer", choices=["none", "cpu"],
                        default=os.environ.get("DEEPSPEED_OFFLOAD_OPTIMIZER", "none"),
                        help="Set to cpu to use DeepSpeed CPU optimizer offload.")

    # Optimisation
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning-rate for the projector.")
    parser.add_argument("--llm_lr", type=float, default=5e-6,
                        help="Learning-rate for the LLM (once unfrozen).")
    parser.add_argument("--llm_unfreeze_epoch", type=int, default=1,
                        help="Epoch after which to unfreeze the LLM.")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Total number of epochs.")

    # Projector
    parser.add_argument("--projector_type", type=str, default="nonlinear",
                        choices=["linear", "nonlinear", "qformer"],
                        help="Which projector architecture to use.")

    # I/O
    parser.add_argument("--output_dir", type=str, default="../checkpoints",
                        help="Directory to save checkpoints.")
    parser.add_argument("--ckpt_key", type=str, default="icl_stage1_seed42",
                        help="Checkpoint key used to save the model.")
    parser.add_argument("--hub_model_id", type=str, default=None,
                        help="Optional HF checkpoint to start from.")
    parser.add_argument("--cached_embedding_file", type=str,
                        default="Qwen3-8B-embedding_stage1.pt",
                        help="File used to cache text embeddings.")
    parser.add_argument("--wandb_project", type=str,
                        default=os.environ.get("WANDB_PROJECT", "C2C_IRL"),
                        help="Weights & Biases project. Use empty string to disable.")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Optional Weights & Biases run name.")
    parser.add_argument("--wandb_mode", type=str,
                        default=os.environ.get("WANDB_MODE", "online"),
                        help="Weights & Biases mode: online, offline, or disabled.")
    parser.add_argument("--wandb_tags", type=str,
                        default=os.environ.get("WANDB_TAGS", ""),
                        help="Comma-separated Weights & Biases tags.")

    # Distributed
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank passed by torch.distributed.")

    # Let DeepSpeed inject its own flags
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()


# --------------------------------------------------------------------------- #
#                         Helper / utility functions                          #
# --------------------------------------------------------------------------- #
def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """
    Pool the final token for each sequence (accounts for left or right padding).

    Args:
        last_hidden_states: (B, L, H) hidden states.
        attention_mask:     (B, L) attention mask.

    Returns:
        (B, H) tensor containing the last valid token for each sequence.
    """
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        seq_len = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), seq_len]


def pad_batch(batch, pad_token_id, label_pad_token_id: int = -100):
    """
    Right-pad a batch of variable-length tensors to the same length.
    Applies to input_ids, labels, and attention_mask.
    """
    max_len = max(t.size(1) for t in batch["input_ids"])

    padded_iids, padded_lbls, padded_masks = [], [], []
    for iids, lbls, amask in zip(batch["input_ids"], batch["labels"], batch["attention_mask"]):
        curr_len = iids.size(1)
        pad_len = max_len - curr_len

        # input_ids
        padded_iids.append(
            torch.cat([iids,
                       torch.full((iids.size(0), pad_len), pad_token_id, dtype=torch.long)], dim=1)
            if pad_len > 0 else iids
        )

        # labels
        padded_lbls.append(
            torch.cat([lbls,
                       torch.full((lbls.size(0), pad_len), label_pad_token_id, dtype=torch.long)], dim=1)
            if pad_len > 0 else lbls
        )

        # mask
        padded_masks.append(
            torch.cat([amask, torch.zeros((amask.size(0), pad_len), dtype=torch.long)], dim=1)
            if pad_len > 0 else amask
        )

    batch["input_ids"] = torch.cat(padded_iids, dim=0)
    batch["labels"] = torch.cat(padded_lbls, dim=0)
    batch["attention_mask"] = torch.cat(padded_masks, dim=0)
    return batch


def compute_expert_text_mapping(dataset):
    """
    Iterate over the dataset and build a mapping:
        expert_text  -->  unique global index
    Returns:
        expert_text_to_idx  : dict
        global_expert_texts : list[str]
    """
    expert_text_to_idx, global_expert_texts = {}, []
    for sample in dataset:
        question = sample["question"]
        if question not in expert_text_to_idx:
            expert_text_to_idx[question] = len(global_expert_texts)
            global_expert_texts.append(question)
    return expert_text_to_idx, global_expert_texts


def process_batched(questions, tokenizer, template_type, expert_text_to_idx, max_length=None):
    """
    Build the prompt / target tensors for a list of questions.

    The prompt format:
    -------------------------------------------------------------------
    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    Here is a query embedding:  <PAD_FOR_EMBED>
    <SEP>

    <random expression>              (e.g. "What's the content of it?")
    <|im_start|>assistant
    <gold question text>
    -------------------------------------------------------------------
    Labels are shifted such that the loss is computed only over
    the assistant's answer (<gold question text>).
    """
    def encode(text: str) -> Tensor:
        return tokenizer.encode(text,
                                return_tensors="pt",
                                truncation=False,
                                padding=False,
                                add_special_tokens=False)

    expressions = [
        "What's the content of it?",
        "What does this embedding represent?",
        "What is the semantic content of this embedding?",
        "What is encoded in this embedding?",
        "Infer what the original query might be about.",
    ]

    batch = {"input_ids": [], "labels": [], "attention_mask": [], "text_indices": []}

    icl_pad = torch.tensor([[151662]], dtype=torch.long)  # special placeholder token
    eos = torch.tensor([[tokenizer.eos_token_id]], dtype=torch.long)
    sep = encode("\n\n")

    for question in questions:
        text_indices = [expert_text_to_idx[question]]

        # System + user prefix
        template = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\nHere is a query embedding: "
        )
        template_tokens = encode(template)

        # Random clarification sentence
        random_expr = encode(f"{random.choice(expressions)}{tokenizer.eos_token}\n")

        # Assistant prefix
        assistant_tok = encode("<|im_start|>assistant\n")

        prefix_len = (
            template_tokens.size(1)
            + 1
            + sep.size(1)
            + random_expr.size(1)
            + assistant_tok.size(1)
        )

        # Gold answer (original question text). Keep the original stage-1
        # reconstruction objective, but honor --max_length so rare very long
        # examples do not make full-LLM unfreeze infeasible on a single 24GB GPU.
        if max_length is not None:
            gold_budget = max(1, max_length - prefix_len)
            question_budget = max(0, gold_budget - 1)
            gold_ids = tokenizer.encode(
                question,
                truncation=True,
                max_length=question_budget,
                padding=False,
                add_special_tokens=False,
            ) if question_budget > 0 else []
            gold_ids.append(tokenizer.eos_token_id)
            gold = torch.tensor([gold_ids], dtype=torch.long)
        else:
            gold = encode(f"{question}{tokenizer.eos_token}")

        input_ids = torch.cat(
            [template_tokens, icl_pad, sep, random_expr, assistant_tok, gold], dim=1
        )

        loss_start = prefix_len - 1
        output_len = gold.size(1)

        # Build shifted labels: ignore everything except the answer text
        labels = input_ids.clone()
        labels_shifted = torch.full_like(labels, -100)
        labels_shifted[:, :-1] = labels[:, 1:]
        labels_shifted[:, -1] = -100
        labels_shifted[:, :loss_start] = -100
        labels_shifted[:, loss_start + output_len :] = -100

        batch["input_ids"].append(input_ids)
        batch["labels"].append(labels_shifted)
        batch["text_indices"].append(text_indices)
        batch["attention_mask"].append(torch.ones_like(input_ids))

    return batch


def custom_collate_fn(batch, tokenizer):
    """Collate list-dict into padded tensor batch."""
    collated = {}
    for key in batch[0].keys():
        if key in {"input_ids", "labels", "attention_mask", "text_indices"}:
            collated[key] = [
                torch.tensor(item[key]) if not isinstance(item[key], torch.Tensor) else item[key]
                for item in batch
            ]
        else:
            collated[key] = [item[key] for item in batch]
    return pad_batch(collated,
                     pad_token_id=tokenizer.pad_token_id,
                     label_pad_token_id=-100)


def get_rank() -> int:
    """Return distributed rank, or 0 if not initialised."""
    return dist.get_rank() if dist.is_initialized() else 0


def maybe_init_wandb(args, extra_config=None):
    if get_rank() != 0 or not args.wandb_project or args.wandb_mode == "disabled":
        return None
    if wandb is None:
        print("[wandb] package is not installed; skipping wandb logging.")
        return None
    tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
    config = vars(args).copy()
    if extra_config:
        config.update(extra_config)
    try:
        return wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or args.ckpt_key,
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


# --------------------------------------------------------------------------- #
#                         Saving helper (rank-0 only)                         #
# --------------------------------------------------------------------------- #
def save_model_and_configs(args, model_engine, key, stage: str):
    """
    Save projector + LLM weights to SAFETENSORS, plus all configs/tokenizers.

    Args:
        key   : identifier string used in checkpoint path
        stage : e.g. "epoch_1" or "final"
    """
    if get_rank() != 0:
        return

    ckpt_dir = os.path.join(args.output_dir, key, stage)
    os.makedirs(ckpt_dir, exist_ok=True)

    # 1) projector
    projector_dir = os.path.join(ckpt_dir, "projector")
    os.makedirs(projector_dir, exist_ok=True)
    safe_save_file(model_engine.module.projector.state_dict(),
                   os.path.join(projector_dir, "model.safetensors"),
                   metadata={"format": "pt"})
    print(f"[Checkpoint] Projector saved to {projector_dir}")

    # 2) LLM. Qwen ties embed_tokens.weight and lm_head.weight, so saving the
    # raw state_dict with safetensors.save_file fails on shared tensors.
    llm_dir = os.path.join(ckpt_dir, "llm")
    os.makedirs(llm_dir, exist_ok=True)
    model_engine.module.big_model.save_pretrained(
        llm_dir,
        safe_serialization=True,
    )
    print(f"[Checkpoint] LLM saved to {llm_dir}")

    # 3) Configs
    AutoConfig.from_pretrained(args.base_model_name_or_path,
                               trust_remote_code=True).save_pretrained(llm_dir)
    AutoTokenizer.from_pretrained(args.base_model_name_or_path,
                                  trust_remote_code=True).save_pretrained(llm_dir)

    # 4) generation_config.json may not be handled by save_pretrained
    src_gen_cfg = os.path.join(args.base_model_name_or_path, "generation_config.json")
    dst_gen_cfg = os.path.join(llm_dir, "generation_config.json")
    if os.path.exists(src_gen_cfg):
        shutil.copy(src_gen_cfg, dst_gen_cfg)
        print(f"[Checkpoint] Copied generation_config.json")


# --------------------------------------------------------------------------- #
#                        Composite model used by DS                           #
# --------------------------------------------------------------------------- #
class CompositeModel(nn.Module):
    """Wrap projector + frozen (or later unfrozen) LLM to train end-to-end."""

    def __init__(self, projector, big_model):
        super().__init__()
        self.projector = projector
        self.big_model = big_model

    def forward(self, last_states_tensor, input_ids, attention_mask):
        # Cast projector output to LLM dtype
        last_states_tensor = last_states_tensor.to(next(self.projector.parameters()).dtype)
        projected = self.projector(last_states_tensor)

        # Embed input_ids
        embedding_layer = self.big_model.get_input_embeddings()
        model_input = embedding_layer(input_ids)

        # Replace placeholder token 151662 with projected vectors
        mask = input_ids.eq(151662)
        order = torch.cumsum(mask, dim=1) - 1
        rows = torch.arange(input_ids.size(0), device=input_ids.device).unsqueeze(1).expand_as(input_ids)

        replacement = torch.zeros_like(model_input)
        replacement[mask] = projected[rows[mask], order[mask]]
        model_input = torch.where(mask.unsqueeze(-1), replacement, model_input)

        outputs = self.big_model(inputs_embeds=model_input, attention_mask=attention_mask)
        return outputs.logits


# --------------------------------------------------------------------------- #
#                                   Main                                      #
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()

    # -------------------------- Distributed setup -------------------------- #
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        device = torch.device("cuda", args.local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------ Load LLM ------------------------------- #
    big_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    big_model.config.use_cache = False
    big_model.gradient_checkpointing_enable()
    big_model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_name_or_path,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    embedding_layer = big_model.get_input_embeddings()

    # --------------------------- Load embed model -------------------------- #
    if any(k in args.embed_model_name_or_path for k in ["stella_en_1.5B_v5", "bge"]):
        embed_model = SentenceTransformer(args.embed_model_name_or_path, trust_remote_code=True)
        embed_model.max_seq_length = args.max_length
        in_features = 2048 if "stella_en_1.5B_v5" in args.embed_model_name_or_path else \
                      embed_model.get_sentence_embedding_dimension()
        expansion_ratio = 1

    elif "NV-Embed-v2" in args.embed_model_name_or_path:
        embed_cfg = AutoConfig.from_pretrained(args.embed_model_name_or_path, trust_remote_code=True)
        embed_model = AutoModel.from_pretrained(
            args.embed_model_name_or_path,
            config=embed_cfg,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        in_features, expansion_ratio = 4096, 1

    elif "mxbai-embed-large-v1" in args.embed_model_name_or_path:
        in_features, expansion_ratio = 1024, 1
        embed_model = SentenceTransformer(
            args.embed_model_name_or_path,
            truncate_dim=in_features,
            trust_remote_code=True,
        )
        embed_model.max_seq_length = 512

    elif args.embed_model_name_or_path == "sentence-transformers/gtr-t5-base":
        embed_model = SentenceTransformer(args.embed_model_name_or_path)
        embed_model.max_seq_length = args.max_length
        in_features, expansion_ratio = 768, 1

    elif any(k in args.embed_model_name_or_path for k in ["Qwen3-Embedding", "gte_Qwen2"]):
        embed_tokenizer = AutoTokenizer.from_pretrained(args.embed_model_name_or_path, padding_side="left")
        embed_model = AutoModel.from_pretrained(
            args.embed_model_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        in_features = embed_model.config.hidden_size
        expansion_ratio = 1

    else:
        raise ValueError("Unsupported embedding model path")

    embed_model.to(device).eval()
    for p in embed_model.parameters():
        p.requires_grad = False

    # --------------------------- Init projector --------------------------- #
    out_features = embedding_layer.weight.shape[1]

    if args.projector_type == "qformer":
        proj_cfg = QFormerProjectorConfig(
            in_features=in_features,
            out_features=out_features,
            expansion_ratio=8,
            num_attention_heads=16,
        )
        Projector = QFormerProjector
    elif args.projector_type == "linear":
        proj_cfg = ProjectorConfig(in_features=in_features, out_features=out_features, expansion_ratio=1)
        Projector = LinearProjector
    else:  # nonlinear (MLP)
        proj_cfg = ProjectorConfig(in_features=in_features, out_features=out_features, expansion_ratio=expansion_ratio)
        Projector = MLPProjector

    projector = (
        Projector.from_pretrained(args.hub_model_id, config=proj_cfg, dtype=torch.bfloat16)
        if args.hub_model_id
        else Projector(config=proj_cfg, dtype=torch.bfloat16)
    )
    projector.to(device)

    # ------------------------------ Dataset ------------------------------- #
    data_files = json.loads(args.dataset_name)
    raw = datasets.load_dataset("json", data_files=data_files)
    full_ds = datasets.concatenate_datasets([raw["train"], raw["validation"]])

    expert_text_to_idx, global_expert_texts = compute_expert_text_mapping(full_ds)

    train_ds = raw["train"].map(
        process_batched,
        fn_kwargs=dict(tokenizer=tokenizer, template_type="qwen", expert_text_to_idx=expert_text_to_idx, max_length=args.max_length),
        batched=True,
        batch_size=args.batch_size,
        num_proc=8,
        input_columns=["question"],
        remove_columns=raw["train"].column_names,
    )
    val_ds = raw["validation"].map(
        process_batched,
        fn_kwargs=dict(tokenizer=tokenizer, template_type="qwen", expert_text_to_idx=expert_text_to_idx, max_length=args.max_length),
        batched=True,
        batch_size=args.batch_size,
        num_proc=8,
        input_columns=["question"],
        remove_columns=raw["validation"].column_names,
    )

    collate_fn = partial(custom_collate_fn, tokenizer=tokenizer)

    if args.local_rank != -1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds, shuffle=False)
        train_dl = torch.utils.data.DataLoader(
            train_ds, batch_size=args.batch_size, sampler=train_sampler,
            collate_fn=collate_fn, drop_last=True
        )
        val_dl = torch.utils.data.DataLoader(
            val_ds, batch_size=args.batch_size, sampler=val_sampler,
            collate_fn=collate_fn
        )
    else:
        train_dl = torch.utils.data.DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_fn, drop_last=True
        )
        val_dl = torch.utils.data.DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_fn
        )

    # ----------------------- Embedding cache handling ---------------------- #
    cache_path = os.path.join(args.output_dir, args.cached_embedding_file)
    rank, world_size = get_rank(), (dist.get_world_size() if dist.is_initialized() else 1)

    if os.path.exists(cache_path):
        cache = torch.load(cache_path, map_location="cpu")
        cached_texts, cached_embeds = cache["global_expert_texts"], cache["embeddings"]
        if rank == 0:
            print(f"[Cache] Loaded {len(cached_texts)} cached expert texts.")
    else:
        cached_texts, cached_embeds = [], None

    old_idx = {t: i for i, t in enumerate(cached_texts)}
    missing_texts, new_text_indices = [], []
    for text in global_expert_texts:
        if text in old_idx:
            new_text_indices.append(old_idx[text])
        else:
            new_text_indices.append(-1)
            missing_texts.append(text)

    if missing_texts:  # Need to compute embeddings for new texts
        total_missing = len(missing_texts)
        slice_size = total_missing // world_size
        remainder = total_missing % world_size

        if rank < remainder:
            start, end = rank * (slice_size + 1), rank * (slice_size + 1) + slice_size + 1
        else:
            start = rank * slice_size + remainder
            end = start + slice_size
        local_missing = missing_texts[start:end]
        print(f"[Rank {rank}] Encoding {len(local_missing)} new expert texts.")

        with torch.no_grad():
            local_embeds = []
            for i in tqdm.tqdm(range(0, len(local_missing), 8), desc=f"Rank {rank}"):
                batch_texts = local_missing[i : i + 8]
                if "NV-Embed-v2" in args.embed_model_name_or_path:
                    emb = embed_model.encode(batch_texts, instruction="", max_length=args.max_length,
                                             convert_to_tensor=True)
                elif any(k in args.embed_model_name_or_path for k in ["Qwen3-Embedding", "gte_Qwen2"]):
                    batch_dict = embed_tokenizer(
                        batch_texts, padding=True, truncation=True, max_length=args.max_length,
                        return_tensors="pt"
                    ).to(embed_model.device)
                    outputs = embed_model(**batch_dict)
                    emb = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
                else:
                    emb = embed_model.encode(batch_texts, convert_to_tensor=True, normalize_embeddings=False)
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
            embed_dim = all_new_embeds.shape[1] if missing_texts else cached_embeds.shape[1]
            new_ordered = []
            ptr = 0
            for idx in new_text_indices:
                new_ordered.append(
                    all_new_embeds[ptr : ptr + 1] if idx == -1 else cached_embeds[idx : idx + 1]
                )
                if idx == -1:
                    ptr += 1
            new_global_embeds = torch.cat(new_ordered, dim=0)
            torch.save({"global_expert_texts": global_expert_texts, "embeddings": new_global_embeds}, cache_path)
            print(f"[Cache] Updated cache saved to {cache_path}.")
    else:
        if rank == 0:
            print("[Cache] No new expert texts; cache unchanged.")

    if dist.is_initialized():
        dist.barrier()

    # --------------------------- DeepSpeed config -------------------------- #
    steps_per_epoch = len(train_dl)
    total_steps = steps_per_epoch * args.num_train_epochs
    total_update_steps = math.ceil(steps_per_epoch / args.gradient_accumulation_steps) * args.num_train_epochs
    warmup_steps = int(0.1 * total_update_steps)

    ds_cfg = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "weight_decay": 0.01,
                "fp32_optimizer_states": False,
                "torch_adam": True,
            },
        },
        "scheduler": {
            "type": "WarmupCosineLR",
            "params": {"warmup_num_steps": warmup_steps, "total_num_steps": total_update_steps},
            "param_schedulers": {
                "projector": {
                    "scheduler": "WarmupCosineLR",
                    "warmup_num_steps": warmup_steps,
                    "total_num_steps": total_update_steps,
                }
            },
        },
        "bf16": {"enabled": True},
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 5e7,
            "overlap_comm": False,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e7,
            "contiguous_gradients": True,
            "round_robin_gradients": False,
            "ignore_unused_parameters": True,
        },
    }
    if args.offload_optimizer == "cpu":
        ds_cfg["zero_optimization"]["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": False,
        }

    composite = CompositeModel(projector, big_model)
    for p in composite.big_model.parameters():
        p.requires_grad = False  # start frozen

    optim_groups = [
        {"params": composite.projector.parameters(), "lr": args.lr, "weight_decay": 0.01, "name": "projector"}
    ]

    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        args=args, model=composite, model_parameters=optim_groups, config=ds_cfg
    )

    proj_dtype = next(composite.projector.parameters()).dtype
    cached_embeds = torch.load(cache_path, map_location="cpu")["embeddings"].to(device, dtype=proj_dtype)

    # Move embedding model to CPU to free VRAM
    embed_model.to("cpu")
    del embed_model

    loss_fn = nn.CrossEntropyLoss()
    pbar = tqdm.tqdm(total=total_steps)

    
    os.makedirs(os.path.join(args.output_dir, args.ckpt_key), exist_ok=True)
    wandb_run = maybe_init_wandb(
        args,
        extra_config={
            "stage": "stage1_query_reconstruction",
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "total_update_steps": total_update_steps,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "offload_optimizer": args.offload_optimizer,
        },
    )

    # ------------------------------- Training ------------------------------ #
    for epoch in range(args.num_train_epochs):

        # Unfreeze LLM if needed
        if epoch == args.llm_unfreeze_epoch:
            for p in composite.big_model.parameters():
                p.requires_grad = True

            full_cfg = copy.deepcopy(ds_cfg)
            remaining_steps = total_steps - epoch * steps_per_epoch
            remaining_update_steps = math.ceil(remaining_steps / args.gradient_accumulation_steps)
            full_cfg["scheduler"]["param_schedulers"]["big_model"] = {
                "scheduler": "WarmupCosineLR",
                "warmup_num_steps": int(0.1 * remaining_update_steps),
                "total_num_steps": remaining_update_steps,
            }
            full_cfg["scheduler"]["param_schedulers"]["projector"]["start_step"] = math.ceil(
                epoch * steps_per_epoch / args.gradient_accumulation_steps
            )

            del model_engine, optimizer, lr_scheduler
            torch.cuda.empty_cache()

            optim_groups = [
                {"params": composite.projector.parameters(), "lr": args.lr, "weight_decay": 0.01, "name": "projector"},
                {"params": composite.big_model.parameters(), "lr": args.llm_lr, "weight_decay": 0.01, "name": "big_model"},
            ]
            model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
                args=args, model=composite, model_parameters=optim_groups, config=full_cfg
            )

        if args.local_rank != -1:
            train_dl.sampler.set_epoch(epoch)

        model_engine.train()
        epoch_losses = []

        for step, batch in enumerate(train_dl):
            with torch.no_grad():
                flat_idx = [int(i) for sub in batch["text_indices"] for i in sub]
                embeds = cached_embeds[torch.tensor(flat_idx, device=device)]
                n, m, d = len(batch["text_indices"]), len(batch["text_indices"][0]), cached_embeds.shape[1]
                last_states = embeds.view(n, m, d)

            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model_engine(last_states, input_ids, attn_mask)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

            model_engine.backward(loss)

            if (not torch.isfinite(loss) or
                any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model_engine.module.parameters())):
                if get_rank() == 0:
                    print(f"[Epoch {epoch} Step {step}] Non-finite loss/grad; skipping batch.")
                model_engine.zero_grad()
                continue

            model_engine.step()
            torch.cuda.empty_cache()

            with torch.no_grad():
                red_loss = loss.clone()
                if dist.is_initialized():
                    dist.all_reduce(red_loss)
                    red_loss /= dist.get_world_size()
                epoch_losses.append(red_loss.item())

            if get_rank() == 0 and step % 5 == 0:
                global_step = epoch * steps_per_epoch + step + 1
                tqdm.tqdm.write(f"[Epoch {epoch}] Step {step} -- Loss: {red_loss:.4f} | "
                                f"Grad-norm: {model_engine.get_global_grad_norm():.4f}")
                wandb_log(
                    wandb_run,
                    {
                        "stage1/train_loss": float(red_loss.item()),
                        "stage1/grad_norm": float(model_engine.get_global_grad_norm()),
                        "stage1/epoch": epoch,
                        "stage1/step_in_epoch": step,
                    },
                    step=global_step,
                )

            # Periodic checkpointing
            if (step + 1) == len(train_dl):
                if get_rank() == 0:
                    save_model_and_configs(args, model_engine, args.ckpt_key, f"epoch_{epoch}")
                    wandb_log(
                        wandb_run,
                        {"stage1/checkpoint_epoch": epoch},
                        step=(epoch + 1) * steps_per_epoch,
                    )

            pbar.update(1)

        # --------------------------- Validation --------------------------- #
        model_engine.eval()
        val_loss_sum, val_steps = 0.0, 0
        with torch.no_grad():
            for vbatch in val_dl:
                flat_idx = [int(i) for sub in vbatch["text_indices"] for i in sub]
                embeds = cached_embeds[torch.tensor(flat_idx, device=device)]
                n, m, d = len(vbatch["text_indices"]), len(vbatch["text_indices"][0]), cached_embeds.shape[1]
                last_states = embeds.view(n, m, d)

                logits = model_engine(
                    last_states,
                    vbatch["input_ids"].to(device),
                    vbatch["attention_mask"].to(device),
                )
                loss = loss_fn(logits.view(-1, logits.size(-1)), vbatch["labels"].to(device).view(-1))
                val_loss_sum += loss.item()
                val_steps += 1

        if get_rank() == 0:
            train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
            val_loss = float(val_loss_sum / (val_steps or 1))
            print(f"\n★ Epoch {epoch} complete. "
                  f"Train-loss: {train_loss:.4f} | "
                  f"Val-loss: {val_loss:.4f}\n")
            wandb_log(
                wandb_run,
                {
                    "stage1/epoch_train_loss": train_loss,
                    "stage1/val_loss": val_loss,
                    "stage1/epoch": epoch,
                },
                step=(epoch + 1) * steps_per_epoch,
            )

    if wandb_run is not None:
        wandb.finish()


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    main()
