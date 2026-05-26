#!/usr/bin/env python
"""Evaluate an ICL-Router stage2 checkpoint on router JSON rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import datasets
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from icl_stage2_routing_training import (
    CompositeModel,
    ProjectorConfig,
    LinearProjector,
    MLPProjector,
    custom_collate_fn,
    process_batched,
)
try:
    import wandb
except ImportError:
    wandb = None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--experts-information-file", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--projector-type", choices=["linear", "nonlinear"], default="nonlinear")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scores-jsonl", type=Path, default=None)
    parser.add_argument("--wandb-project", type=str, default=os.environ.get("WANDB_PROJECT", "C2C_IRL"))
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default=os.environ.get("WANDB_MODE", "online"))
    parser.add_argument("--wandb-tags", type=str, default=os.environ.get("WANDB_TAGS", "stage2_eval"))
    return parser.parse_args()


def qid(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    llm_dir = args.checkpoint / "llm"
    projector_dir = args.checkpoint / "projector"

    tokenizer = AutoTokenizer.from_pretrained(llm_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    big_model = AutoModelForCausalLM.from_pretrained(
        llm_dir,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    )
    big_model.config.use_cache = False
    big_model.to(device).eval()

    cache = torch.load(args.embedding_cache, map_location="cpu")
    cached_texts = list(cache["global_expert_texts"])
    expert_text_to_idx = {text: idx for idx, text in enumerate(cached_texts)}
    cached_embeds = cache["embeddings"]

    proj_cfg = ProjectorConfig(
        in_features=int(cached_embeds.shape[1]),
        out_features=int(big_model.get_input_embeddings().weight.shape[1]),
        expansion_ratio=1,
    )
    projector_cls = LinearProjector if args.projector_type == "linear" else MLPProjector
    target_dtype = big_model.get_input_embeddings().weight.dtype
    projector = projector_cls.from_pretrained(
        projector_dir,
        config=proj_cfg,
        dtype=target_dtype,
    )
    projector.to(device=device, dtype=target_dtype).eval()

    model = CompositeModel(projector, big_model).to(device).eval()
    cached_embeds = cached_embeds.to(device, dtype=target_dtype)

    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("No", add_special_tokens=False)[0]

    experts_information = json.loads(args.experts_information_file.read_text(encoding="utf-8"))
    raw = datasets.load_dataset("json", data_files={"eval": str(args.dataset)})["eval"]
    ds = raw.map(
        process_batched,
        fn_kwargs={
            "tokenizer": tokenizer,
            "template_type": "qwen",
            "expert_info": experts_information,
            "expert_text_to_idx": expert_text_to_idx,
        },
        batched=True,
        batch_size=args.batch_size,
        input_columns=["query", "model", "is_correct_direct", "is_correct", "task", "index"],
        remove_columns=raw.column_names,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: custom_collate_fn(b, tokenizer=tokenizer),
    )

    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="scoring"):
            flat_idx = [int(i) for sl in batch["text_indices"] for i in sl]
            embeds = cached_embeds[torch.tensor(flat_idx, device=device)]
            n = len(batch["text_indices"])
            m = len(batch["text_indices"][0])
            last_states = embeds.view(n, m, cached_embeds.shape[1])

            logits = model(
                last_states,
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            labels = batch["labels"].to(device)
            mask = labels != -100
            ans_pos = mask.float().argmax(dim=1).long()
            b_idx = torch.arange(logits.size(0), device=device)
            score = (logits[b_idx, ans_pos, yes_id] - logits[b_idx, ans_pos, no_id]).float().cpu()

            for i, value in enumerate(score.tolist()):
                rows.append(
                    {
                        "query_id": qid(batch["query"][i]),
                        "query": batch["query"][i],
                        "task": batch["task"][i],
                        "index": int(batch["index"][i]),
                        "model": batch["model"][i],
                        "score": float(value),
                        "pred_yes": bool(value > 0.0),
                        "is_correct_direct": bool(batch["is_correct_direct"][i]),
                    }
                )

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["query_id"]].append(row)

    selected = []
    missing_groups = 0
    for group_rows in grouped.values():
        if not group_rows:
            continue
        if len({r["model"] for r in group_rows}) != len(group_rows):
            missing_groups += 1
        selected.append(max(group_rows, key=lambda r: r["score"]))

    candidates = sorted({row["model"] for row in rows})
    static = {
        name: sum(row["is_correct_direct"] for row in rows if row["model"] == name)
        / max(1, sum(1 for row in rows if row["model"] == name))
        for name in candidates
    }
    selected_correct = sum(row["is_correct_direct"] for row in selected) / max(1, len(selected))
    oracle = sum(any(row["is_correct_direct"] for row in group_rows) for group_rows in grouped.values()) / max(1, len(grouped))
    row_binary_accuracy = sum((row["score"] > 0.0) == row["is_correct_direct"] for row in rows) / max(1, len(rows))

    selection_counts = Counter(row["model"] for row in selected)
    selected_by_model = {}
    for name in candidates:
        subset = [row for row in selected if row["model"] == name]
        selected_by_model[name] = {
            "count": len(subset),
            "fraction": len(subset) / max(1, len(selected)),
            "accuracy": sum(row["is_correct_direct"] for row in subset) / max(1, len(subset)),
            "pred_yes_fraction": sum(row["pred_yes"] for row in subset) / max(1, len(subset)),
        }

    summary = {
        "checkpoint": str(args.checkpoint),
        "num_rows": len(rows),
        "num_queries": len(grouped),
        "missing_or_duplicate_candidate_groups": missing_groups,
        "row_binary_accuracy": row_binary_accuracy,
        "selected_correct": selected_correct,
        "oracle_possible": oracle,
        "static_accuracy": static,
        "selection_counts": dict(selection_counts),
        "selected_by_model": selected_by_model,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.scores_jsonl is not None:
        args.scores_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.scores_jsonl.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.wandb_project and args.wandb_mode != "disabled" and wandb is not None:
        try:
            tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
            run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or f"eval-{args.checkpoint.name}",
                mode=args.wandb_mode,
                tags=tags,
                config={
                    "checkpoint": str(args.checkpoint),
                    "dataset": str(args.dataset),
                    "experts_information_file": str(args.experts_information_file),
                },
            )
            wandb.log(
                {
                    "eval/row_binary_accuracy": row_binary_accuracy,
                    "eval/selected_correct": selected_correct,
                    "eval/oracle_possible": oracle,
                    "eval/num_rows": len(rows),
                    "eval/num_queries": len(grouped),
                    **{f"eval/static_accuracy/{k}": v for k, v in static.items()},
                    **{f"eval/selection_fraction/{k}": v["fraction"] for k, v in selected_by_model.items()},
                    **{f"eval/selected_accuracy/{k}": v["accuracy"] for k, v in selected_by_model.items()},
                }
            )
            run.finish()
        except Exception as exc:
            print(f"[wandb] eval logging failed: {exc}")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
