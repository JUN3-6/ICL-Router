#!/usr/bin/env python
"""Precompute or complete the stage2 expert-text embedding cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    seq_len = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), seq_len]


def load_profile_texts(experts_information_file: Path) -> list[str]:
    experts_information = json.loads(experts_information_file.read_text(encoding="utf-8"))
    first_expert = next(iter(experts_information.values()))
    return list(dict.fromkeys(ex["input"] for ex in first_expert))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experts-information-file", type=Path, required=True)
    parser.add_argument("--embed-model-name-or-path", type=str, required=True)
    parser.add_argument("--cache-file", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()

    profile_texts = load_profile_texts(args.experts_information_file)
    cached_texts: list[str] = []
    cached_embeds = None

    if args.cache_file.exists():
        cache = torch.load(args.cache_file, map_location="cpu")
        cached_texts = list(cache["global_expert_texts"])
        cached_embeds = cache["embeddings"]

    old_idx = {text: idx for idx, text in enumerate(cached_texts)}
    missing_texts = [text for text in profile_texts if text not in old_idx]

    if not missing_texts:
        print(f"[Cache] Complete: {args.cache_file} ({len(profile_texts)} texts)")
        return

    print(
        f"[Cache] Completing {args.cache_file}: "
        f"{len(cached_texts)} cached, {len(missing_texts)} missing, {len(profile_texts)} target"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.embed_model_name_or_path, padding_side="left")
    model = AutoModel.from_pretrained(
        args.embed_model_name_or_path,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device).eval()

    new_embeds = []
    with torch.no_grad():
        for start in tqdm(range(0, len(missing_texts), args.batch_size), desc="embedding missing texts"):
            batch_texts = missing_texts[start : start + args.batch_size]
            batch = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            ).to(device)
            outputs = model(**batch)
            embeds = last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
            new_embeds.append(embeds.float().cpu())

    new_embeds_tensor = torch.cat(new_embeds, dim=0)
    missing_to_embed = dict(zip(missing_texts, new_embeds_tensor))

    ordered_embeds = []
    for text in profile_texts:
        if text in old_idx:
            ordered_embeds.append(cached_embeds[old_idx[text] : old_idx[text] + 1].float())
        else:
            ordered_embeds.append(missing_to_embed[text].unsqueeze(0))

    args.cache_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "global_expert_texts": profile_texts,
            "embeddings": torch.cat(ordered_embeds, dim=0),
        },
        args.cache_file,
    )
    print(f"[Cache] Saved complete cache: {args.cache_file} ({len(profile_texts)} texts)")


if __name__ == "__main__":
    main()
