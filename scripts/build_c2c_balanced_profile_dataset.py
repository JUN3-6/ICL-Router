#!/usr/bin/env python
"""Build a C2C router dataset with a balanced fixed profile set.

The output keeps ICL-Router's candidate-independent Yes/No training format:

* experts_information_500.json contains the same 500 profile queries for every
  candidate, with each candidate's own Yes/No label.
* train_router.json and test_router.json contain target queries that do not
  overlap the profile queries.
* all-correct and all-wrong target groups are excluded by default so the router
  is trained and validated on discriminative routing decisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CANDIDATES = ["receiver", "behavior_p1", "behavior_p2", "behavior_p3"]
SINGLE_PATTERNS = {
    "receiver_only": (1, 0, 0, 0),
    "behavior_p1_only": (0, 1, 0, 0),
    "behavior_p2_only": (0, 0, 1, 0),
    "behavior_p3_only": (0, 0, 0, 1),
}


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def query_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (row["query"], row.get("task", ""), int(row.get("index", -1)))


def query_id(key: tuple[str, str, int]) -> str:
    return hashlib.sha1(json.dumps(key, ensure_ascii=False).encode("utf-8")).hexdigest()


def pattern_for_rows(rows: list[dict[str, Any]]) -> tuple[int, ...]:
    by_model = {row["model"]: bool(row["is_correct_direct"]) for row in rows}
    missing = [name for name in CANDIDATES if name not in by_model]
    if missing:
        raise ValueError(f"Missing candidate rows for query: {missing}")
    return tuple(int(by_model[name]) for name in CANDIDATES)


def load_grouped_rows(source_dir: Path) -> dict[tuple[str, str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for filename in ["train_router.json", "test_router.json"]:
        for row in read_json(source_dir / filename):
            if row["model"] not in CANDIDATES:
                continue
            grouped[query_key(row)].append(row)

    out = {}
    for key, rows in grouped.items():
        if len(rows) != len(CANDIDATES):
            continue
        by_model = {row["model"]: row for row in rows}
        out[key] = [by_model[name] for name in CANDIDATES]
    return out


def task_counts(keys, grouped):
    return Counter(grouped[key][0].get("task", "") for key in keys)


def pattern_counts(keys, grouped):
    return Counter("".join(map(str, pattern_for_rows(grouped[key]))) for key in keys)


def select_from_bucket(bucket, n, rng):
    items = list(bucket)
    rng.shuffle(items)
    return items[:n], items[n:]


def greedy_fill(keys, grouped, already_selected, target_total, rng):
    selected = list(already_selected)
    selected_set = set(selected)
    candidates = [key for key in keys if key not in selected_set]
    rng.shuffle(candidates)

    target_yes = {name: int(round(target_total * 0.42)) for name in CANDIDATES}
    all_tasks = sorted({grouped[key][0].get("task", "") for key in keys})
    target_task = {task: max(1, target_total // max(1, len(all_tasks))) for task in all_tasks}

    yes_counts = Counter()
    for key in selected:
        pat = pattern_for_rows(grouped[key])
        for idx, bit in enumerate(pat):
            yes_counts[CANDIDATES[idx]] += bit
    selected_task_counts = task_counts(selected, grouped)

    while len(selected) < target_total and candidates:
        best_idx = None
        best_score = None
        for idx, key in enumerate(candidates):
            pat = pattern_for_rows(grouped[key])
            task = grouped[key][0].get("task", "")

            before_yes = sum(abs(yes_counts[name] - target_yes[name]) for name in CANDIDATES)
            after_yes = sum(
                abs(yes_counts[name] + pat[cidx] - target_yes[name])
                for cidx, name in enumerate(CANDIDATES)
            )
            before_task = abs(selected_task_counts[task] - target_task.get(task, 0))
            after_task = abs(selected_task_counts[task] + 1 - target_task.get(task, 0))
            score = (after_yes - before_yes) * 3.0 + (after_task - before_task) * 0.5
            score += rng.random() * 0.01
            if best_score is None or score < best_score:
                best_score = score
                best_idx = idx

        key = candidates.pop(best_idx)
        selected.append(key)
        pat = pattern_for_rows(grouped[key])
        for cidx, name in enumerate(CANDIDATES):
            yes_counts[name] += pat[cidx]
        selected_task_counts[grouped[key][0].get("task", "")] += 1

    if len(selected) != target_total:
        raise RuntimeError(f"Could only select {len(selected)} profile queries, target={target_total}")
    return selected


def candidate_yes_counts(keys, grouped):
    yes_counts = Counter()
    for key in keys:
        pat = pattern_for_rows(grouped[key])
        for idx, bit in enumerate(pat):
            yes_counts[CANDIDATES[idx]] += bit
    return yes_counts


def imbalance_score(yes_counts, target_min, target_max):
    score = 0
    for name in CANDIDATES:
        if yes_counts[name] < target_min:
            score += (target_min - yes_counts[name]) ** 2
        elif yes_counts[name] > target_max:
            score += (yes_counts[name] - target_max) ** 2
    return score


def balance_profile_by_swaps(selected, candidate_pool, grouped, rng, min_rate=0.40, max_rate=0.60):
    selected = list(selected)
    selected_set = set(selected)
    unselected = [key for key in candidate_pool if key not in selected_set]
    rng.shuffle(unselected)

    target_min = int(math.ceil(len(selected) * min_rate))
    target_max = int(math.floor(len(selected) * max_rate))
    yes_counts = candidate_yes_counts(selected, grouped)
    current_score = imbalance_score(yes_counts, target_min, target_max)

    for _ in range(120):
        if current_score == 0:
            break
        low_indices = [
            idx for idx, name in enumerate(CANDIDATES)
            if yes_counts[name] < target_min
        ]
        high_indices = [
            idx for idx, name in enumerate(CANDIDATES)
            if yes_counts[name] > target_max
        ]
        if not low_indices and not high_indices:
            break
        best = None
        best_score = current_score
        out_candidates = [
            (idx, key) for idx, key in enumerate(selected)
            if (
                any(pattern_for_rows(grouped[key])[low_idx] == 0 for low_idx in low_indices)
                or any(pattern_for_rows(grouped[key])[high_idx] == 1 for high_idx in high_indices)
            )
        ]
        in_candidates = [
            (idx, key) for idx, key in enumerate(unselected)
            if (
                any(pattern_for_rows(grouped[key])[low_idx] == 1 for low_idx in low_indices)
                or any(pattern_for_rows(grouped[key])[high_idx] == 0 for high_idx in high_indices)
            )
        ]
        def out_potential(item):
            pat = pattern_for_rows(grouped[item[1]])
            return (
                sum(1 for high_idx in high_indices if pat[high_idx] == 1)
                + sum(1 for low_idx in low_indices if pat[low_idx] == 0)
            )

        def in_potential(item):
            pat = pattern_for_rows(grouped[item[1]])
            return (
                sum(1 for high_idx in high_indices if pat[high_idx] == 0)
                + sum(1 for low_idx in low_indices if pat[low_idx] == 1)
            )

        out_candidates = sorted(out_candidates, key=out_potential, reverse=True)[:256]
        in_candidates = sorted(in_candidates, key=in_potential, reverse=True)[:256]
        for out_idx, out_key in out_candidates:
            out_pat = pattern_for_rows(grouped[out_key])
            for in_idx, in_key in in_candidates:
                in_pat = pattern_for_rows(grouped[in_key])
                improves_low = any(in_pat[low_idx] > out_pat[low_idx] for low_idx in low_indices)
                improves_high = any(in_pat[high_idx] < out_pat[high_idx] for high_idx in high_indices)
                if not (improves_low or improves_high):
                    continue
                new_counts = yes_counts.copy()
                for cidx, name in enumerate(CANDIDATES):
                    new_counts[name] += in_pat[cidx] - out_pat[cidx]
                score = imbalance_score(new_counts, target_min, target_max)
                if score < best_score:
                    best_score = score
                    best = (out_idx, in_idx, new_counts)
                    if score == 0:
                        break
            if best_score == 0:
                break
        if best is None:
            break
        out_idx, in_idx, yes_counts = best
        old_key = selected[out_idx]
        new_key = unselected[in_idx]
        selected[out_idx] = new_key
        unselected[in_idx] = old_key
        current_score = best_score

    return selected


def build_profile_keys(grouped, profile_size: int, seed: int):
    rng = random.Random(seed)
    by_pattern = defaultdict(list)
    challenge_keys = []
    for key, rows in grouped.items():
        pat = pattern_for_rows(rows)
        if sum(pat) in (0, len(CANDIDATES)):
            continue
        challenge_keys.append(key)
        by_pattern[pat].append(key)

    selected = []
    selected_by_rule = Counter()

    single_target = min(80, profile_size // max(1, len(SINGLE_PATTERNS)))
    for name, pat in SINGLE_PATTERNS.items():
        picked, _ = select_from_bucket(by_pattern[pat], single_target, rng)
        selected.extend(picked)
        selected_by_rule[name] += len(picked)

    multi_patterns = [
        pat for pat in sorted(by_pattern) if sum(pat) in (2, 3)
    ]
    per_multi = 12
    for pat in multi_patterns:
        available = [key for key in by_pattern[pat] if key not in set(selected)]
        picked, _ = select_from_bucket(available, min(per_multi, len(available)), rng)
        selected.extend(picked)
        selected_by_rule["multi_pattern_seed"] += len(picked)

    if len(selected) > profile_size:
        selected = selected[:profile_size]
    selected = greedy_fill(challenge_keys, grouped, selected, profile_size, rng)
    selected = balance_profile_by_swaps(selected, challenge_keys, grouped, rng)
    return selected, selected_by_rule


def flatten_groups(keys, grouped):
    rows = []
    for key in keys:
        rows.extend(grouped[key])
    return rows


def split_remaining(keys, grouped, val_fraction, seed):
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for key in keys:
        pat = "".join(map(str, pattern_for_rows(grouped[key])))
        task = grouped[key][0].get("task", "")
        buckets[(pat, task)].append(key)

    train_keys, val_keys = [], []
    for bucket_keys in buckets.values():
        rng.shuffle(bucket_keys)
        n_val = max(1, int(round(len(bucket_keys) * val_fraction))) if len(bucket_keys) >= 5 else max(0, int(round(len(bucket_keys) * val_fraction)))
        val_keys.extend(bucket_keys[:n_val])
        train_keys.extend(bucket_keys[n_val:])

    rng.shuffle(train_keys)
    rng.shuffle(val_keys)
    return train_keys, val_keys


def yesno(flag: bool) -> str:
    return "Yes" if flag else "No"


def build_experts_information(profile_keys, grouped):
    experts = {name: [] for name in CANDIDATES}
    for key in profile_keys:
        rows = grouped[key]
        by_model = {row["model"]: row for row in rows}
        for name in CANDIDATES:
            row = by_model[name]
            experts[name].append(
                {
                    "input": row["query"],
                    "label": yesno(bool(row["is_correct_direct"])),
                    "task": row.get("task", ""),
                }
            )
    return experts


def summarize(keys, grouped):
    yes_counts = Counter()
    for key in keys:
        pat = pattern_for_rows(grouped[key])
        for idx, bit in enumerate(pat):
            yes_counts[CANDIDATES[idx]] += bit
    return {
        "num_queries": len(keys),
        "num_rows": len(keys) * len(CANDIDATES),
        "pattern_counts": dict(sorted(pattern_counts(keys, grouped).items())),
        "task_counts": dict(sorted(task_counts(keys, grouped).items())),
        "candidate_yes_counts": dict(yes_counts),
        "candidate_yes_rate": {
            name: yes_counts[name] / max(1, len(keys))
            for name in CANDIDATES
        },
    }


def write_sha256s(out_dir: Path):
    lines = []
    for path in sorted(out_dir.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS":
            h = hashlib.sha256(path.read_bytes()).hexdigest()
            lines.append(f"{h}  {path.name}\n")
    (out_dir / "SHA256SUMS").write_text("".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("data/c2c_projectors_p123"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/c2c_projectors_p123_balanced_profile500"))
    parser.add_argument("--profile-size", type=int, default=500)
    parser.add_argument("--target-size", type=int, default=1500,
                        help="Number of non-profile routing queries for train+validation. Set 0 to keep all.")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite {args.output_dir}; pass --overwrite")
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    grouped = load_grouped_rows(args.source_dir)
    profile_keys, profile_rules = build_profile_keys(grouped, args.profile_size, args.seed)
    profile_set = set(profile_keys)

    challenge_remaining = [
        key for key, rows in grouped.items()
        if key not in profile_set and 0 < sum(pattern_for_rows(rows)) < len(CANDIDATES)
    ]
    if args.target_size > 0 and len(challenge_remaining) > args.target_size:
        challenge_remaining = greedy_fill(challenge_remaining, grouped, [], args.target_size, random.Random(args.seed + 7))
        challenge_remaining = balance_profile_by_swaps(
            challenge_remaining,
            [key for key, rows in grouped.items() if key not in profile_set and 0 < sum(pattern_for_rows(rows)) < len(CANDIDATES)],
            grouped,
            random.Random(args.seed + 8),
        )
    train_keys, val_keys = split_remaining(challenge_remaining, grouped, args.val_fraction, args.seed + 1)

    experts = build_experts_information(profile_keys, grouped)
    train_rows = flatten_groups(train_keys, grouped)
    val_rows = flatten_groups(val_keys, grouped)
    question_train = [{"question": grouped[key][0]["query"], "answer": ""} for key in train_keys]
    question_test = [{"question": grouped[key][0]["query"], "answer": ""} for key in val_keys]

    write_json(args.output_dir / "experts_information_500.json", experts)
    write_json(args.output_dir / "train_router.json", train_rows)
    write_json(args.output_dir / "test_router.json", val_rows)
    write_json(args.output_dir / "question_train.json", question_train)
    write_json(args.output_dir / "question_test.json", question_test)

    metadata = {
        "source": str(args.source_dir),
        "mode": "balanced_profile500_challenging_targets",
        "seed": args.seed,
        "profile_size": args.profile_size,
        "target_size": args.target_size,
        "val_fraction": args.val_fraction,
        "candidates": CANDIDATES,
        "profile_selection_rules": dict(profile_rules),
        "profile": summarize(profile_keys, grouped),
        "train": summarize(train_keys, grouped),
        "validation": summarize(val_keys, grouped),
        "overlap_checks": {
            "profile_train": len(set(profile_keys) & set(train_keys)),
            "profile_validation": len(set(profile_keys) & set(val_keys)),
            "train_validation": len(set(train_keys) & set(val_keys)),
        },
        "profile_query_ids": [query_id(key) for key in profile_keys],
        "train_query_ids": [query_id(key) for key in train_keys],
        "validation_query_ids": [query_id(key) for key in val_keys],
    }
    write_json(args.output_dir / "metadata.json", metadata)
    write_sha256s(args.output_dir)

    print(f"Wrote {args.output_dir}")
    print(json.dumps({
        "profile": metadata["profile"],
        "train": metadata["train"],
        "validation": metadata["validation"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
