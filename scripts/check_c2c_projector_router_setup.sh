#!/usr/bin/env bash
set -euo pipefail

# Fast preflight check for C2C projector-router training.
#
# Usage:
#   bash scripts/check_c2c_projector_router_setup.sh
#   ENV_NAME=kjh_c2c bash scripts/check_c2c_projector_router_setup.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ENV_NAME="${ENV_NAME:-${CONDA_DEFAULT_ENV:-route-IRL}}"
DATA_DIR="${DATA_DIR:-data/c2c_projectors_p123}"

required_files=(
  "$DATA_DIR/question_train.json"
  "$DATA_DIR/question_test.json"
  "$DATA_DIR/train_router.json"
  "$DATA_DIR/test_router.json"
  "$DATA_DIR/experts_information_500.json"
)

echo "[$(date '+%F %T')] repo: $ROOT_DIR"
echo "[$(date '+%F %T')] env: $ENV_NAME"
echo "[$(date '+%F %T')] data_dir: $DATA_DIR"

missing=0
for file in "${required_files[@]}"; do
  if [[ -s "$file" ]]; then
    printf '  OK   %s (%s)\n' "$file" "$(du -h "$file" | awk '{print $1}')"
  else
    printf '  MISS %s\n' "$file" >&2
    missing=1
  fi
done
if [[ "$missing" == "1" ]]; then
  exit 1
fi

if [[ -s "$DATA_DIR/SHA256SUMS" ]] && command -v sha256sum >/dev/null 2>&1; then
  echo "[$(date '+%F %T')] checking data SHA256SUMS"
  (cd "$DATA_DIR" && sha256sum -c SHA256SUMS)
fi

C2C_ROUTER_DATA_DIR="$DATA_DIR" \
conda run --no-capture-output -n "$ENV_NAME" python - <<'PY'
import json
import os
import pathlib
from collections import Counter, defaultdict

import torch
import deepspeed
import transformers
import datasets
import sentence_transformers
import peft

base = pathlib.Path(os.environ["C2C_ROUTER_DATA_DIR"])
print("python packages:")
print("  torch", torch.__version__, "cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available())
print("  deepspeed", deepspeed.__version__)
print("  transformers", transformers.__version__)
print("  peft", peft.__version__)
print("  datasets", datasets.__version__)
print("  sentence_transformers", sentence_transformers.__version__)
if torch.cuda.is_available():
    print("gpus:")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  {i}: {props.name}, {props.total_memory / 1024**3:.1f} GiB")

print("data:")
for name in ["question_train.json", "question_test.json", "train_router.json", "test_router.json"]:
    rows = json.load(open(base / name, encoding="utf-8"))
    print(f"  {name}: {len(rows)} rows")

experts = json.load(open(base / "experts_information_500.json", encoding="utf-8"))
print("  experts:", ", ".join(f"{k}={len(v)}" for k, v in experts.items()))

for name in ["train_router.json", "test_router.json"]:
    rows = json.load(open(base / name, encoding="utf-8"))
    by_query = defaultdict(list)
    for row in rows:
        by_query[row["query"]].append(row)
    yes_dist = Counter(sum(bool(r["is_correct_direct"]) for r in group) for group in by_query.values())
    print(f"  {name} query_groups={len(by_query)} yes_count_dist={dict(sorted(yes_dist.items()))}")
PY

echo "[$(date '+%F %T')] setup check passed"
