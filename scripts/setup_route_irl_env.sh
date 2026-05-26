#!/usr/bin/env bash
set -euo pipefail

# Create the conda environment used by the C2C ICL-Router scripts.
#
# Usage:
#   bash scripts/setup_route_irl_env.sh
#
# Optional overrides:
#   ENV_NAME=route-IRL
#   PYTHON_VERSION=3.10
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ENV_NAME="${ENV_NAME:-route-IRL}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_VERSION="${TORCH_VERSION:-2.4.0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found in PATH. Install Miniconda/Anaconda first." >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "[$(date '+%F %T')] conda env already exists: $ENV_NAME"
else
  echo "[$(date '+%F %T')] creating conda env: $ENV_NAME"
  conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION" pip
fi

echo "[$(date '+%F %T')] installing base Python build tools"
conda run --no-capture-output -n "$ENV_NAME" \
  python -m pip install --upgrade pip setuptools wheel packaging ninja

echo "[$(date '+%F %T')] installing torch==$TORCH_VERSION from $TORCH_INDEX_URL"
conda run --no-capture-output -n "$ENV_NAME" \
  python -m pip install --index-url "$TORCH_INDEX_URL" "torch==$TORCH_VERSION"

tmp_req="$(mktemp)"
trap 'rm -f "$tmp_req"' EXIT
grep -vE '^torch==' requirements.txt > "$tmp_req"

echo "[$(date '+%F %T')] installing remaining requirements"
conda run --no-capture-output -n "$ENV_NAME" \
  python -m pip install -r "$tmp_req"

echo "[$(date '+%F %T')] checking imports"
conda run --no-capture-output -n "$ENV_NAME" python - <<'PY'
import torch
import deepspeed
import transformers
print("torch", torch.__version__, "cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available())
print("deepspeed", deepspeed.__version__)
print("transformers", transformers.__version__)
PY

echo "[$(date '+%F %T')] done. Activate with: conda activate $ENV_NAME"
