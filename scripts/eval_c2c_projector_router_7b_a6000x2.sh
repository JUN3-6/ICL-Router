#!/usr/bin/env bash
set -euo pipefail

# Evaluate a stage2 checkpoint produced by train_c2c_projector_router_7b_a6000x2.sh.
#
# Usage:
#   bash scripts/eval_c2c_projector_router_7b_a6000x2.sh 0
#   CHECKPOINT=/path/to/final_step4220 bash scripts/eval_c2c_projector_router_7b_a6000x2.sh 0

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GPU="${1:-${GPU:-0}}"
export CUDA_VISIBLE_DEVICES="$GPU"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

ENV_NAME="${ENV_NAME:-${CONDA_DEFAULT_ENV:-route-IRL}}"
EMBED_MODEL="${EMBED_MODEL:-Qwen/Qwen3-Embedding-8B}"
DATA_DIR="${DATA_DIR:-data/c2c_projectors_p123_mcq_challenging_router}"
OUT_DIR="${OUT_DIR:-checkpoints_c2c_projectors_p123_mcq_challenging_qwen25_7b_a6000x2}"
STAGE2_OUT_DIR="${STAGE2_OUT_DIR:-$OUT_DIR/stage2}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-5}"
STAGE2_PROJ_LR="${STAGE2_PROJ_LR:-1e-5}"
STAGE2_LLM_LR="${STAGE2_LLM_LR:-2e-6}"
SEED="${SEED:-42}"
LOG_DIR="${LOG_DIR:-logs/c2c_projector_router_7b_a6000x2}"
EVAL_DIR="${EVAL_DIR:-eval/c2c_projector_router_7b_a6000x2}"

save_key="nonlinear_router_projLR${STAGE2_PROJ_LR}_llmLR${STAGE2_LLM_LR}_epochs${STAGE2_EPOCHS}_${EMBED_MODEL##*/}_seed${SEED}"
default_checkpoint="$(find "$STAGE2_OUT_DIR/$save_key" -maxdepth 1 -type d -name 'final_step*' 2>/dev/null | sort -V | tail -n 1 || true)"
CHECKPOINT="${CHECKPOINT:-$default_checkpoint}"

if [[ -z "$CHECKPOINT" || ! -s "$CHECKPOINT/llm/config.json" ]]; then
  echo "missing stage2 checkpoint. Set CHECKPOINT=/path/to/final_step... or check $STAGE2_OUT_DIR/$save_key" >&2
  exit 1
fi

mkdir -p "$LOG_DIR" "$EVAL_DIR"

conda run --no-capture-output -n "$ENV_NAME" python eval_stage2_router.py \
  --checkpoint "$CHECKPOINT" \
  --experts-information-file "$DATA_DIR/experts_information_500.json" \
  --dataset "$DATA_DIR/test_router.json" \
  --embedding-cache "$STAGE2_OUT_DIR/${EMBED_MODEL##*/}_stage2_c2c_p123_profile500.pt" \
  --projector-type nonlinear \
  --batch-size "${EVAL_BATCH_SIZE:-4}" \
  --output "$EVAL_DIR/summary.json" \
  --scores-jsonl "$EVAL_DIR/scores.jsonl" \
  --wandb-project "${WANDB_PROJECT:-C2C_IRL}" \
  --wandb-run-name "${WANDB_RUN_NAME:-c2c-p123-stage2-qwen25-7b-a6000x2-eval}" \
  --wandb-mode "${WANDB_MODE:-online}" \
  --wandb-tags "${WANDB_TAGS:-paper_code,c2c_projector_router,p123,7b,a6000x2,eval}" \
  2>&1 | tee "$LOG_DIR/eval_qwen25_7b_a6000x2.log"
