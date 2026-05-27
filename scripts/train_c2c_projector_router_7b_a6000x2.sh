#!/usr/bin/env bash
set -euo pipefail

# End-to-end ICL-Router paper-code training for C2C projector routing.
# Target machine: 2x A6000 48GB. No training is started unless this script is run.
#
# Expected data directory:
#   data/c2c_projectors_p123/
#     question_train.json
#     question_test.json
#     train_router.json
#     test_router.json
#     experts_information_500.json
#
# Usage:
#   bash scripts/train_c2c_projector_router_7b_a6000x2.sh 0,1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GPUS_STRING="${1:-${GPUS:-0,1}}"
IFS=',' read -r -a GPUS <<< "$GPUS_STRING"
NUM_GPUS="${#GPUS[@]}"

export CUDA_VISIBLE_DEVICES="$GPUS_STRING"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

ENV_NAME="${ENV_NAME:-${CONDA_DEFAULT_ENV:-route-IRL}}"
ROUTER_MODEL="${ROUTER_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
EMBED_MODEL="${EMBED_MODEL:-Qwen/Qwen3-Embedding-8B}"
DATA_DIR="${DATA_DIR:-data/c2c_projectors_p123}"
OUT_DIR="${OUT_DIR:-checkpoints_c2c_projectors_p123_alltrain_qwen25_7b_lora_a6000x2}"
LOG_DIR="${LOG_DIR:-logs/c2c_projector_router_7b_a6000x2}"

STAGE1_KEY="${STAGE1_KEY:-icl_stage1_qwen25_7b_lora_projector_p123_alltrain}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-3}"
STAGE1_BATCH_SIZE="${STAGE1_BATCH_SIZE:-1}"
STAGE1_GRAD_ACCUM="${STAGE1_GRAD_ACCUM:-8}"
STAGE1_MAX_LENGTH="${STAGE1_MAX_LENGTH:-1024}"
STAGE1_PROJ_LR="${STAGE1_PROJ_LR:-2e-5}"
STAGE1_LLM_LR="${STAGE1_LLM_LR:-5e-6}"
STAGE1_UNFREEZE_EPOCH="${STAGE1_UNFREEZE_EPOCH:-999}"
OFFLOAD_OPTIMIZER="${OFFLOAD_OPTIMIZER:-none}"
ZERO_STAGE="${ZERO_STAGE:-2}"
USE_LORA="${USE_LORA:-1}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"

STAGE2_OUT_DIR="${STAGE2_OUT_DIR:-$OUT_DIR/stage2}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-5}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-2}"
STAGE2_GRAD_ACCUM="${STAGE2_GRAD_ACCUM:-16}"
STAGE2_EVAL_STEPS="${STAGE2_EVAL_STEPS:-50}"
STAGE2_EVAL_MAX_BATCHES="${STAGE2_EVAL_MAX_BATCHES:-128}"
STAGE2_MAX_LENGTH="${STAGE2_MAX_LENGTH:-1024}"
STAGE2_PROJ_LR="${STAGE2_PROJ_LR:-1e-5}"
STAGE2_LLM_LR="${STAGE2_LLM_LR:-2e-6}"
STAGE2_FREEZE_LLM="${STAGE2_FREEZE_LLM:-0}"
SEED="${SEED:-42}"

WANDB_PROJECT="${WANDB_PROJECT:-C2C_IRL}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_TAGS="${WANDB_TAGS:-paper_code,c2c_projector_router,p123,7b,a6000x2}"
WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-alltrain,lora}"
STAGE1_WANDB_RUN_NAME="${STAGE1_WANDB_RUN_NAME:-c2c-p123-alltrain-stage1-qwen25-7b-lora-a6000x2}"
STAGE2_WANDB_RUN_NAME="${STAGE2_WANDB_RUN_NAME:-c2c-p123-alltrain-stage2-qwen25-7b-lora-a6000x2}"

required_files=(
  "$DATA_DIR/question_train.json"
  "$DATA_DIR/question_test.json"
  "$DATA_DIR/train_router.json"
  "$DATA_DIR/test_router.json"
  "$DATA_DIR/experts_information_500.json"
)

for file in "${required_files[@]}"; do
  if [[ ! -s "$file" ]]; then
    echo "missing required data file: $file" >&2
    exit 1
  fi
done

mkdir -p "$OUT_DIR" "$STAGE2_OUT_DIR" "$LOG_DIR"

run_ds() {
  conda run --no-capture-output -n "$ENV_NAME" deepspeed --num_gpus "$NUM_GPUS" "$@"
}

stage1_last_epoch=$((STAGE1_EPOCHS - 1))
stage1_llm="$OUT_DIR/$STAGE1_KEY/epoch_${stage1_last_epoch}/llm"
stage1_projector="$OUT_DIR/$STAGE1_KEY/epoch_${stage1_last_epoch}/projector"
stage1_llm_config="$stage1_llm/config.json"
stage1_lora_config="$stage1_llm/adapter_config.json"
stage1_projector_weights="$stage1_projector/model.safetensors"

echo "[$(date '+%F %T')] GPUs=$GPUS_STRING NUM_GPUS=$NUM_GPUS"
echo "[$(date '+%F %T')] router_model=$ROUTER_MODEL embed_model=$EMBED_MODEL"
echo "[$(date '+%F %T')] data_dir=$DATA_DIR"
echo "[$(date '+%F %T')] output_dir=$OUT_DIR"
echo "[$(date '+%F %T')] use_lora=$USE_LORA lora_r=$LORA_R stage2_freeze_llm=$STAGE2_FREEZE_LLM zero_stage=$ZERO_STAGE"
echo "[$(date '+%F %T')] stage2_batch=$STAGE2_BATCH_SIZE grad_accum=$STAGE2_GRAD_ACCUM eval_steps=$STAGE2_EVAL_STEPS eval_max_batches=$STAGE2_EVAL_MAX_BATCHES"

lora_args=()
if [[ "$USE_LORA" == "1" ]]; then
  lora_args+=(
    --use_lora
    --lora_r "$LORA_R"
    --lora_alpha "$LORA_ALPHA"
    --lora_dropout "$LORA_DROPOUT"
    --lora_target_modules "$LORA_TARGET_MODULES"
  )
fi

if [[ "${SKIP_STAGE1:-0}" != "1" && ( ! -s "$stage1_llm_config" || ! -s "$stage1_projector_weights" ) ]]; then
  echo "[$(date '+%F %T')] starting stage1 query reconstruction"
  run_ds icl_stage1_query_rec_training.py \
    --base_model_name_or_path "$ROUTER_MODEL" \
    --embed_model_name_or_path "$EMBED_MODEL" \
    --dataset_name "{\"train\":\"$DATA_DIR/question_train.json\",\"validation\":\"$DATA_DIR/question_test.json\"}" \
    --max_length "$STAGE1_MAX_LENGTH" \
    --batch_size "$STAGE1_BATCH_SIZE" \
    --gradient_accumulation_steps "$STAGE1_GRAD_ACCUM" \
    --offload_optimizer "$OFFLOAD_OPTIMIZER" \
    --zero_stage "$ZERO_STAGE" \
    --lr "$STAGE1_PROJ_LR" \
    --llm_lr "$STAGE1_LLM_LR" \
    --llm_unfreeze_epoch "$STAGE1_UNFREEZE_EPOCH" \
    --num_train_epochs "$STAGE1_EPOCHS" \
    "${lora_args[@]}" \
    --projector_type nonlinear \
    --output_dir "$OUT_DIR" \
    --ckpt_key "$STAGE1_KEY" \
    --cached_embedding_file "${EMBED_MODEL##*/}_stage1_c2c_p123.pt" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_mode "$WANDB_MODE" \
    --wandb_tags "$WANDB_TAGS,$WANDB_EXTRA_TAGS,stage1" \
    --wandb_run_name "$STAGE1_WANDB_RUN_NAME" \
    2>&1 | tee "$LOG_DIR/stage1_qwen25_7b_a6000x2.log"
else
  echo "[$(date '+%F %T')] skipping stage1; found $stage1_llm"
fi

if [[ ! -s "$stage1_llm_config" || ! -s "$stage1_projector_weights" ]]; then
  echo "stage1 checkpoint is incomplete: $stage1_llm / $stage1_projector" >&2
  exit 1
fi
if [[ "$USE_LORA" == "1" && ! -s "$stage1_lora_config" ]]; then
  echo "stage1 LoRA checkpoint is incomplete: $stage1_lora_config" >&2
  exit 1
fi

echo "[$(date '+%F %T')] starting stage2 router training"
stage2_freeze_args=()
if [[ "$STAGE2_FREEZE_LLM" == "1" ]]; then
  stage2_freeze_args+=(--freeze_llm)
fi
stage2_base_model="$stage1_llm"
stage2_lora_args=()
if [[ "$USE_LORA" == "1" ]]; then
  stage2_base_model="$ROUTER_MODEL"
  stage2_lora_args+=(
    --use_lora
    --lora_adapter_path "$stage1_llm"
    --lora_r "$LORA_R"
    --lora_alpha "$LORA_ALPHA"
    --lora_dropout "$LORA_DROPOUT"
    --lora_target_modules "$LORA_TARGET_MODULES"
  )
fi
run_ds icl_stage2_routing_training.py \
  --base_model_name_or_path "$stage2_base_model" \
  --embed_model_name_or_path "$EMBED_MODEL" \
  --projector_path "$stage1_projector" \
  --experts_information_file "$DATA_DIR/experts_information_500.json" \
  --dataset_name "{\"train\":\"$DATA_DIR/train_router.json\",\"validation\":\"$DATA_DIR/test_router.json\"}" \
  --output_dir "$STAGE2_OUT_DIR" \
  --batch_size "$STAGE2_BATCH_SIZE" \
  --gradient_accumulation_steps "$STAGE2_GRAD_ACCUM" \
  --eval_steps "$STAGE2_EVAL_STEPS" \
  --eval_max_batches "$STAGE2_EVAL_MAX_BATCHES" \
  --offload_optimizer "$OFFLOAD_OPTIMIZER" \
  --zero_stage "$ZERO_STAGE" \
  --max_length "$STAGE2_MAX_LENGTH" \
  --proj_lr "$STAGE2_PROJ_LR" \
  --llm_lr "$STAGE2_LLM_LR" \
  "${stage2_freeze_args[@]}" \
  "${stage2_lora_args[@]}" \
  --num_train_epochs "$STAGE2_EPOCHS" \
  --projector_type nonlinear \
  --cached_embedding_file "${EMBED_MODEL##*/}_stage2_c2c_p123_profile500.pt" \
  --seed "$SEED" \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_mode "$WANDB_MODE" \
  --wandb_tags "$WANDB_TAGS,$WANDB_EXTRA_TAGS,stage2" \
  --wandb_run_name "$STAGE2_WANDB_RUN_NAME" \
  2>&1 | tee "$LOG_DIR/stage2_qwen25_7b_a6000x2.log"

echo "[$(date '+%F %T')] done"
