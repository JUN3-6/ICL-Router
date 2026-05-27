# C2C Projector Router: 7B / 2x A6000

This runs the ICL-Router paper-code path with C2C candidates:

- router LLM: `Qwen/Qwen2.5-7B-Instruct`
- embedding model: `Qwen/Qwen3-Embedding-8B`
- candidates: `receiver`, `behavior_p1`, `behavior_p2`, `behavior_p3`
- stage1: query reconstruction projector/LLM alignment
- stage2: Yes/No router training with profile-500 prompts

## Required Data

Place the prepared router files here:

```text
data/c2c_projectors_p123/
  question_train.json
  question_test.json
  train_router.json
  test_router.json
  experts_information_500.json
```

Those files are produced from C2C candidate labels by the C2C_IRL dataset-prep
script. The default dataset uses all 3659 train queries with four labels per
query and does not use the previous `router_source=challenging` subset. The
training script intentionally does not regenerate labels.

To train stage2 on the discriminative challenge subset instead, use:

```bash
ROUTER_SOURCE=challenging bash scripts/train_c2c_projector_router_7b_a6000x2.sh 0,1
```

This selects `data/c2c_projectors_p123_mcq_challenging_router` for stage2
router rows and reuses the default alltrain stage1 projector/adapter unless
`STAGE1_OUT_DIR` is explicitly overridden. For `ROUTER_SOURCE=challenging`,
stage2 defaults to `STAGE2_ROUTING_LOSS=group_softmax`, which optimizes the
query-level candidate ranking instead of independent row-wise Yes/No BCE.

## Train

Create the conda environment first:

```bash
bash scripts/setup_route_irl_env.sh
conda activate route-IRL
```

If you already have an active environment, the train/eval scripts use it by default.
For example, from `(kjh_c2c)` you can run the commands directly, or set
`ENV_NAME=kjh_c2c` explicitly.

Before starting a long run, check data, imports, and visible GPUs:

```bash
bash scripts/check_c2c_projector_router_setup.sh
```

Then start training:

```bash
bash scripts/train_c2c_projector_router_7b_a6000x2.sh 0,1
```

Useful overrides:

```bash
ROUTER_SOURCE=alltrain|mcq|challenging \
STAGE2_ROUTING_LOSS=row_bce|group_softmax \
DATA_DIR=/path/to/router_data \
OUT_DIR=/path/to/checkpoints \
WANDB_PROJECT=C2C_IRL \
bash scripts/train_c2c_projector_router_7b_a6000x2.sh 0,1
```

Default effective batch sizes:

- stage1: `2 GPUs * batch 1 * grad_accum 8 = 16`
- stage2: `2 GPUs * batch 2 * grad_accum 16 = 64`

The default A6000 path trains the 7B router LLM through LoRA:

- stage1 trains the projector plus LoRA adapter for 3 epochs.
- stage2 starts from the stage1 projector/adapter and continues LoRA router
  training.

This avoids allocating full 7B optimizer states while still updating the router
LLM. Full LLM fine-tuning is likely to OOM on 2x48GB without reliable CPU/NVMe
offload. To force a frozen-LLM run instead:

```bash
USE_LORA=0 STAGE1_EPOCHS=1 STAGE1_UNFREEZE_EPOCH=999 STAGE2_FREEZE_LLM=1 \
  bash scripts/train_c2c_projector_router_7b_a6000x2.sh 0,1
```

Optimizer offload is disabled by default on A6000 because DeepSpeed CPUAdam
often fails when the extension is not built in the environment. If 48GB VRAM is
still insufficient, retry with:

```bash
OFFLOAD_OPTIMIZER=cpu bash scripts/train_c2c_projector_router_7b_a6000x2.sh 0,1
```

## Evaluate

```bash
bash scripts/eval_c2c_projector_router_7b_a6000x2.sh 0
```

or evaluate a specific checkpoint:

```bash
CHECKPOINT=/path/to/final_step4220 bash scripts/eval_c2c_projector_router_7b_a6000x2.sh 0
```
