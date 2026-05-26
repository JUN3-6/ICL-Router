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
data/c2c_projectors_p123_mcq_challenging_router/
  question_train.json
  question_test.json
  train_router.json
  test_router.json
  experts_information_500.json
```

Those files are produced from C2C candidate labels by the C2C_IRL dataset-prep script.
The training script intentionally does not regenerate labels.

## Train

Create the conda environment first:

```bash
bash scripts/setup_route_irl_env.sh
conda activate route-IRL
```

Then start training:

```bash
bash scripts/train_c2c_projector_router_7b_a6000x2.sh 0,1
```

Useful overrides:

```bash
DATA_DIR=/path/to/router_data \
OUT_DIR=/path/to/checkpoints \
WANDB_PROJECT=C2C_IRL \
bash scripts/train_c2c_projector_router_7b_a6000x2.sh 0,1
```

Default effective batch sizes:

- stage1: `2 GPUs * batch 1 * grad_accum 8 = 16`
- stage2: `2 GPUs * batch 1 * grad_accum 16 = 32`

## Evaluate

```bash
bash scripts/eval_c2c_projector_router_7b_a6000x2.sh 0
```

or evaluate a specific checkpoint:

```bash
CHECKPOINT=/path/to/final_step4220 bash scripts/eval_c2c_projector_router_7b_a6000x2.sh 0
```
