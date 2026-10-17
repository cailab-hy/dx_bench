#!/usr/bin/env bash
# Short ACT training on a converted dx24 dataset (dx_policy env). Local only: no hub push, no wandb.
#   DATA=data/lerobot/dx_bed03_oracle_dx24 OUT=outputs/train/act_bed03 STEPS=20000 dxbench_policy/train_act.sh
set -eu
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniforge3/envs/dx_policy/bin}
DATA=${DATA:-data/lerobot/dx_bed03_oracle_dx24}
OUT=${OUT:-outputs/train/act_bed03}
STEPS=${STEPS:-20000}
BS=${BS:-16}
mkdir -p "$(dirname "$OUT")"
"$PY/lerobot-train" \
  --dataset.repo_id=dxbench/local --dataset.root="$DATA" \
  --policy.type=act --policy.push_to_hub=false --policy.device=cuda \
  --policy.chunk_size=${CHUNK:-30} --policy.n_action_steps=${NSTEPS:-8} \
  --output_dir="$OUT" --steps="$STEPS" --batch_size="$BS" --num_workers=${WORKERS:-6} \
  --save_freq=${SAVE:-5000} --log_freq=200 --wandb.enable=false --seed=1 ${EXTRA:-}
