#!/usr/bin/env bash
# SmolVLA fine-tune (lerobot/smolvla_base) on a converted dx24 dataset (dx_policy env). Local only: no hub push, no wandb.
# The dataset cameras front / right / left are renamed to the pretrained camera1..3 inputs.
#   DATA=data/lerobot/bed03_oracle_v2_dx24 OUT=outputs/train/smolvla_bed03 STEPS=20000 dxbench_policy/train_smolvla.sh
set -eu
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniforge3/envs/dx_policy/bin}
DATA=${DATA:-data/lerobot/bed03_oracle_v2_dx24}
OUT=${OUT:-outputs/train/smolvla_bed03}
STEPS=${STEPS:-20000}
BS=${BS:-32}
mkdir -p "$(dirname "$OUT")"
"$PY/lerobot-train" \
  --dataset.repo_id=dxbench/local --dataset.root="$DATA" \
  --policy.path=lerobot/smolvla_base --policy.push_to_hub=false --policy.device=cuda \
  --rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.right": "observation.images.camera2", "observation.images.left": "observation.images.camera3"}' \
  --output_dir="$OUT" --steps="$STEPS" --batch_size="$BS" --num_workers=${WORKERS:-6} \
  --save_freq=${SAVE:-5000} --log_freq=200 --wandb.enable=false --seed=1 ${EXTRA:-}
