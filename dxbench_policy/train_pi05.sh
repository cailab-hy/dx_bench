#!/usr/bin/env bash
# pi0.5 fine-tune (lerobot/pi05_base, full model, bf16 + gradient checkpointing: 37 GB at batch 8 on the 48 GB
# RTX 6000 Ada, ~2.6 s/step) on a converted dx24 dataset (dx_policy env). Local only: no hub push, no wandb.
# The dataset cameras front / left / right are renamed to the pretrained base / left wrist / right wrist inputs.
# Needs the gated google/paligemma-3b-pt-224 tokenizer (hf auth login with an account that accepted its license).
#   DATA=data/lerobot/bed03_oracle_v4_dx24 OUT=outputs/train/pi05_bed03_v4 STEPS=15000 dxbench_policy/train_pi05.sh
set -eu
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniforge3/envs/dx_policy/bin}
DATA=${DATA:-data/lerobot/bed03_oracle_v4_dx24}
OUT=${OUT:-outputs/train/pi05_bed03_v4}
STEPS=${STEPS:-15000}
BS=${BS:-8}
mkdir -p "$(dirname "$OUT")"
"$PY/lerobot-train" \
  --dataset.repo_id=dxbench/local --dataset.root="$DATA" \
  --policy.path=lerobot/pi05_base --policy.push_to_hub=false --policy.device=cuda \
  --policy.dtype=bfloat16 --policy.gradient_checkpointing=true \
  --rename_map='{"observation.images.front": "observation.images.base_0_rgb", "observation.images.left": "observation.images.left_wrist_0_rgb", "observation.images.right": "observation.images.right_wrist_0_rgb"}' \
  --output_dir="$OUT" --steps="$STEPS" --batch_size="$BS" --num_workers=${WORKERS:-6} \
  --save_freq=${SAVE:-5000} --log_freq=200 --wandb.enable=false --seed=1 ${EXTRA:-}
