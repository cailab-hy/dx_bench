#!/usr/bin/env bash
# Towel material sweep for the fold springback test (tools/cloth_spike/towel_fold.py), one Kit at a time.
#   tools/cloth_spike/towel_sweep.sh "tag:args" ...
cd "$(dirname "$0")/../.."
mkdir -p outputs/logs/towel_fold
for spec in "$@"; do
  tag=${spec%%:*}; extra=${spec#*:}
  [ -f reports/towel_fold/runs/$tag.json ] && continue
  for try in 1 2 3; do
    timeout 1200 ~/miniforge3/envs/dx_arena/bin/python tools/cloth_spike/towel_fold.py --tag $tag $extra \
      > outputs/logs/towel_fold/$tag.log 2>&1
    grep -q "^\[towel_fold\]" outputs/logs/towel_fold/$tag.log && break
    grep -q "platforminfo\|Segmentation" outputs/logs/towel_fold/$tag.log || break
  done
  grep "^\[towel_fold\]" outputs/logs/towel_fold/$tag.log || echo "[towel_sweep] $tag failed"
done
