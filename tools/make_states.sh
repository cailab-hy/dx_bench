#!/usr/bin/env bash
# Settled cloth initial states, one Kit process per scene (retry up to 3x on the known Kit startup segfault).
#   tools/make_states.sh benchmarks/dx_bed_v0/scenes/bed03_easy_*.json
set -u
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniforge3/envs/dx_arena/bin/python}
mkdir -p outputs/logs/states
for f in "$@"; do
  id=$(basename "$f" .json)
  if [ -z "${FORCE:-}" ] && grep -q cloth_state.*npz "$f" && ! grep -q "cloth_state\": null" "$f"; then echo "[make_states] $id exists"; continue; fi
  for try in 1 2 3; do
    timeout 900 "$PY" dxbench/dxbench/tasks/make_cloth_states.py --scenes "$f" \
      --states "${STATES:-$(dirname "$(dirname "$f")")/states}" > "outputs/logs/states/$id.log" 2>&1
    rc=$?
    if grep -q "^\[states\]" "outputs/logs/states/$id.log"; then grep "^\[states\]" "outputs/logs/states/$id.log"; break; fi
    echo "[make_states] $id try $try failed rc=$rc"; tail -3 "outputs/logs/states/$id.log"
  done
done
