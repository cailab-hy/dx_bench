#!/usr/bin/env bash
# Settled cloth initial states, one Kit process per scene (retry up to 3x on the known Kit startup segfault).
#   tools/make_states.sh benchmarks/dx_bed_v0/scenes/bed03_easy_*.json
# A rejected state ([states-reject], e.g. a BED-02 garment slid off the desk) or one whose start score is not 0 is
# redrawn with --salt 1..4.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniforge3/envs/dx_arena/bin/python}
mkdir -p outputs/logs/states
for f in "$@"; do
  id=$(basename "$f" .json)
  if [ -z "${FORCE:-}" ] && grep -q cloth_state.*npz "$f" && ! grep -q "cloth_state\": null" "$f"; then echo "[make_states] $id exists"; continue; fi
  for salt in 0 1 2 3 4; do
    for try in 1 2 3; do
      timeout 900 "$PY" dxbench/dxbench/tasks/make_cloth_states.py --scenes "$f" --salt $salt \
        --states "${STATES:-$(dirname "$(dirname "$f")")/states}" > "outputs/logs/states/$id.log" 2>&1
      rc=$?
      if grep -q "^\[states" "outputs/logs/states/$id.log"; then grep "^\[states" "outputs/logs/states/$id.log"; break; fi
      echo "[make_states] $id try $try failed rc=$rc"; tail -3 "outputs/logs/states/$id.log"
    done
    grep -q "^\[states-reject\]" "outputs/logs/states/$id.log" && continue
    # a start state that already scores (e.g. a light BED-01 duvet that stayed spread, D-47) is redrawn too;
    # BED-03 "second" scenes start with the first garment in the basket on purpose (D-49)
    case "$id" in *_second_*) break ;; esac
    score=$("$PY" -c "import sys, json; sys.argv = ['x']; sys.path.insert(0, 'tools'); import predicate_audit as A
print(A.eval_scene(json.load(open('$f')))['score'])" 2>/dev/null)
    [ -z "$score" ] && { echo "[make_states] $id start score check failed"; break; }
    [ "$score" = "0.0" ] && break
    echo "[make_states] $id start score $score: redraw"
  done
done
