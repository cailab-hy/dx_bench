#!/usr/bin/env bash
# Run a policy on scene JSONs, one Kit process per scene (retry on Kit startup crash).
#   POLICY=dxbench.policies.oracle_bed03.Bed03OraclePolicy OUT=outputs/oracle tools/run_oracle.sh scenes/*.json [-- extra args]
set -u
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniforge3/envs/dx_arena/bin/python}
POLICY=${POLICY:-dxbench.policies.oracle_bed03.Bed03OraclePolicy}
OUT=${OUT:-outputs/oracle}
EXTRA=${EXTRA:-}
mkdir -p "$OUT/logs"
for f in "$@"; do
  id=$(basename "$f" .json)
  task=$(python3 -c "import json,sys;print(json.load(open('$f'))['task'][-1])")
  for try in 1 2 3; do
    timeout -k 120 ${TIMEOUT:-1800} "$PY" dxbench/dxbench/scripts/run_episode.py --policy "$POLICY" --out "$OUT/$id" $EXTRA \
      --external_environment_class_path dxbench.tasks.bed_tasks:DxBed0${task}Environment dx_bed0${task} --scene "$f" \
      ${ENV_EXTRA:-} > "$OUT/logs/$id.log" 2>&1
    rc=$?
    if grep -q "^\[episode\]" "$OUT/logs/$id.log"; then grep "^\[episode\]" "$OUT/logs/$id.log"; break; fi
    echo "[run_oracle] $id try $try failed rc=$rc"; grep -E "Error|error" "$OUT/logs/$id.log" | tail -3
    grep -q "platforminfo\|Segmentation" "$OUT/logs/$id.log" || break
  done
done
