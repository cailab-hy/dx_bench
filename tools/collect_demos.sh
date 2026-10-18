#!/usr/bin/env bash
# Oracle demonstrations (raw recording, only successful episodes kept) over training scenes, P at a time, until N saved.
#   N=20 P=3 POLICY=dxbench.policies.oracle_bed03.Bed03OraclePolicy RAW=data/raw/bed03_oracle \
#     tools/collect_demos.sh data/scenes_train/bed03_easy_train_*.json
set -u
cd "$(dirname "$0")/.."
N=${N:-20}; P=${P:-3}; RAW=${RAW:-data/raw/bed03_oracle}
POLICY=${POLICY:-dxbench.policies.oracle_bed03.Bed03OraclePolicy}
EMB=${EMB:-rby1_ik}
OUTD=${OUTD:-outputs/demos}   # per-scene results + logs (a scene with a result here is skipped)
mkdir -p "$RAW" "$OUTD/logs"
saved() { ls -d "$RAW"/episode-* 2>/dev/null | wc -l; }
one() {
  f=$1; id=$(basename "$f" .json)
  [ "$(saved)" -ge "$N" ] && return
  [ -f "$OUTD/$id/result.json" ] && return
  task=$(python3 -c "import json;print(json.load(open('$f'))['task'][-1])")
  for try in 1 2 3; do
    timeout ${TIMEOUT:-7200} ~/miniforge3/envs/dx_arena/bin/python dxbench/dxbench/scripts/run_episode.py --policy "$POLICY" \
      --out "$OUTD/$id" --record "$RAW" --progress_every 150 \
      --external_environment_class_path dxbench.tasks.bed_tasks:DxBed0${task}Environment dx_bed0${task} --scene "$f" \
      --embodiment "$EMB" > "$OUTD/logs/$id.log" 2>&1
    grep -q "^\[episode\]" "$OUTD/logs/$id.log" && { grep "^\[episode\]" "$OUTD/logs/$id.log"; break; }
    grep -q "platforminfo\|Segmentation" "$OUTD/logs/$id.log" || break
  done
}
export -f one saved; export N RAW POLICY EMB TIMEOUT OUTD
printf '%s\n' "$@" | xargs -P "$P" -I{} bash -c 'one {}'
echo "[collect] saved $(saved) episodes in $RAW"
