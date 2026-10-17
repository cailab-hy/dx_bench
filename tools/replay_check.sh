#!/usr/bin/env bash
# Replay a converted dataset episode through the policy server (dx_policy) in the sim (dx_arena, rby1_joint) and
# compare the trajectory with the original raw recording.
#   tools/replay_check.sh <dataset_dir> <episode_index> <raw_root> <scene_json> <out_name> [port]
set -u
cd "$(dirname "$0")/.."
DS=$1; EP=$2; RAW=$3; SCENE=$4; NAME=$5; PORT=${6:-8771}
OUT=outputs/replay/$NAME; mkdir -p "$OUT"
~/miniforge3/envs/dx_policy/bin/python -m dxbench_policy.serve.server --backend replay --dataset "$DS" --episode "$EP" \
  --port "$PORT" > "$OUT/server.log" 2>&1 &
SRV=$!
for i in $(seq 1 90); do grep -q listening "$OUT/server.log" 2>/dev/null && break; sleep 2; done
SRC=$(python3 -c "import json;print([s for s in json.load(open('$DS/meta/dx_sources.json')) if s['episode_index']==$EP][0]['raw'])")
N=$(python3 -c "import json;print([s for s in json.load(open('$DS/meta/dx_sources.json')) if s['episode_index']==$EP][0]['frames'])")
timeout 7200 ~/miniforge3/envs/dx_arena/bin/python dxbench/dxbench/scripts/run_episode.py \
  --policy dxbench.policies.rby1_lerobot_remote.Rby1LerobotRemotePolicy --policy_kw remote_port=$PORT \
  --out "$OUT" --record "$OUT/raw" --record_all --max_steps "$N" \
  --external_environment_class_path dxbench.tasks.bed_tasks:DxBed03Environment dx_bed03 --scene "$SCENE" \
  --embodiment rby1_joint > "$OUT/episode.log" 2>&1
kill $SRV
R=$(ls -d "$OUT"/raw/episode-* | head -1)
~/miniforge3/envs/dx_arena/bin/python dxbench/dxbench/scripts/compare_replay.py --orig "$RAW/$SRC" --replay "$R" \
  --out "reports/stage4/replay_check_$NAME.json"
