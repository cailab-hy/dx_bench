#!/usr/bin/env bash
# Replay one episode of a dx24 LeRobot dataset in the simulator, open loop: the policy server (dx_policy) streams the
# recorded actions, run_episode (dx_arena, rby1_joint) plays them in the episode's scene and films it (head + wrist
# cameras and a third-person view). The task environment comes from the scene JSON.
#   tools/replay_episode.sh <dataset_dir> <episode_index> <scene_json> <out_dir> [port]
#   tools/replay_episode.sh samples/oracle_d70_dx24 2 samples/oracle_d70_dx24/scenes/bed03_easy_train_10008.json outputs/replay/bed03
# Cloth physics is not bit-reproducible on the GPU, so an open-loop replay follows the recording closely but the cloth
# can end up differently (the task may not succeed); compare the arm / base motion, not the outcome.
set -u
cd "$(dirname "$0")/.."
DS=$1; EP=$2; SCENE=$3; OUT=$4; PORT=${5:-8771}
P=${P:-$HOME/miniforge3/envs/dx_policy/bin/python}
A=${A:-$HOME/miniforge3/envs/dx_arena/bin/python}
mkdir -p "$OUT"
T=$(python3 -c "import json;print(json.load(open('$SCENE'))['task'][-1])")       # BED-0<T>
N=$(python3 -c "import json;print([s for s in json.load(open('$DS/meta/dx_sources.json')) if s['episode_index']==$EP][0]['frames'])")
"$P" -m dxbench_policy.serve.server --backend replay --dataset "$DS" --episode "$EP" --port "$PORT" > "$OUT/server.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
for i in $(seq 1 90); do grep -q listening "$OUT/server.log" 2>/dev/null && break; sleep 2; done
"$A" dxbench/dxbench/scripts/run_episode.py --policy dxbench.policies.rby1_lerobot_remote.Rby1LerobotRemotePolicy \
  --policy_kw remote_port="$PORT" --out "$OUT" --video --max_steps "$N" \
  --external_environment_class_path dxbench.tasks.bed_tasks:DxBed0${T}Environment dx_bed0$T --scene "$SCENE" \
  --embodiment rby1_joint --observer > "$OUT/episode.log" 2>&1
grep "^\[episode\]" "$OUT/episode.log"
echo "[replay] videos: $OUT/observer.mp4 head.mp4 left_wrist.mp4 right_wrist.mp4"
