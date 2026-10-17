#!/usr/bin/env bash
# Stages 4-5 end to end for BED-03 Easy (run after the oracle passes):
#   oracle demos (raw, dx_arena) -> LeRobot v3 dx24 + rby1 (dx_policy) -> validation -> replay check of episode 0
#   -> short ACT training -> 10 closed-loop plays on the evaluation scenes (benchmark act_bed03)
#   N=20 P=3 STEPS=20000 tools/stage45.sh
set -u
cd "$(dirname "$0")/.."
N=${N:-20}; P=${P:-3}; STEPS=${STEPS:-20000}
RAW=data/raw/bed03_oracle; DS=data/lerobot/bed03_oracle_dx24; DS16=data/lerobot/bed03_oracle_rby1
PYP=$HOME/miniforge3/envs/dx_policy/bin/python
export SVT_LOG=1
mkdir -p reports/stage4 reports/stage5 outputs/logs
echo "[stage45] demos $(date)"
N=$N P=$P RAW=$RAW tools/collect_demos.sh data/scenes_train/bed03_easy_train_*.json | tee outputs/logs/collect_demos.log
echo "[stage45] convert $(date)"
~/miniforge3/envs/dx_arena/bin/python tools/raw_png_to_jpg.py $RAW | tail -1
# leave out episodes whose Kit log shows PhysX GPU kernel failures (GPU memory exhaustion while collecting)
EXCL=$(python3 -c "
import json, pathlib
for ep in sorted(pathlib.Path('$RAW').glob('episode-*')):
    sid = json.loads((ep / 'episode.json').read_text()).get('scene_id')
    log = pathlib.Path('outputs/demos/logs') / f'{sid}.log'
    if log.exists() and 'fail to launch' in log.read_text(errors='ignore'):
        print(ep.name)")
echo "[stage45] excluded: $EXCL" | tee reports/stage4/excluded_episodes.txt
$PYP -m dxbench_policy.convert.raw_to_lerobot --raw $RAW --out $DS --fmt dx24 --image_scale 0.5 --repo_id dxbench/bed03_oracle --exclude $EXCL \
  2>&1 | grep "^\[convert\] {" | tee reports/stage4/convert_dx24.txt
$PYP -m dxbench_policy.convert.raw_to_lerobot --raw $RAW --out $DS16 --fmt rby1 --repo_id dxbench/bed03_oracle_rby1 --exclude $EXCL \
  2>&1 | grep "^\[convert\] {" | tee reports/stage4/convert_rby1.txt
$PYP -m dxbench_policy.convert.validate --out $DS --raw $RAW 2>&1 | grep "^\[validate\] ok" | tee reports/stage4/validate_dx24.txt
$PYP -m dxbench_policy.convert.validate --out $DS16 --raw $RAW 2>&1 | grep "^\[validate\] ok" | tee reports/stage4/validate_rby1.txt
cp $DS/meta/dx_validation.json reports/stage4/validation_dx24.json 2>/dev/null
cp $DS16/meta/dx_validation.json reports/stage4/validation_rby1.json 2>/dev/null
echo "[stage45] replay check $(date)"
SCENE=$(python3 -c "
import json; s=json.load(open('$DS/meta/dx_sources.json'))[0]; print('data/scenes_train/' + s['scene_id'] + '.json')")
tools/replay_check.sh $DS 0 $RAW "$SCENE" ep0 8771 | tail -1 | tee reports/stage4/replay_check_ep0.txt
echo "[stage45] train ACT $(date)"
DATA=$DS OUT=outputs/train/act_bed03 STEPS=$STEPS BS=${BS:-16} dxbench_policy/train_act.sh > outputs/logs/train_act.log 2>&1
tr "\r" "\n" < outputs/logs/train_act.log | grep "ot_train.py:641 step:" | awk "NR%20==1" | sed -E "s/.*(step:[^ ]+).*(epch:[^ ]+) (loss:[^ ]+).*(l1_loss:[^ ]+).*/\1 \2 \3 \4/" | tee reports/stage5/train_act_tail.txt
echo "[stage45] closed-loop plays $(date)"
~/miniforge3/envs/dx_arena/bin/python -m dxbench.bench.run_benchmark benchmarks/dx_bed_v0/experiments/act_bed03.yaml \
  | tee outputs/logs/bench_act.log | grep "^\[bench\]"
echo "[stage45] done $(date)"
