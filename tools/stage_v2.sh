#!/usr/bin/env bash
# D-U10 pipeline on the v2 demos (2.5 cm grasp, towel E 1e5): wait for the collector -> convert (dx24, 240 x 320) ->
# validate -> ACT 100k + SmolVLA 20k trained at the same time -> 10 closed-loop plays each -> RESULTS.md.
#   tools/stage_v2.sh        (run with no other GPU job; ~5-6 h after the collection ends)
cd "$(dirname "$0")/.."
RAW=data/raw/bed03_oracle_v2; DS=data/lerobot/bed03_oracle_v2_dx24
PYP=$HOME/miniforge3/envs/dx_policy/bin/python; A=$HOME/miniforge3/envs/dx_arena/bin/python
mkdir -p reports/stage_v2 outputs/logs
while pgrep -f '^bash tools/collect_demos.sh' >/dev/null; do sleep 120; done
echo "[v2] convert $(date), raw episodes $(ls -d $RAW/episode-* | wc -l)"
EXCL=$(python3 -c "
import json, pathlib
for ep in sorted(pathlib.Path('$RAW').glob('episode-*')):
    sid = json.loads((ep / 'episode.json').read_text()).get('scene_id')
    log = pathlib.Path('outputs/demos_v2/logs') / f'{sid}.log'
    if log.exists() and 'fail to launch' in log.read_text(errors='ignore'):
        print(ep.name)")
echo "[v2] excluded: $EXCL" | tee reports/stage_v2/excluded_episodes.txt
$PYP -m dxbench_policy.convert.raw_to_lerobot --raw $RAW --out $DS --fmt dx24 --image_scale 0.5 \
  --repo_id dxbench/bed03_oracle_v2 --exclude $EXCL 2>&1 | grep "^\[convert\] {" | tee reports/stage_v2/convert_dx24.txt
$PYP -m dxbench_policy.convert.validate --out $DS --raw $RAW 2>&1 | grep "^\[validate\] ok" | tee reports/stage_v2/validate_dx24.txt
cp $DS/meta/dx_validation.json reports/stage_v2/validation_dx24.json 2>/dev/null
echo "[v2] train $(date)"
DATA=$DS OUT=outputs/train/act_bed03_v2 STEPS=100000 SAVE=25000 dxbench_policy/train_act.sh > outputs/logs/train_act_v2.log 2>&1 &
ACT=$!
DATA=$DS OUT=outputs/train/smolvla_bed03 STEPS=20000 dxbench_policy/train_smolvla.sh > outputs/logs/train_smolvla.log 2>&1 &
SVL=$!
wait $ACT; echo "[v2] act done rc=$? $(date)"
wait $SVL; echo "[v2] smolvla done rc=$? $(date)"
for f in act_v2 smolvla; do
  tr '\r' '\n' < outputs/logs/train_$f.log | grep "ot_train.py:641 step:" | awk 'NR%25==1' \
    | sed -E 's/.*(step:[^ ]+).*(epch:[^ ]+) (loss:[^ ]+).*/\1 \2 \3/' > reports/stage_v2/train_${f}_tail.txt
done
echo "[v2] plays $(date)"
$A -m dxbench.bench.run_benchmark benchmarks/dx_bed_v0/experiments/act_bed03_v2.yaml > outputs/logs/bench_act_v2.log 2>&1 &
sleep 120
$A -m dxbench.bench.run_benchmark benchmarks/dx_bed_v0/experiments/smolvla_bed03.yaml > outputs/logs/bench_smolvla.log 2>&1 &
wait
grep -h "^\[bench\].*: [0-9]*/10" outputs/logs/bench_act_v2.log outputs/logs/bench_smolvla.log
$A -m dxbench.bench.make_results
echo "[v2] done $(date)"
