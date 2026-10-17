#!/usr/bin/env bash
# Longer ACT run on the BED-03 oracle demos (100k steps) after the replay baseline, then 10 closed-loop plays.
#   tools/stage5_long_act.sh      (waits for running benchmarks to finish; run with no other GPU job)
cd "$(dirname "$0")/.."
until [ -f outputs/bench/replay_bed03/summary.json ] && ! pgrep -f 'dxbench.bench.run_benchmark' >/dev/null; do sleep 60; done
echo "[long_act] train $(date)"
DATA=data/lerobot/bed03_oracle_dx24 OUT=outputs/train/act_bed03_100k STEPS=100000 SAVE=25000 dxbench_policy/train_act.sh \
  > outputs/logs/train_act_100k.log 2>&1 || { echo "[long_act] training failed"; exit 1; }
tr '\r' '\n' < outputs/logs/train_act_100k.log | grep "ot_train.py:641 step:" | awk 'NR%50==1' \
  | sed -E 's/.*(step:[^ ]+).*(epch:[^ ]+) (loss:[^ ]+).*(l1_loss:[^ ]+).*/\1 \2 \3 \4/' | tee reports/stage5/train_act_100k_tail.txt
echo "[long_act] plays $(date)"
~/miniforge3/envs/dx_arena/bin/python -m dxbench.bench.run_benchmark benchmarks/dx_bed_v0/experiments/act_bed03_100k.yaml \
  > outputs/logs/bench_act_100k.log 2>&1
grep "^\[bench\]" outputs/logs/bench_act_100k.log | tail -3
~/miniforge3/envs/dx_arena/bin/python -m dxbench.bench.make_results
echo "[long_act] done $(date)"
