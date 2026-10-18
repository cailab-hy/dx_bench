#!/usr/bin/env bash
# After the towel state regeneration (tools/regen_towel_states.sh): BED-02 oracle on 2 easy scenes as soon as the
# priority states exist, then the BED-03 easy oracle benchmark once all states are regenerated.
cd "$(dirname "$0")/.."
L=outputs/logs/states/regen.log
until grep -q "priority done" $L 2>/dev/null; do sleep 60; done
POLICY=dxbench.policies.oracle_bed02.Bed02OraclePolicy OUT=outputs/oracle_bed02_v6 TIMEOUT=10800 \
  EXTRA="--progress_every 75" ENV_EXTRA="--embodiment rby1_ik" \
  tools/run_oracle.sh benchmarks/dx_bed_v0/scenes/bed02_easy_000.json > outputs/logs/bed02_v6_000.log 2>&1 &
sleep 30
POLICY=dxbench.policies.oracle_bed02.Bed02OraclePolicy OUT=outputs/oracle_bed02_v6 TIMEOUT=10800 \
  EXTRA="--progress_every 75" ENV_EXTRA="--embodiment rby1_ik" \
  tools/run_oracle.sh benchmarks/dx_bed_v0/scenes/bed02_easy_001.json > outputs/logs/bed02_v6_001.log 2>&1 &
until grep -q "all done" $L 2>/dev/null; do sleep 60; done
~/miniforge3/envs/dx_arena/bin/python -m dxbench.bench.run_benchmark benchmarks/dx_bed_v0/experiments/oracle_bed03.yaml \
  --parallel 4 > outputs/logs/bench_oracle_final.log 2>&1
wait
grep "^\[bench\]" outputs/logs/bench_oracle_final.log | tail -3
cat outputs/logs/bed02_v6_00*.log
