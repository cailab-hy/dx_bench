#!/usr/bin/env bash
# T3a/T3g matrix: idle hold under dt, actuator model, device, cloning and damping variants.
set -u
cd "$(dirname "$0")"
PY=${PY:-~/miniforge3/envs/dx_arena/bin/python}
LOG=../../reports/rby1_check/t3/logs; mkdir -p "$LOG"
run() { local name=$1; shift; echo "== $name: $*"; timeout 1200 $PY t3_load.py "$@" > "$LOG/$name.log" 2>&1; grep -E "^\[T3\]" "$LOG/$name.log" | tail -1; }
run idle_impl_500      --test idle --dt 500
run idle_impl_200      --test idle --dt 200
run idle_pd_500        --test idle --dt 500 --mode ideal_pd --effort usd
run idle_pd_200        --test idle --dt 200 --mode ideal_pd --effort usd
run idle_ready_500     --test idle --dt 500 --pose ready --tag ready
run idle_cpu_500       --test idle --dt 500 --device cpu
run idle_env4_500      --test idle --dt 500 --num_envs 4
run idle_zeta07_500    --test idle --dt 500 --damping_ratio 0.7 --tag zeta07
