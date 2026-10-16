#!/usr/bin/env bash
# T3a follow-up: (a) original-style explicit PD vs implicit, (b) damping-ratio tuning, per friction model.
set -u
cd "$(dirname "$0")"
PY=${PY:-~/miniforge3/envs/dx_arena/bin/python}
LOG=../../reports/rby1_check/t3/logs; mkdir -p "$LOG"
run() { local name=$1; shift; echo "== $name: $*"; timeout 1200 $PY t3_load.py "$@" > "$LOG/$name.log" 2>&1; grep -E "^\[T3\]" "$LOG/$name.log" | tail -1; }
for dt in 500 200; do
  run idle_impl_none_$dt     --test idle --dt $dt --friction none --tag nofric
  run idle_pd_none_$dt       --test idle --dt $dt --friction none --mode ideal_pd --effort usd --tag nofric
  run idle_impl_visc_$dt     --test idle --dt $dt --friction viscous --tag visc
  run idle_pd_visc_$dt       --test idle --dt $dt --friction viscous --mode ideal_pd --effort usd --tag visc
  for z in 0.7 1.0; do
    run idle_impl_none_z${z}_$dt --test idle --dt $dt --friction none --damping_ratio $z --tag nofric_z$z
    run idle_impl_visc_z${z}_$dt --test idle --dt $dt --friction viscous --damping_ratio $z --tag visc_z$z
    run idle_impl_full_z${z}_$dt --test idle --dt $dt --friction full --damping_ratio $z --tag full_z$z
  done
done
run idle_impl_none_cpu500 --test idle --dt 500 --device cpu --friction none --tag nofric
run idle_impl_visc_cpu500 --test idle --dt 500 --device cpu --friction viscous --tag visc
