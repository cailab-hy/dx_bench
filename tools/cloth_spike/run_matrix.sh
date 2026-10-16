#!/usr/bin/env bash
# Stage 1 cloth spike matrix (runs tagged "m"). Each run writes reports/cloth_spike/runs/<run>_m.json.
# Kit startup segfaults are retried (max 3).
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-~/miniforge3/envs/dx_arena/bin/python}
LOG=reports/cloth_spike/logs; mkdir -p "$LOG"
run() {
  local name=$1; shift
  for attempt in 1 2 3; do
    timeout 2400 $PY tools/cloth_spike/cloth_spike.py "$@" --tag m > "$LOG/$name.log" 2>&1
    if grep -q "^\[cloth\]" "$LOG/$name.log"; then grep "^\[cloth\]" "$LOG/$name.log"; return; fi
    if ! grep -q "platforminfo\|Segmentation\|exit code 139" "$LOG/$name.log"; then echo "FAIL $name (see log)"; return; fi
    echo "retry $name ($attempt)"
  done
}
STAGE=${STAGE:-all}
# Reduced matrix (time-box): 60 s stability for the assist grasp only; friction / RTF / sweeps shorter.
# Already done in the first pass: duvet friction 240/360/480 (60 s), duvet assist 240 (60 s).
if [ "$STAGE" = all ] || [ "$STAGE" = core ]; then
  for dt in 360 480; do run core_duvet_assist_$dt --test duvet --grip assist --dt $dt --duration 60; done
  for t in sleeve floor; do for dt in 240 360 480; do run core_${t}_assist_$dt --test $t --grip assist --dt $dt --duration 60; done; done
  for t in sleeve floor; do run core_${t}_friction_360 --test $t --grip friction --dt 360 --duration 20; done
fi
if [ "$STAGE" = all ] || [ "$STAGE" = rtf ]; then
  for t in duvet sleeve floor; do for dt in 240 480; do
    run rtf_${t}_${dt} --test $t --grip assist --dt $dt --duration 15 --cameras
  done; done
fi
if [ "$STAGE" = all ] || [ "$STAGE" = sweep ]; then
  D="--test duvet --grip assist --dt 360 --duration 20"
  run sw_res47  $D --res_cm 4.7
  run sw_res12  $D --res_cm 1.2
  run sw_th1    $D --thickness 0.001
  run sw_th4    $D --thickness 0.004
  run sw_mu03   $D --friction 0.3
  run sw_mu10   $D --friction 1.0
  run sw_it8    $D --cloth_iters 8
  run sw_it32   $D --cloth_iters 32
  run sw_sc0    $D --self_collision 0
  run sw_sleeve_fr_mu15 --test sleeve --grip friction --dt 360 --duration 20 --friction 1.5
  run sw_floor_fr_mu15  --test floor  --grip friction --dt 360 --duration 20 --friction 1.5
  run sw_sleeve_sc0     --test sleeve --grip assist --dt 360 --duration 20 --self_collision 0
fi
