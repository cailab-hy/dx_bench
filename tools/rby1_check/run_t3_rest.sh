#!/usr/bin/env bash
# T3b-f: crouch, sweep, gripper, drive, camera (GPU) + the proposed tuning as comparison.
# "tuned" = viscous friction only + damping ratio 0.7 (+ gravity feedforward where noted).
set -u
cd "$(dirname "$0")"
PY=${PY:-~/miniforge3/envs/dx_arena/bin/python}
LOG=../../reports/rby1_check/t3/logs; mkdir -p "$LOG"
run() { local name=$1; shift; echo "== $name: $*"; timeout 1800 $PY t3_load.py "$@" > "$LOG/$name.log" 2>&1; grep -E "^\[T3\]" "$LOG/$name.log" | tail -1; }
TUNED="--friction viscous --damping_ratio 0.7"
run idle_tuned_gc_500    --test idle --dt 500 $TUNED --gravity_comp --tag tuned_gc
run idle_tuned_gc_200    --test idle --dt 200 $TUNED --gravity_comp --tag tuned_gc
run crouch_impl_500      --test crouch --dt 500
run crouch_impl_200      --test crouch --dt 200
run crouch_pd_500        --test crouch --dt 500 --mode ideal_pd --effort usd
run crouch_tuned_500     --test crouch --dt 500 $TUNED --tag tuned
run crouch_tuned_gc_500  --test crouch --dt 500 $TUNED --gravity_comp --tag tuned_gc
run crouch_deep_500     --test crouch --dt 500 --crouch 75,-105,90 --tag deep
run crouch_deep_gc_500  --test crouch --dt 500 --crouch 75,-105,90 $TUNED --gravity_comp --tag deep_tuned_gc
run crouch_deeper_500   --test crouch --dt 500 --crouch 90,-130,90 --tag deeper
run crouch_deeper_gc_500 --test crouch --dt 500 --crouch 90,-130,90 $TUNED --gravity_comp --tag deeper_tuned_gc
run sweep_impl_500       --test sweep --dt 500
run sweep_impl_200       --test sweep --dt 200
run sweep_torso_fix_500   --test sweep --dt 500 --fix_root --sweep_joints torso_0,torso_1,torso_2,torso_3,torso_4,torso_5 --tag fixroot
run sweep_tuned_500      --test sweep --dt 500 $TUNED --gravity_comp --tag tuned_gc
run gripper_impl_500     --test gripper --dt 500
run gripper_mimic_500    --test gripper --dt 500 --grip_mimic --tag mimic
run gripper_mimic_gc_500 --test gripper --dt 500 --grip_mimic $TUNED --gravity_comp --tag mimic_tuned_gc
run drive_impl_500       --test drive --dt 500
run drive_impl_200       --test drive --dt 200
run drive_pd_500         --test drive --dt 500 --mode ideal_pd --effort usd
run drive_impl_c0_500     --test drive --dt 500 --caster_mu 0 --tag caster0
run drive_pd_c0_500       --test drive --dt 500 --mode ideal_pd --effort usd --caster_mu 0 --tag caster0
run camera_impl_500      --test camera --dt 500
