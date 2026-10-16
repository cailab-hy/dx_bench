#!/usr/bin/env bash
# Stage 2 regression: embodiment-level re-run of the 0-stage checks (reports/rby1_check/t5/*.json).
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-~/miniforge3/envs/dx_arena/bin/python}
LOG=reports/rby1_check/t5/logs; mkdir -p "$LOG"
FLAT="--external_environment_class_path dxbench.environments.rby1_envs:DxFlatEnvironment dx_rby1_flat"
BED="--external_environment_class_path dxbench.environments.rby1_envs:DxBedroomEnvironment dx_bedroom"
run() {
  local name=$1; shift
  for attempt in 1 2 3; do
    timeout 1200 $PY tools/rby1_check/t5_embodiment.py "$@" > "$LOG/$name.log" 2>&1
    if grep -q "^\[t5\]" "$LOG/$name.log"; then grep "^\[t5\]" "$LOG/$name.log"; return; fi
    grep -q "platforminfo\|exit code 139" "$LOG/$name.log" || { echo "FAIL $name"; return; }
  done
}
ONLY=${ONLY:-all}
want() { [ "$ONLY" = all ] || [[ " $ONLY " == *" $1 "* ]]; }
want idle && run idle    --test idle    $FLAT
want crouch && run crouch  --test crouch  $FLAT
want drive && run drive   --test drive   $FLAT
want gripper && run gripper --test gripper $FLAT --embodiment rby1_ik --props grasp_cube
want floor && run floor   --test floor   $FLAT --embodiment rby1_ik
want bedroom_camera && run bedroom_camera --test bedroom_camera --enable_cameras $BED
