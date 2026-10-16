#!/usr/bin/env bash
# Full rby1_check pipeline (about 40-60 min on an RTX 6000 Ada).
set -u
cd "$(dirname "$0")"
PY=${PY:-~/miniforge3/envs/dx_arena/bin/python}
$PY t0_env.py
$PY t1_usd_static.py
[ -f _ext/rby1-sdk/models/rby1a/urdf/model_v1.2.urdf ] || bash fetch_urdf.sh
$PY t2_urdf_compare.py
$PY make_override.py
$PY t1_usd_static.py --only override_24dof
bash run_idle_matrix.sh
bash run_damping_study.sh
bash run_t3_rest.sh
for m in open reference; do timeout 1800 $PY t4_bedroom.py --load $m > ../../reports/rby1_check/t4_run_$m.log 2>&1; grep "^\[T4\]" ../../reports/rby1_check/t4_run_$m.log; done
$PY make_report.py
