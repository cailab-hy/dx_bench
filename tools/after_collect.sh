#!/usr/bin/env bash
# Wait for running demo collectors to exit, then run the rest of stages 4-5 (tools/stage45.sh).
# N=21: one saved episode (scene 10002) is excluded for PhysX GPU kernel failures, so 21 saved -> 20 usable.
cd "$(dirname "$0")/.."
while pgrep -f '^bash tools/collect_demos.sh' >/dev/null; do sleep 60; done
echo "[after_collect] collectors done $(date), saved $(ls -d data/raw/bed03_oracle/episode-* | wc -l)"
N=${N:-21} P=${P:-3} STEPS=${STEPS:-20000} tools/stage45.sh
