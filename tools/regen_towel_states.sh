#!/usr/bin/env bash
# Regenerate settled cloth states after the towel material change (D-28): 4 streams at a time, priority scenes first.
cd "$(dirname "$0")/.."
mkdir -p outputs/logs/states
S=benchmarks/dx_bed_v0/scenes; T=data/scenes_train
# priority: BED-03 / BED-02 easy evaluation + BED-03 easy training (demos)
(FORCE=1 tools/make_states.sh $S/bed03_easy_*.json > outputs/logs/states/regen_a.log 2>&1) &
(FORCE=1 tools/make_states.sh $S/bed02_easy_*.json > outputs/logs/states/regen_b.log 2>&1) &
(FORCE=1 tools/make_states.sh $T/bed03_easy_train_1000*.json $T/bed03_easy_train_1001[0-4].json > outputs/logs/states/regen_c.log 2>&1) &
(FORCE=1 tools/make_states.sh $T/bed03_easy_train_1001[5-9].json $T/bed03_easy_train_1002*.json > outputs/logs/states/regen_d.log 2>&1) &
wait
echo "[regen] priority done $(date)"
# the rest: BED-02 hard, BED-03 medium / hard evaluation, BED-02 easy training
(FORCE=1 tools/make_states.sh $S/bed02_hard_*.json > outputs/logs/states/regen_e.log 2>&1) &
(FORCE=1 tools/make_states.sh $S/bed03_medium_*.json > outputs/logs/states/regen_f.log 2>&1) &
(FORCE=1 tools/make_states.sh $S/bed03_hard_*.json > outputs/logs/states/regen_g.log 2>&1) &
(FORCE=1 tools/make_states.sh $T/bed02_easy_train_*.json > outputs/logs/states/regen_h.log 2>&1) &
wait
echo "[regen] all done $(date)"
