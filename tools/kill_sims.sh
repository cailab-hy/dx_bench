#!/usr/bin/env bash
# Stop dxbench simulation jobs (benchmark runners, demo collectors, episode / state / teleop processes).
# A script file, so `pkill -f <pattern>` cannot match the calling shell's own command line.
#   tools/kill_sims.sh [all|episodes|states|demos]
what=${1:-all}
pat_ep='^/home/cai/miniforge3/envs/dx_arena/bin/python (/home/cai/dx_bench/)?dxbench/dxbench/scripts/run_episode.py'
pat_st='^/home/cai/miniforge3/envs/dx_arena/bin/python dxbench/dxbench/tasks/make_cloth_states.py'
if [ "$what" = all ] || [ "$what" = episodes ]; then
  pkill -f 'dxbench.bench.run_benchmark' ; pkill -f '^bash tools/collect_demos.sh' ; pkill -f '^bash tools/run_oracle.sh'
  pkill -f '^xargs -P' ; sleep 1 ; pkill -f "$pat_ep"
fi
if [ "$what" = demos ]; then
  pkill -f '^bash tools/collect_demos.sh' ; pkill -f '^xargs -P' ; sleep 1
  pkill -f 'scripts/run_episode.py .*--out outputs/demos/'
fi
if [ "$what" = all ] || [ "$what" = states ]; then
  pkill -f '^bash tools/make_states.sh' ; sleep 1 ; pkill -f "$pat_st"
fi
sleep 3
pgrep -af 'run_episode.py|make_cloth_states.py|run_benchmark|collect_demos' | grep -v pgrep | cut -c1-120 || true
