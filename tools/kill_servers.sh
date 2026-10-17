#!/usr/bin/env bash
# Stop dxbench_policy websocket servers (script file: the pattern cannot match the caller's command line).
pkill -f 'dxbench_policy.serve.server' ; sleep 1
pgrep -af 'dxbench_policy.serve.server' | grep -v pgrep || true
