# Data → training → closed-loop pipeline

The two conda environments split the work:
- `dx_arena` runs the simulation, teleoperation and raw recording, and the policy client. It never imports lerobot.
- `dx_policy` runs the LeRobot conversion, training and the policy server.

All commands run from `~/dx_bench`. Variables used below: `A=~/miniforge3/envs/dx_arena/bin/python` and `P=~/miniforge3/envs/dx_policy/bin/python`.

## 1. Cloth initial states (once per scene)

```bash
tools/make_states.sh benchmarks/dx_bed_v0/scenes/*.json        # evaluation (benchmarks/dx_bed_v0/states)
tools/make_states.sh data/scenes_train/bed03_easy_train_*.json  # training scenes (data/states)
```

## 2. Demonstrations (raw recording, `dx_arena`)

```bash
# scripted oracle demos (privileged; only successful episodes are kept), 3 Kit processes at once, until 20 saved
N=20 P=3 RAW=data/raw/bed03_oracle tools/collect_demos.sh data/scenes_train/bed03_easy_train_*.json

# keyboard teleop (GUI). Keys: W/S A/D Q/E move, Z/X T/G C/V rotate, TAB arm, K gripper, arrows base, U/J torso, H arms home,
# B start, ENTER save, BACKSPACE discard
$A dxbench/dxbench/teleop/run_teleop.py --teleop_device keyboard --record data/raw/bed03_human --visualizer kit \
   --external_environment_class_path dxbench.tasks.bed_tasks:DxBed03Environment dx_bed03 \
   --scene data/scenes_train/bed03_easy_train_10000.json --physics_hz 360

# Quest 2/3 over USB: adb reverse tcp:8012 tcp:8012, then open http://localhost:8012 in the headset browser
# grip = clutch, trigger = gripper, A start, B (hold) discard, X pause, Y (hold) recenter,
# right stick = base (click: slow / fast; locked while a grip is held), left stick up/down = torso preset,
# left stick left/right = camera panel size
$A dxbench/dxbench/teleop/run_teleop.py --teleop_device vr --record data/raw/bed03_vr --realtime ...same env args...

# headless VR end-to-end check (scripted operator through the real WebSocket)
$A dxbench/dxbench/teleop/run_teleop.py --teleop_device scripted_vr --record outputs/teleop_test --save_any ...env args...
```

Raw episode layout: `dxbench/teleop/recorder.py`, which contains:
- PNG frames per camera
- `trajectory.npz`
- `initial_state.json`
- `episode.json`

## 3. LeRobot v3 conversion and validation (`dx_policy`)

```bash
$P -m dxbench_policy.convert.raw_to_lerobot --raw data/raw/bed03_oracle --out data/lerobot/bed03_oracle_dx24 --fmt dx24 \
   --image_scale 0.5   # 240 x 320 frames for training speed; the server resizes live frames to match
$P -m dxbench_policy.convert.raw_to_lerobot --raw data/raw/bed03_oracle --out data/lerobot/bed03_oracle_rby1 --fmt rby1
$P -m dxbench_policy.convert.validate --out data/lerobot/bed03_oracle_dx24 --raw data/raw/bed03_oracle
```

The two formats (D-12):
- `rby1`: the real RB-Y1 16-D layout.
- `dx24`: the full `rby1_joint` action.

Extra columns are named `dx.*`.

## 4. Replay check (`dx_arena` + `dx_policy`)

```bash
# one command: serve episode <i> of the dataset, replay it on rby1_joint, compare with the raw recording
tools/replay_check.sh data/lerobot/bed03_oracle_dx24 0 data/raw/bed03_oracle data/scenes_train/<scene of episode 0>.json ep0
#   -> reports/stage4/replay_check_ep0.json (base xy, joint, TCP errors; pass thresholds in compare_replay.py)
# or by hand:
$P -m dxbench_policy.serve.server --backend replay --dataset data/lerobot/bed03_oracle_dx24 --episode 0 --port 8765 &
$A dxbench/dxbench/scripts/run_episode.py --policy dxbench.policies.rby1_lerobot_remote.Rby1LerobotRemotePolicy \
   --policy_kw remote_port=8765 --out outputs/play/replay0 --video \
   --external_environment_class_path dxbench.tasks.bed_tasks:DxBed03Environment dx_bed03 \
   --scene data/scenes_train/<scene of episode 0>.json --embodiment rby1_joint
```

## 5. Training (`dx_policy`)

```bash
DATA=data/lerobot/bed03_oracle_dx24 OUT=outputs/train/act_bed03 STEPS=20000 dxbench_policy/train_act.sh
# SmolVLA fine-tune (camera names mapped to camera1..3)
lerobot-train --dataset.root=data/lerobot/bed03_oracle_dx24 --dataset.repo_id=dxbench/local --policy.path=lerobot/smolvla_base \
  --policy.push_to_hub=false --rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.right": "observation.images.camera2", "observation.images.left": "observation.images.camera3"}' ...
```

## 6. Closed-loop play and benchmark

```bash
$P -m dxbench_policy.serve.server --backend lerobot --checkpoint outputs/train/act_bed03/checkpoints/last/pretrained_model --port 8765 &
$A dxbench/dxbench/scripts/run_episode.py --policy dxbench.policies.rby1_lerobot_remote.Rby1LerobotRemotePolicy \
   --policy_kw remote_port=8765 --out outputs/play/act_000 --video ...env args... --embodiment rby1_joint
$A -m dxbench.bench.run_benchmark benchmarks/dx_bed_v0/experiments/act_bed03.yaml
```
