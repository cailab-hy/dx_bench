# dx_bench

A VLA / world-action-model benchmark for the Rainbow Robotics RB-Y1 in the DX Hub 11F CareRoom digital twin. It
runs on Isaac Sim 6.1, Isaac Lab 3.0 and IsaacLab-Arena 0.3.1 with the PhysX backend. It covers three bedroom
tasks: BED-01 make the bed, BED-02 fold clothes, BED-03 laundry to the basket.

Get it (the RB-Y1 model and IsaacLab-Arena are submodules):

```bash
git clone --recursive https://github.com/cailab-hy/dx_bench.git
```

Not in this repository:
- the DX Hub 11F building twin (`DX_Hub_Building_add_obj_v1`, CareRoom USD): the tasks load it from
  `~/P2-DX_project` (`dxbench/dxbench/paths.py`); ask the lab for access;
- demonstration datasets, raw recordings, checkpoints and run logs (`data/`, `outputs/`, tens of GB);
- third-party photos and videos used while matching the cloth to real garments (product photos, YouTube frames,
  our own footage): the reports keep the numbers.

Most project notes (`PROGRESS.md`, `DECISIONS.md`, `HUMAN_TODO.md`, `CLAUDE.md`, the stage reports) are in Korean.

Start here:
- [`PROGRESS.md`](PROGRESS.md): status and repro commands
- [`DECISIONS.md`](DECISIONS.md): every design decision, with how to revert it
- [`HUMAN_TODO.md`](HUMAN_TODO.md): what needs a person
- [`CLAUDE.md`](CLAUDE.md): fixed constraints and simulator gotchas

## Layout

| Path | What |
|---|---|
| `dxbench/` | `dx_arena` package: assets, RB-Y1 embodiments (`rby1_joint`, `rby1_ik`), tasks, oracles, teleop + raw recorder, policy client, benchmark runner. Never imports lerobot |
| `dxbench_policy/` | `dx_policy` package: LeRobot v3 conversion / validation, websocket policy server (hold / replay / ACT / SmolVLA), ACT training wrapper |
| `benchmarks/dx_bed_v0/` | Fixed evaluation scenes (3 tasks × 3 difficulties × 10), settled cloth states, experiment YAMLs, `RESULTS.md` |
| `tools/` | Stage scripts: cloth spike, RB-Y1 regression (`rby1_check/`), state generation, oracle / demo / replay / stage 4–5 pipelines |
| `reports/` | Per-stage and per-study reports (`cloth_spike`, `rby1_check`, `stage3`–`stage5`, `stage_v2`, cloth material studies: `swing`, `crumple`, `duvet_match`, `tee_match`, `towel_match`) |
| `docs/` | `action_space.md`, `BENCHMARK.md`, `PIPELINE.md`, notes |
| `config/cloth_params.yaml` | Cloth simulation parameters (stage 1) |

## Installation

Two conda environments, kept apart on purpose: the simulator stack and LeRobot pin different torch / websockets
builds, so **never install lerobot into `dx_arena`**.

| Environment | Role | Key versions |
|---|---|---|
| `dx_arena` | simulation, teleop, raw recording, benchmark runner, policy client | Python 3.12, isaacsim 6.1.0.0, torch 2.11.0+cu128, Isaac Lab 17.0.2 (Arena submodule), IsaacLab-Arena 0.3.1, numpy 2.3.1, warp-lang 1.16.0, newton 1.5.2, websockets 16.1.1 |
| `dx_policy` | LeRobot conversion, training, policy server | Python 3.12, lerobot 0.6.1, torch 2.11.0 (see [`VERSIONS.md`](VERSIONS.md)) |

Tested on Ubuntu 22.04, NVIDIA driver 580, one RTX 6000 Ada (48 GB). The simulator alone runs on a smaller GPU;
π0.5 fine-tuning needs about 37 GB. Isaac Sim, the submodules and the conda environments take roughly 60 GB of disk.

> These commands rebuild the environments recorded in this repository (the installed package lists above); they
> have not been re-run on a clean machine yet. If a step pulls a different version of a pinned package, reinstall
> the pinned one.

### 1. Clone

IsaacLab-Arena's own submodules (Isaac Lab, Isaac-GR00T) use SSH URLs (`git@github.com:`), so on a machine
without a GitHub SSH key let git fetch them over HTTPS, and skip Isaac-GR00T (1.4 GB, not used here):

```bash
git config --global url."https://github.com/".insteadOf "git@github.com:"   # only without a GitHub SSH key
git clone https://github.com/cailab-hy/dx_bench.git
cd dx_bench
git submodule update --init                                                    # IsaacLab-Arena, rby1-sim-isaac
git -C submodules/IsaacLab-Arena submodule update --init submodules/IsaacLab   # Isaac Lab 3.0 (91a0906)
git submodule status --recursive    # Isaac-GR00T stays "-" (not checked out); everything else has a commit
```

`git clone --recursive` also works once the URL rewrite is set, but then downloads Isaac-GR00T too.

### 2. `dx_arena` (simulation)

Run every command from the repository root (`cd dx_bench`); the paths below are relative to it. The Isaac Lab line
runs in a subshell `( ... )` so it comes back to the root afterwards.

```bash
cd dx_bench
conda create -n dx_arena python=3.12 -y
conda activate dx_arena
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install "isaacsim[all,extscache]==6.1.0.0" --extra-index-url https://pypi.nvidia.com
export OMNI_KIT_ACCEPT_EULA=YES            # Isaac Sim EULA, also set by the scripts
# Isaac Lab 3.0 from the Arena submodule (installs its packages editable into the active env)
(cd submodules/IsaacLab-Arena/submodules/IsaacLab && ./isaaclab.sh --install)   # back in dx_bench afterwards
pip install -e submodules/IsaacLab-Arena   # IsaacLab-Arena 0.3.1 (release/0.3.1, unmodified)
pip install --no-deps -e dxbench           # this repository's simulation package
pip list | grep -E "^(isaacsim |torch |numpy |warp-lang|newton |websockets)"   # check the pins above
```

### 3. `dx_policy` (training and policy server)

```bash
conda create -n dx_policy python=3.12 -y
conda activate dx_policy
pip install "lerobot[training,smolvla]==0.6.1" websockets msgpack pytest
pip install --no-deps -e dxbench_policy
```

π0.5 (`lerobot/pi05_base`, 14.5 GB) also needs the gated PaliGemma tokenizer: accept the license at
<https://huggingface.co/google/paligemma-3b-pt-224> with your Hugging Face account, then `hf auth login`.

### 4. DX Hub building twin

The tasks load the CareRoom USD (`DX_Hub_Building_add_obj_v1`), which is not in this repository. Get it from the lab
and point `DX_USD_DIR` at its `DX/USD` folder (default `~/P2-DX_project/DX/USD`, see `dxbench/dxbench/paths.py`):

```bash
export DX_USD_DIR=/path/to/DX/USD
```

The RB-Y1 model comes from the `rby1-sim-isaac` submodule; nothing else needs downloading.

### 5. Check

```bash
conda activate dx_arena
python -m pytest -q dxbench/tests dxbench/dxbench/teleop/vr              # no simulator
python dxbench/dxbench/scripts/run_episode.py --policy dxbench.policies.oracle_bed03.Bed03OraclePolicy \
  --out outputs/check_000 --external_environment_class_path dxbench.tasks.bed_tasks:DxBed03Environment dx_bed03 \
  --scene benchmarks/dx_bed_v0/scenes/bed03_near_000.json             # one oracle episode, ~10-20 min
conda activate dx_policy
python -m pytest -q dxbench_policy/tests
```

The Kit app sometimes segfaults in its first seconds (`platforminfo`, ~8 %); the repository's runner scripts retry
up to three times.

## Quick start

```bash
cd ~/dx_bench
A=~/miniforge3/envs/dx_arena/bin/python
# one oracle episode with video (BED-03 Easy, evaluation scene 0)
$A dxbench/dxbench/scripts/run_episode.py --policy dxbench.policies.oracle_bed03.Bed03OraclePolicy \
   --out outputs/demo_000 --video \
   --external_environment_class_path dxbench.tasks.bed_tasks:DxBed03Environment dx_bed03 \
   --scene benchmarks/dx_bed_v0/scenes/bed03_easy_000.json
# a benchmark experiment (oracle / hold / replay / ACT YAMLs in benchmarks/dx_bed_v0/experiments)
$A -m dxbench.bench.run_benchmark benchmarks/dx_bed_v0/experiments/oracle_bed03.yaml
# tests (no simulator)
$A -m pytest -q dxbench/tests dxbench/dxbench/teleop/vr
~/miniforge3/envs/dx_policy/bin/python -m pytest -q dxbench_policy/tests
```

For data collection, conversion, training and closed-loop play, see [`docs/PIPELINE.md`](docs/PIPELINE.md). For the benchmark definition, see [`docs/BENCHMARK.md`](docs/BENCHMARK.md).

## VR teleoperation (Meta Quest 2 / 3)

Human demonstrations (BED-01 make the bed and BED-02 fold clothes need two-handed motions the scripted oracles do
not do well). The headset connects over USB; its browser opens a WebXR page served by the simulator, which shows
the robot's three camera views in VR and sends the controller poses back. Episodes are recorded in the same raw
format as the oracle demonstrations, so they go through the same LeRobot conversion and training.

### One-time setup

1. **adb on the PC:** `sudo apt install adb`
2. **Developer mode on the headset:** the headset's Meta account must belong to a developer organization
   (create one at <https://developer.oculus.com/manage/organizations/>). Then in the Meta Horizon phone app:
   Devices → your headset → Headset settings → Developer mode → on. Restart the headset.
3. **USB debugging:** connect the headset with a USB-C data cable, put it on and accept "Allow USB debugging"
   (tick "Always allow from this computer"). Check on the PC:
   ```bash
   adb devices        # the headset is listed as "device" (not "unauthorized")
   ```

### Scenes to record in

Training scenes and their cloth initial states live in `data/`, which is not in the repository; make them once
(seeds 10000+ never overlap the evaluation scenes):

```bash
conda activate dx_arena
python dxbench/dxbench/tasks/scene_gen.py --task BED-02 --difficulty easy --n 30 --seed0 10000 --split train --out data/scenes_train
tools/make_states.sh data/scenes_train/bed02_easy_train_*.json        # settled cloth states, ~3 min per scene
```

(`--task BED-01 / BED-02 / BED-03`, `--difficulty easy / medium / hard`; BED-03 also `near`, `second`.)

### Each session

```bash
conda activate dx_arena
adb reverse tcp:8012 tcp:8012                     # headset localhost:8012 -> PC (again after every reconnect)
python dxbench/dxbench/teleop/run_teleop.py --teleop_device vr --record data/raw/bed02_vr \
  --external_environment_class_path dxbench.tasks.bed_tasks:DxBed02Environment dx_bed02 \
  --scene data/scenes_train/bed02_easy_train_10000.json --physics_hz 360   # task flags go after the task name
```

- In the headset browser open **http://localhost:8012**. The page lists its checks (WebXR, connection, video);
  press **Enter VR**.
- Stand facing the way the robot should look forward and **hold Y for 1 s** (recenter).
- Press **A** to start recording, do the task. The episode is saved automatically when the task succeeds.
  Hold **B** to throw it away. Either way the scene resets for the next episode. An episode that reaches the task
  time limit (simulated time: BED-01 / BED-02 240 s, BED-03 180 s) without success is not kept.
- `--physics_hz 360` is fine for BED-02 / BED-03 (towel, T-shirt); keep the default 480 Hz for BED-01, where the
  large duvet drifts at 360 Hz (DECISIONS D-11). Other tasks: `DxBed01Environment dx_bed01` / `DxBed03Environment dx_bed03`.
- More flags: `--vr_port` (default 8012), `--max_episodes`, `--realtime` (pace to 15 Hz wall time when the sim is
  faster than real time).

| Control | Action |
|---|---|
| Grip (hold) | That arm follows the controller (clutch). Release, move your hand, grip again: no jump. The base does not drive while a grip is held |
| Trigger | Close the gripper (release: open) |
| Right stick | Drive the base: up / down forward / back, left / right turn; diagonal = both. Gentle push = slow, full = top speed; within 15° of straight up / sideways it drives exactly straight / turns in place; starts and stops are ramped |
| Right stick click | Slow / fast driving (slow 0.1 m/s, 0.3 rad/s for lining up at a garment or the basket; fast 0.3 m/s, 0.6 rad/s) |
| Left stick up / down | Torso one posture step deeper / more upright (crouch to reach the desk or the floor) |
| Left stick left / right | Camera panel size |
| A | Start recording |
| B (hold) | Discard the episode |
| X | Pause |
| Y (hold 1 s) | Recenter: your forward = robot forward, the panel in front of you |

Notes:
- With three cameras and cloth the simulation runs at about half real time (360 Hz): move slowly and let the
  robot catch up. The panel's status lines show recording, the torso posture and the base mode / speed.
- To carry a garment, hold it with the trigger, release the grip, then drive (the base is locked while a grip is held).
- If a controller loses tracking, that arm stops; the grippers keep their state, so a held garment is not dropped.
- Grasping is the assisted pad grasp (`pad_grasp`): close the gripper with the fingertips touching the cloth, between
  the two pads.

### Check without a headset

A scripted operator plays the controllers through the same WebSocket. `scripted_vr` checks the arm clutch (recenter,
start, grip, move the right hand 10 cm forward / up and 5 cm sideways, trigger); `scripted_vr_drive` checks the
sticks (forward 4 s, turn 2 s, slow mode 3 s, grip lock, one torso step):

```bash
python dxbench/dxbench/teleop/run_teleop.py --teleop_device scripted_vr --record outputs/teleop_test --save_any \
  --max_episodes 1 --external_environment_class_path dxbench.tasks.bed_tasks:DxBed03Environment dx_bed03 \
  --scene benchmarks/dx_bed_v0/scenes/bed03_near_000.json
# prints scripted={"tcp_offset": [...], "expected": [0.1, -0.05, 0.1], "err_m": ...}: err_m should be well under 1 cm
# --teleop_device scripted_vr_drive prints forward_m ~1.2, turn_rad ~-1.1, slow_m ~0.3, locked_m 0, torso_steps 1
```

### After recording

```bash
conda activate dx_policy
python -m dxbench_policy.convert.raw_to_lerobot --raw data/raw/bed02_vr --out data/lerobot/bed02_vr_dx24 --fmt dx24 --image_scale 0.5
python -m dxbench_policy.convert.validate --out data/lerobot/bed02_vr_dx24 --raw data/raw/bed02_vr
```

Training and evaluation: [`docs/PIPELINE.md`](docs/PIPELINE.md) sections 5–6. Keyboard teleoperation (GUI) is in
section 2 there.
