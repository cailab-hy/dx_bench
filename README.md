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

```bash
conda create -n dx_arena python=3.12 -y
conda activate dx_arena
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install "isaacsim[all,extscache]==6.1.0.0" --extra-index-url https://pypi.nvidia.com
export OMNI_KIT_ACCEPT_EULA=YES            # Isaac Sim EULA, also set by the scripts
# Isaac Lab 3.0 from the Arena submodule (installs its packages editable into the active env)
(cd submodules/IsaacLab-Arena/submodules/IsaacLab && ./isaaclab.sh --install)
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
