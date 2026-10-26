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

## Environments

- **`dx_arena`** (simulation, recording, policy client):
  - pinned isaacsim 6.1.0.0, torch 2.11.0+cu128, numpy 2.3.1, warp-lang 1.16.0, newton 1.5.2, websockets 16.1.1
  - plus `dxbench` (`pip install --no-deps -e dxbench`)
- **`dx_policy`** (LeRobot 0.6.1, training, policy server): see [`VERSIONS.md`](VERSIONS.md).

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
