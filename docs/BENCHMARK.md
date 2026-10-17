# DX bed benchmark v0 (`dx_bed_v0`)

The RB-Y1 performs three bedroom tasks in the DX Hub 11F CareRoom digital twin (Isaac Sim 6.1, Isaac Lab 3.0,
IsaacLab-Arena 0.3.1, PhysX). Source of the task definitions: "DX Hub 실내 서비스 벤치마크 침실·주방 Task 정의 v0.1".

## Tasks

| ID | Instruction | Success | Partial score |
|---|---|---|---|
| BED-01 | "Make the bed." / "침대 정리해 줘" | The duvet covers ≥ 90 % of the mattress top. The left/right overhang differs by ≤ 15 cm. The pillow is within 20 cm of the headboard centre. The pillow was moved | pillow cleared 0.1, duvet spread 0.5, even 0.2, pillow placed 0.2. **Fail** if the duvet touches the floor |
| BED-02 | "Fold the clothes." / "옷 개어 줘" | Garment flattened, 3 folds done, fold bbox ≤ 35 × 30 cm (+2 cm), on the stack spot | flattened 0.3, each fold 0.15, on the stack 0.25 |
| BED-03 | "Put the clothes in the laundry basket." / "방에 있는 옷 정리해 줘" | Every loose garment is inside the basket. No distractor (doll, book) and no folded towel is in it | fraction of garments in the basket − 0.2 per wrong item. **Fail** if the basket tips over (> 45°) |

Subtask labels for each task are in `dxbench/tasks/predicates.py` `TEXT`; the oracle and recorder use them.
The predicates are pure numpy and are tested in `dxbench/tests/test_predicates.py`.

## Difficulties (`dxbench/tasks/scene_gen.py`)

| Task | easy | medium | hard |
|---|---|---|---|
| BED-01 | Duvet folded at the foot of the bed; pillow at the head | Duvet crumpled on the bed; pillow displaced | Half the duvet on the floor; pillow on the floor |
| BED-02 | Small towel flat on the desk | One T-shirt crumpled on the desk | Pile of 3 shirts and 2 towels |
| BED-03 | Basket + 2 garments on the open floor, each within 2.0 m of the basket | 3 garments on the floor + 1 shirt on the bed | 4 on the floor + shirt and towel on the bed + distractors (doll, book, folded towel) |

## Splits

- **Evaluation:** `benchmarks/dx_bed_v0/scenes/<task>_<difficulty>_<seed>.json`, 10 fixed scenes per task × difficulty (90 in total), seeds 0–9.
- **Training / demonstration:** `data/scenes_train/`, seeds 10000+. Seeds never overlap with evaluation.
- **Garment variants (OOD):**
  - Training uses T-shirts `m_blue` and `s_red`, and full-size towels.
  - Evaluation uses the unseen T-shirts `l_gray` (115 % scale) and `m_white`, and small towels.
- **Cloth initial states:** "drop and settle" node positions in `benchmarks/dx_bed_v0/states/*.npz`. Every reset loads them, so evaluation scenes are deterministic (`tools/make_states.sh`).

## Robot and interfaces

- Embodiments `rby1_joint` (24-D, policies) and `rby1_ik` (17-D, teleop and oracles): see `docs/action_space.md`.
- Control 15 Hz; physics 480 Hz (DECISIONS D-11).
- Cameras: head plus two wrist cameras, 480 × 640 RGB.
- **Assisted cloth grasp:** closing the gripper pins the cloth nodes within 5 cm of the TCP to the hand. This differs from the real robot (stage 1 report, D-4/D-5).

## Metrics (per task × difficulty)

| Metric | Definition |
|---|---|
| success rate | Episodes whose task predicate says success at the end (or earlier: success terminates the episode) |
| score mean | Mean partial score at the end of the episode |
| stage completion | Fraction of episodes in which each predicate stage is true at the end |
| stage time | Mean sim time to first reach each stage |
| tipped | Episodes where the robot base tilt exceeded 10° (safety) |
| errors | Episodes without a result (crash / timeout) |
| sim s / wall s | Episode length and wall time |

Episode limits (sim time): BED-01 240 s, BED-02 240 s, BED-03 180 s. Learned-policy and replay experiments are capped
at 1800 steps (120 s) to bound compute (DECISIONS D-23); oracle demos need 1340–1851 steps on BED-03 Easy.

## Running

```bash
cd ~/dx_bench
# one experiment (YAML: policy, policy_kw, env args, tasks x difficulties)
~/miniforge3/envs/dx_arena/bin/python -m dxbench.bench.run_benchmark benchmarks/dx_bed_v0/experiments/oracle_bed03.yaml
# a learned policy served from dx_policy (the YAML's `server:` starts it)
~/miniforge3/envs/dx_arena/bin/python -m dxbench.bench.run_benchmark benchmarks/dx_bed_v0/experiments/act_bed03.yaml
# report only
~/miniforge3/envs/dx_arena/bin/python -m dxbench.bench.run_benchmark <yaml> --aggregate_only
```

Each experiment writes `outputs/bench/<name>/`:
- `<scene>/result.json` (one per episode)
- `summary.json`
- `REPORT.md`

## Experiment config

`benchmarks/dx_bed_v0/experiments/*.yaml` (the runner docstring in `dxbench/bench/run_benchmark.py` lists every key):

| YAML | Policy | Embodiment | Cap | Notes |
|---|---|---|---|---|
| `oracle_bed03.yaml` | `Bed03OraclePolicy` (privileged: reads cloth nodes) | `rby1_ik` | task limit | upper reference |
| `hold_bed03.yaml` | server `--backend hold` (holds the current posture) | `rby1_joint` | 300 steps | zero baseline |
| `replay_bed03.yaml` | server `--backend replay` of training demo 0 (open loop) | `rby1_joint` | 1800 | blind baseline: shows how much scene variation matters |
| `act_bed03.yaml` | server `--backend lerobot` ACT trained on the 24 oracle demos | `rby1_joint` | 1800 | first learned policy, mp4s on |

The runner re-runs scenes whose result is missing or an error (`--rerun` forces all), starts scenes `parallel` at a time
with staggered starts, and retries Kit start-up crashes up to three times.

## Results

`benchmarks/dx_bed_v0/RESULTS.md` holds the current table (regenerate with `python -m dxbench.bench.make_results`).
Sample results on BED-03 Easy (10 evaluation scenes, 2026-10-17):

| Experiment | Success | Score mean | Notes |
|---|---|---|---|
| `oracle_bed03` | 8/10 | 0.90 | Upper reference; both failures are the second garment |
| `hold_bed03` | 0/10 | 0.00 | Zero baseline |
| `replay_bed03` | 0/10 | 0.00 | One training demo replayed open loop; knocks the basket over once. Scenes differ enough that blind replay fails |
| `act_bed03` | 0/10 | 0.00 | ACT, 20k steps on 24 oracle demos; drives past the garments. Undertrained (`reports/stage5/REPORT.md`) |
| `act_bed03_100k` | 0/10 | 0.00 | Same data, 100k steps; in one scene it carries a garment into the basket (40 s) and pulls it out again |

Only BED-03 has an oracle so far; BED-01 / BED-02 scenes, predicates and cloth states are ready (HUMAN_TODO 8).
