# DX bed benchmark v0 (`dx_bed_v0`)

The RB-Y1 performs three bedroom tasks in the DX Hub 11F CareRoom digital twin (Isaac Sim 6.1, Isaac Lab 3.0,
IsaacLab-Arena 0.3.1, PhysX). Source of the task definitions: "DX Hub 실내 서비스 벤치마크 침실·주방 Task 정의 v0.1".

## Tasks

| ID | Instruction | Success | Partial score |
|---|---|---|---|
| BED-01 | "Make the bed." / "침대 정리해 줘" | The duvet covers all of the mattress top (5 cm cells; a 3 cm rim along the edges, where the duvet bends over, is left out; with a pillow, also the pillow zone at the head - D-55, D-66) | covers 0.5, neat (covered and left/right overhang difference ≤ 15 cm) 0.5. **Fail** if ≥ 30 % of the duvet lies on the floor |
| BED-02 | "Fold the clothes." / "옷 개어 줘" | Garment flattened, its folds done (mirror pairs within 5 cm, D-41), on the stack spot (desk, 40 cm in from the front edge straight behind the garments, radius 12 cm; D-32). Towels (D-64): long edges together, both ends to the middle, then once more across the middle to a ~0.2 m square; the first three folds count once reached in the episode, the last by the final footprint (long side ≤ 0.375 × the flat length, area ≤ ¼ of the flat area) | flattened 0.3, each fold 0.15, on the stack 0.25, re-weighted to sum to 1 |
| BED-03 | "Put the clothes in the laundry basket." / "방에 있는 옷 정리해 줘" | Every loose garment is inside the basket and released (a garment still held by a hand does not count, D-60). No distractor (doll, book) and no folded towel is in it | fraction of garments in the basket − 0.2 per wrong item. **Fail** if the basket tips over (> 45°) |

Subtask labels for each task are in `dxbench/tasks/predicates.py` `TEXT`; the oracle and recorder use them.
The predicates are pure numpy and are tested in `dxbench/tests/test_predicates.py`.

## Difficulties (`dxbench/tasks/scene_gen.py`)

| Task | easy | medium | hard |
|---|---|---|---|
| BED-01 | Duvet slid 48-52 cm towards the far side (its near edge 0.19-0.23 m in from the open side), head edge at the mattress end, no pillow; robot crouched 0.35 m from the pull pose with both hands over the bed (D-63, D-66) | Head half folded back over the foot half (0.55-0.59) | Folded, then bundled on the foot half or on the open long edge (0.31-0.58) |
| BED-02 | Small towel flat on the desk; robot leaning at the desk with both hands over it (D-66) | One T-shirt crumpled on the desk | Pile of 3 shirts and 2 towels |
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
- Control 15 Hz; physics 480 Hz for BED-01 (DECISIONS D-11) and 120 Hz for BED-02 / BED-03 (D-59, D-62: the demos and
  VR teleoperation run at 120 Hz; cloth folding is rate-sensitive). The task flag `--physics_hz` overrides it.
- Cameras: head plus two wrist cameras, 480 × 640 RGB.
- **Assisted cloth grasp:** closing the gripper pins the cloth nodes within 2.5 cm of the TCP axis to the hand
  (5 cm until 2026-10-18; D-U11). This is the same scale as the attachment grasps of GarmentLab / GarmentPile
  (`docs/notes/cloth_grasp_survey.md`) and still differs from the real robot's friction grasp (stage 1 report, D-4/D-5).
- **Towels** use Young's modulus 1e5 (other cloth 1e6) so a fold stays folded (D-28, `reports/towel_fold/REPORT.md`).
  Their mesh diagonals are mirrored about the centre line across the long side, so the left and right side folds are
  mirror images (D-61: with Isaac Lab's one-way diagonals the right flap sprang open after the fold).

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

Oracles (privileged, `dxbench/policies/`): BED-01 easy (`oracle_bed01`, 10/10, D-63), BED-02 easy towel (`oracle_bed02`,
10/10 with the last fold, D-64) and BED-03 easy / near (`oracle_bed03`). BED-01 / BED-02 medium and hard have none yet.
