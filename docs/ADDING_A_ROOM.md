# Adding a room or a task family (e.g. the DX kitchen)

The bedroom tasks are one task family in one room. The room-independent parts are shared (DECISIONS D-65):

| Part | Where | Shared as is |
|---|---|---|
| Robot (USD, gains, `rby1_ik` / `rby1_joint` actions, torso presets, base) | `dxbench/embodiments/` | yes |
| Teleop (keyboard, Quest VR), raw recording | `dxbench/teleop/` | yes |
| Episode runner, policy client | `dxbench/scripts/run_episode.py`, `dxbench/policies/` | yes |
| LeRobot conversion / validation, training, policy server | `dxbench_policy/` | yes |
| Env factory from a scene JSON, shared events (solver iterations, cloth states), task base, task registry, observer camera | `dxbench/tasks/task_env.py` | yes |
| Room: background asset + prestartup events | `dxbench/assets/scenes.py` (`ROOMS`) | add one |
| Room layout (furniture AABBs, room box) | `dxbench/tasks/<room>_layout.json` from `extract_layout.py --room <Room>` | add one |
| Task classes, predicates, scene generator, objects, oracles | `tasks/bed_tasks.py`, `tasks/predicates.py`, `tasks/scene_gen.py`, `assets/objects.py`, `policies/oracle_bed0*.py` | bedroom only |

## Steps for a kitchen

1. **Layout.** `python dxbench/dxbench/tasks/extract_layout.py --room Kitchen --parts <furniture sub-prims>` writes
   `dxbench/tasks/kitchen_layout.json` (DX USD `DX_Hub_Building_add_obj_v1_Kitchen.usda`, `Furniture/Kitchen`, placements
   from `add_obj_v1_furniture.json`). The DX scene is read-only: the script only reads it.
2. **Room.** In `assets/scenes.py` add a `Background` subclass for the kitchen USD (prim under `/World/envs/env_0/...`, see
   `DxCareRoom`: cloth only renders on colliders inside the env, D-44), its prestartup events (static furniture with
   `fix_furniture(root=...)`, the floor collider swapped for a plane like `replace_care_room_floor` if it is a thin scaled
   cube, D-22, lighting variants) and register it: `ROOMS["kitchen"] = RoomSpec("dx_kitchen", kitchen_prestartup)`.
3. **Tasks.** New module `tasks/kitchen_tasks.py`:
   - task classes `class KitchenTaskBase(RoomTaskBase): room = "kitchen"` with `evaluate(env)` returning one dict per env
     (`success`, `failure` (None or a reason), `score`, `stages`) — keep the scoring itself pure numpy in a predicates
     module with unit tests;
   - `build_assets(spec)` mapping scene objects to Arena assets;
   - `register_task("KIT-01", TaskEntry(Kit01Task, episode_s, physics_hz, build_assets, observer_eye))`;
   - `class DxKit01Environment(RoomEnvironmentBase): name = "dx_kit01"; task_key = "KIT-01"`.
   Run with `--external_environment_class_path dxbench.tasks.kitchen_tasks:DxKit01Environment dx_kit01 --scene ...`.
4. **Scenes.** A generator writing scene JSONs (`SceneSpec`: task id, robot start, objects, lighting, language) under
   `benchmarks/<suite>/scenes/`, like `tasks/scene_gen.py`; use `scene_spec.load_layout("kitchen")` for free-floor checks.

## Things the bedroom work learned that apply to a new room

- Keep every non-robot rigid body at `objects.SOLVER_ITERS` (the PhysX GPU solver runs the scene-wide maximum, D-53/54);
  `solver_iterations` already does it for anything spawned in the scene.
- Dynamic DX furniture makes cloth in the room buzz: make it static at prestartup (D-44).
- The assisted grasp (`pad_grasp`) is for cloth only. Rigid objects (dishes, cups) need a physical friction grasp or a
  new rigid assist — decide this first, it shapes the oracles and the teleop.
- The physics rate changes manipulation results (cloth folding: 120 vs 480 Hz, D-58/59): collect demos and evaluate at
  the task's registered rate.
- Decisions of the kitchen work: own numbering (`K-01`, ...) in `DECISIONS.md` so parallel branches do not clash.
