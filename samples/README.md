# Sample oracle demonstrations

`oracle_d70_dx24/` holds one scripted-oracle demonstration per task (BED-01, BED-02, BED-03), as a LeRobot v3 dataset
(codebase v3.0, 15 fps, 17 MB), so the data format can be inspected without running the simulator. The full training
sets (D-70: about 120 / 121 / 83 + 43 episodes) are not in the repository; `tools/collect_demos.sh` and
`dxbench_policy.convert.raw_to_lerobot` regenerate them (docs/PIPELINE.md).

| Episode | Task | Instruction | Frames | Success at frame | Scene (data/scenes_train_d70) |
|---|---|---|---|---|---|
| 0 | BED-01 make the bed | "Spread the blanket over the bed." | 345 | 264 | bed01_easy_train_10057 |
| 1 | BED-02 fold the towel | "책상 위 빨래 개 줘" | 1182 | 1113 | bed02_easy_train_10053 |
| 2 | BED-03 laundry basket | "바닥에 있는 옷 바구니에 넣어 줘" | 1768 | 1543 | bed03_easy_train_10008 |

Each episode was chosen as the median-length one of its task, without an injected failure.

## Contents

- `observation.state`, `action` (24-D, `dx24`): right arm 7, left arm 7 (absolute joint positions, rad), gripper right /
  left (action 1 = open), torso 6 (rad), base forward speed (m/s) and turn rate (rad/s). See docs/action_space.md.
- `observation.images.front` / `right` / `left`: head, right wrist and left wrist cameras, 240 x 320 (rendered 480 x 640,
  stored at half size), H.264 video.
- `dx.*` (not policy inputs): `dx.stage_id` subtask index into the task's subtask list (dxbench/tasks/predicates.py
  `TEXT`; the last entry, "release and step back", marks the frames recorded after success), base pose, TCP poses.
- `meta/dx_episodes/*.json`: the raw recording metadata of each episode - the scene's randomized values
  (`scene_spec`: layout, robot start, garment variants, lighting, instruction), the oracle's strategy and injected
  failure (`oracle_meta`), `success_frame` and the number of frames recorded after it (`tail_frames`, D-69).

The demonstrations come from a privileged oracle (it reads the cloth nodes) in the DX Hub care-room digital twin
(Isaac Sim 6.1 / Isaac Lab 3.0, PhysX); the cloth grasp is the assisted pad grasp (D-U12). Load with

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("dxbench/oracle_d70_sample", root="samples/oracle_d70_dx24")
```
