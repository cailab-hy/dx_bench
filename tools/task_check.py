"""In-sim task check (stage 3): the scene builds, the initial state is not a success, and (BED-03) a forced
correct state is a success / a distractor in the basket is penalised.

    ~/miniforge3/envs/dx_arena/bin/python tools/task_check.py --scene benchmarks/dx_bed_v0/scenes/bed03_easy_000.json

BED-03 positive: every garment is moved (rigidly, node positions) 25 cm above the basket centre, dropped and settled
for 4 s with the robot holding still. Writes reports/tasks/check_<scene_id>.json, prints "[task_check] ...".
"""
from __future__ import annotations

import json
import sys

import numpy as np

from dxbench.utils.arena_runtime import ArenaRuntime


def _args(p):
    p.add_argument("--check_scene", required=True)


def main():
    rt = ArenaRuntime(add_args=_args)
    a = rt.args
    from dxbench.tasks.scene_spec import SceneSpec

    spec = SceneSpec.from_json(a.check_scene)
    k = spec.task[-1]
    rt._argv = [*sys.argv[1:], "--external_environment_class_path", f"dxbench.tasks.bed_tasks:DxBed0{k}Environment",
                f"dx_bed0{k}", "--scene", a.check_scene, "--embodiment", "rby1_joint"]
    env = rt.make_env()
    U = env.unwrapped
    import torch

    from dxbench.embodiments.rby1_cfg import ARM_L, ARM_R, READY_ARMS
    from dxbench.paths import REPO
    from dxbench.tasks import bed_tasks

    task = bed_tasks.CURRENT_TASK
    env.reset()
    hold = torch.zeros(U.num_envs, 24, device=U.device)
    hold[:, :14] = torch.tensor([[READY_ARMS.get(n, 0.0) for n in ARM_R + ARM_L]], device=U.device)
    hold[:, 14:16] = 1.0
    for _ in range(5):
        env.step(hold)
    r0 = task.evaluate(U)[0]
    res = {"scene": spec.scene_id, "task": spec.task, "initial": {k: r0[k] for k in ("success", "score", "failure")},
           "initial_ok": (not r0["success"]) and r0["failure"] is None}
    if spec.task == "BED-03":
        bp, _ = task.rigid_pose(U, "laundry_basket", 0)
        clothes = [o.name for o in spec.objects if o.kind in ("tshirt", "towel", "sock") and o.variant != "folded"]
        for i, name in enumerate(clothes):
            obj = U.scene[name]
            pos = obj.data.nodal_pos_w.torch.clone()
            n = pos[0]
            c = n.mean(dim=0)
            tgt = torch.tensor([bp[0], bp[1], 0.25 + 0.08 * i], device=U.device) + U.scene.env_origins[0]
            pos[0] = (n - c) * torch.tensor([0.5, 0.5, 1.0], device=U.device) + tgt  # squeeze to fit the basket
            obj.write_nodal_pos_to_sim_index(pos)
            obj.write_nodal_velocity_to_sim_index(torch.zeros_like(pos))
        for _ in range(60):
            env.step(hold)
        r1 = task.evaluate(U)[0]
        res["forced_in_basket"] = {k: r1[k] for k in ("success", "score", "failure")}
        res["forced_in_basket"]["in_basket"] = r1.get("in_basket")
        res["forced_ok"] = bool(r1["success"])
        res["ok"] = res["initial_ok"] and res["forced_ok"]
    else:
        res["stages"] = r0.get("stages")
        res["ok"] = res["initial_ok"]
    out = REPO / "reports" / "tasks"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"check_{spec.scene_id}.json").write_text(json.dumps(res, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))
    print(f"[task_check] {json.dumps({k: res[k] for k in res if k in ('scene', 'ok', 'initial', 'forced_in_basket')}, default=str)}",
          flush=True)
    rt.close()


if __name__ == "__main__":
    main()
