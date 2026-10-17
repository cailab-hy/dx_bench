"""Make the settled cloth initial states of scene JSONs ("drop and settle"), so evaluation resets are deterministic.

    python dxbench/dxbench/tasks/make_cloth_states.py --scenes benchmarks/dx_bed_v0/scenes/bed03_easy_*.json \
        --states benchmarks/dx_bed_v0/states

For each scene: build the task env (robot idle, assisted grasp off), optionally shape the garments
(BED-01 easy: duvet folded in half at the foot; "medium" garments: crumpled with a smooth random field), let
everything settle for --settle_s seconds, save each garment's world node positions to <states>/<scene>_<name>.npz
and write the path into the scene JSON (`cloth_state`). Cloth simulation differs from seed to seed, which is why
evaluation scenes load these files instead of re-simulating the drop.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

from dxbench.utils.arena_runtime import ArenaRuntime


def _args(p):
    p.add_argument("--scenes", nargs="+", required=True)
    p.add_argument("--states", required=True)
    p.add_argument("--settle_s", type=float, default=4.0)


def shape_nodes(nodes: np.ndarray, kind: str, difficulty: str, task: str, rng: np.random.Generator) -> np.ndarray:
    n = nodes.copy()
    c = n.mean(axis=0)
    if task == "BED-01" and kind == "duvet" and difficulty == "easy":
        # fold in half along the bed length: the head-side half is mirrored onto the foot half, 4 cm higher
        ax = 1
        fold = c[ax]
        top = n[:, ax] > fold
        n[top, ax] = 2 * fold - n[top, ax]
        n[top, 2] += 0.04 + 0.02 * (n[top, ax] - n[top, ax].min()) / max(1e-6, np.ptp(n[top, ax]))
    elif difficulty in ("medium", "hard") and kind in ("tshirt", "towel", "duvet"):
        # crumple: shrink towards the centre and add a smooth random height field
        s = rng.uniform(0.55, 0.8)
        n[:, :2] = c[:2] + (n[:, :2] - c[:2]) * s
        k = rng.normal(size=(4, 2)) * 6.0
        ph = rng.uniform(0, 2 * np.pi, size=4)
        h = sum(np.sin(n[:, :2] @ k[i] + ph[i]) for i in range(4)) / 4
        n[:, 2] += 0.06 + 0.08 * (h - h.min()) / max(1e-6, np.ptp(h))
    return n


def main():
    rt = ArenaRuntime(add_args=_args, argv=None)
    a = rt.args
    files = sorted({f for pat in a.scenes for f in glob.glob(pat)})
    import torch

    from dxbench.paths import REPO
    from dxbench.tasks.scene_spec import SceneSpec

    out_dir = (REPO / a.states) if not os.path.isabs(a.states) else a.states
    os.makedirs(out_dir, exist_ok=True)
    # one Kit process per scene file is the robust option (env rebuild in-process leaks); this script handles one
    assert len(files) == 1, "pass exactly one scene per process (see tools/make_states.sh)"
    path = files[0]
    spec = SceneSpec.from_json(path)
    env_name = {"BED-01": "dx_bed01", "BED-02": "dx_bed02", "BED-03": "dx_bed03"}[spec.task]
    sys.argv = [sys.argv[0], *sys.argv[1:], "--external_environment_class_path",
                f"dxbench.tasks.bed_tasks:DxBed0{spec.task[-1]}Environment", env_name, "--scene", path,
                "--no_assist", "--embodiment", "rby1_joint"]
    rt._argv = sys.argv[1:]
    env = rt.make_env()
    U = env.unwrapped
    env.reset()
    rng = np.random.default_rng(spec.seed + 12345)
    for o in spec.objects:
        if o.kind not in ("tshirt", "towel", "sock", "duvet"):
            continue
        obj = U.scene[o.name]
        pos = obj.data.nodal_pos_w.torch.clone()
        shaped = shape_nodes(pos[0].cpu().numpy(), o.kind, spec.difficulty, spec.task, rng)
        pos[0] = torch.as_tensor(shaped, device=U.device, dtype=pos.dtype)
        obj.write_nodal_pos_to_sim_index(pos)
        obj.write_nodal_velocity_to_sim_index(torch.zeros_like(pos))
    # hold the robot still (rby1_joint default targets) while the garments settle
    from dxbench.embodiments.rby1_cfg import ARM_L, ARM_R, READY_ARMS

    act = torch.zeros(U.num_envs, 24, device=U.device)
    act[:, :14] = torch.tensor([[READY_ARMS.get(n, 0.0) for n in ARM_R + ARM_L]], device=U.device)
    act[:, 14:16] = 1.0
    steps = int(a.settle_s * 15)
    for _ in range(steps):
        env.step(act)
    report = {}
    for o in spec.objects:
        if o.kind not in ("tshirt", "towel", "sock", "duvet"):
            continue
        nodes = (U.scene[o.name].data.nodal_pos_w.torch[0] - U.scene.env_origins[0]).cpu().numpy()
        vel = U.scene[o.name].data.nodal_vel_w.torch[0].norm(dim=-1).max().item()
        f = os.path.join(out_dir, f"{spec.scene_id}_{o.name}.npz")
        np.savez_compressed(f, nodes=nodes.astype(np.float32))
        o.cloth_state = os.path.relpath(f, REPO)
        report[o.name] = {"finite": bool(np.isfinite(nodes).all()), "z_min": float(nodes[:, 2].min()),
                          "z_max": float(nodes[:, 2].max()), "max_speed": vel}
    spec.to_json(path)
    print(f"[states] {spec.scene_id}: {report}")
    rt.close()


if __name__ == "__main__":
    main()
