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
    # BED-02 (2026-10-21): dropping while the 5-garment heap was still landing left a slope the top towel slid down,
    # off the desk front, in 10/10 hard states; landing 2 s first keeps every garment on the desk
    p.add_argument("--presettle_s", type=float, default=2.0, help="BED-02: let the spawned heap land before the drop")
    p.add_argument("--desk_lift", type=float, default=0.30, help="BED-02 drop height (m)")
    p.add_argument("--salt", type=int, default=0, help="new crumple draw (make_states.sh retries a rejected state)")


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
    elif difficulty in ("medium", "hard") and kind == "duvet":
        # crumple: shrink towards the centre and add a smooth random height field
        s = rng.uniform(0.55, 0.8)
        n[:, :2] = c[:2] + (n[:, :2] - c[:2]) * s
        k = rng.normal(size=(4, 2)) * 6.0
        ph = rng.uniform(0, 2 * np.pi, size=4)
        h = sum(np.sin(n[:, :2] @ k[i] + ph[i]) for i in range(4)) / 4
        n[:, 2] += 0.06 + 0.08 * (h - h.min()) / max(1e-6, np.ptp(h))
    return n


TRACE = bool(os.environ.get("DX_STATES_TRACE"))


def _trace(U, spec, tag):
    """Debug (DX_STATES_TRACE=1): garment centroids, lowest node and distance to the nearest robot body."""
    bodies = U.scene["robot"].data.body_pos_w.torch[0]
    out = []
    for o in spec.objects:
        if o.kind in ("tshirt", "towel"):
            n = U.scene[o.name].data.nodal_pos_w.torch[0]
            d = (n[:, None, :] - bodies[None]).norm(dim=-1).min().item()
            c = (n.mean(dim=0) - U.scene.env_origins[0]).tolist()
            out.append(f"{o.name} c=({c[0]:.2f},{c[1]:.2f},{c[2]:.2f}) zmin={n[:, 2].min().item():.2f} robot={d:.2f}")
    print(f"[trace] {tag}: " + " | ".join(out), flush=True)


def main():
    rt = ArenaRuntime(add_args=_args, argv=None)
    a = rt.args
    files = sorted({f for pat in a.scenes for f in glob.glob(pat)})
    import torch

    from dxbench.paths import REPO
    from dxbench.tasks.scene_spec import SceneSpec, load_layout

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
    rng = np.random.default_rng(spec.seed + 12345 + 7919 * a.salt)
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
    # medium / hard garments (D-41, 2026-10-21): a dropped-laundry heap. One random point is pulled up 45 cm (drifting
    # sideways) over 1 s with velocity pins written every physics step, then released to fall for 0.5 s. The old
    # shrink + bump field relaxed back to ~98 % flat; laying random folds by hand jittered (> 4 m/s after 8 s).
    if spec.task == "BED-02" and a.presettle_s > 0:
        for _ in range(int(a.presettle_s * 15)):
            env.step(act)
    drops = []
    if spec.difficulty in ("medium", "hard"):
        for o in spec.objects:
            if o.kind in ("tshirt", "towel") and not (o.variant or "").startswith("folded"):
                n0 = U.scene[o.name].data.nodal_pos_w.torch[0]
                c0 = n0.mean(dim=0)
                if spec.task == "BED-02":
                    # on the desk: lift a point near the middle 30 cm and drift <= 5 cm towards the desk interior (-x,
                    # the robot is on the +x side), so the garment falls onto itself, not over the front edge
                    near = torch.nonzero((n0[:, :2] - c0[:2]).norm(dim=-1) < 0.10).flatten()
                    k = int(near[int(rng.integers(len(near)))])
                    drift = torch.as_tensor([-abs(rng.normal(0, 0.03)), rng.normal(0, 0.02)], device=U.device,
                                            dtype=n0.dtype)
                    lift = a.desk_lift
                else:
                    k = int(rng.integers(len(n0)))
                    drift = torch.as_tensor(rng.normal(0, 0.08, 2), device=U.device, dtype=n0.dtype)
                    lift = 0.45
                ids = torch.nonzero((n0 - n0[k]).norm(dim=-1) < 0.03).flatten()
                drops.append((o.name, ids, n0[ids].clone(), drift, lift))
    if drops:
        sim, dt = U.sim, U.physics_dt
        n_lift, n_fall = int(1.0 / dt), int(0.5 / dt)
        for i in range(n_lift + n_fall):
            if i < n_lift:
                s_ = (i + 1) / n_lift
                for name, ids, p0, drift, lift in drops:
                    obj = U.scene[name]
                    pos = obj.data.nodal_pos_w.torch
                    vel = obj.data.nodal_vel_w.torch.clone()
                    tgt = p0.clone()
                    tgt[:, :2] += drift * s_
                    tgt[:, 2] += lift * s_
                    vel[0, ids] = (tgt - pos[0, ids]) / dt
                    obj.write_nodal_velocity_to_sim_index(vel)
            U.scene.write_data_to_sim()
            sim.step(render=False)
            U.scene.update(dt)
            if TRACE and i % 48 == 0:
                _trace(U, spec, f"drop {i * dt:.2f}s")
    steps = int((a.settle_s + (2.0 if drops else 0.0)) * 15)  # a dropped heap keeps collapsing for ~5 s
    for k in range(steps):
        env.step(act)
        if TRACE and k % 3 == 0:
            _trace(U, spec, f"settle {k / 15:.2f}s")
    if spec.task == "BED-02":
        # BED-02 is folding on the desk: a garment that slid off it while the heap collapsed is no start state
        desk_z = float(load_layout()["parts_aabb"]["CareDesk"][1][2])
        off = {o.name: float((((U.scene[o.name].data.nodal_pos_w.torch[0] - U.scene.env_origins[0])[:, 2])
                              < desk_z - 0.05).float().mean()) for o in spec.objects if o.kind in ("tshirt", "towel")}
        if max(off.values()) > 0.05:
            print(f"[states-reject] {spec.scene_id} salt {a.salt}: fraction below the desk top {off}")
            rt.close()
            return
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
