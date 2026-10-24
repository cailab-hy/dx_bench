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
import math
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


def shape_nodes(nodes: np.ndarray, kind: str, difficulty: str, task: str, rng: np.random.Generator,
                fold_at: float | None = None) -> np.ndarray:
    n = nodes.copy()
    c = n.mean(axis=0)
    if task == "BED-01" and kind == "duvet" and difficulty == "easy" and fold_at is not None and fold_at < 0:
        pass  # folded physically in main() (2026-10-22); kept for the old mirror fold
    elif task == "BED-01" and kind == "duvet" and difficulty == "easy":
        # (old) fold along the bed length: the part beyond `fold_at` (head side) is mirrored onto the foot side, 4 cm higher.
        # The 2.0 m duvet (2026-10-22) is folded at the mattress middle, its head edge 2 cm short of the mattress end
        ax = 1
        fold = c[ax] if fold_at is None else fold_at
        top = n[:, ax] > fold
        n[top, ax] = 2 * fold - n[top, ax]
        n[top, 2] += 0.04 + 0.02 * (n[top, ax] - n[top, ax].min()) / max(1e-6, np.ptp(n[top, ax]))
    elif difficulty in ("medium", "hard") and kind == "duvet" and fold_at is not None and fold_at < 0:
        pass  # crumpled physically in main() (2026-10-22): the shrink + bump teleport went NaN on the 2.0 m duvet
    elif difficulty in ("medium", "hard") and kind == "duvet":
        # (old) crumple: shrink towards the centre and add a smooth random height field
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
        if o.kind in ("tshirt", "towel", "duvet"):
            n = U.scene[o.name].data.nodal_pos_w.torch[0]
            dist = (n[:, None, :] - bodies[None]).norm(dim=-1).min(dim=0).values
            j = int(dist.argmin())
            c = (n.mean(dim=0) - U.scene.env_origins[0]).tolist()
            b = (bodies[j] - U.scene.env_origins[0]).tolist()
            out.append(f"{o.name} c=({c[0]:.2f},{c[1]:.2f},{c[2]:.2f}) zmin={n[:, 2].min().item():.2f} "
                       f"robot={dist[j].item():.2f} {U.scene['robot'].body_names[j]}@({b[0]:.2f},{b[1]:.2f},{b[2]:.2f})")
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
    # the env must start from the spawn shapes: since load_cloth_states really runs (D-42) a scene that already has
    # states would start from the old ones (2026-10-22: a 2 mm-gap tee state under the 2.5 mm offset bunched up 15 cm)
    import tempfile

    fresh = SceneSpec.from_json(path)
    for o in fresh.objects:
        o.cloth_state = None
    tmp = os.path.join(tempfile.mkdtemp(prefix="dx_states_"), os.path.basename(path))
    fresh.to_json(tmp)
    sys.argv = [sys.argv[0], *sys.argv[1:], "--external_environment_class_path",
                f"dxbench.tasks.bed_tasks:DxBed0{spec.task[-1]}Environment", env_name, "--scene", tmp,
                "--no_assist", "--embodiment", "rby1_joint", *os.environ.get("DX_STATES_ENV_ARGS", "").split()]
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
        mat = load_layout()["parts_aabb"]["Bed/Mattress"]
        shaped = shape_nodes(pos[0].cpu().numpy(), o.kind, spec.difficulty, spec.task, rng, fold_at=-1.0)
        pos[0] = torch.as_tensor(shaped, device=U.device, dtype=pos.dtype)
        obj.write_nodal_pos_to_sim_index(pos)
        obj.write_nodal_velocity_to_sim_index(torch.zeros_like(pos))
    # hold the robot still (rby1_joint default targets) while the garments settle
    from dxbench.embodiments.rby1_cfg import ARM_L, ARM_R, READY_ARMS

    def hold_step(act):
        """One env step (decimation physics steps) without the env's termination checks and auto reset: a BED-01
        failure / success fired mid-generation and reset the cloth to its spawn shape (2026-10-22)."""
        U.action_manager.process_action(act)
        for _ in range(U.cfg.decimation):
            U.action_manager.apply_action()
            U.scene.write_data_to_sim()
            U.sim.step(render=False)
            U.scene.update(U.physics_dt)

    act = torch.zeros(U.num_envs, 24, device=U.device)
    act[:, :14] = torch.tensor([[READY_ARMS.get(n, 0.0) for n in ARM_R + ARM_L]], device=U.device)
    act[:, 14:16] = 1.0
    # medium / hard garments (D-41, 2026-10-21): a dropped-laundry heap. One random point is pulled up 45 cm (drifting
    # sideways) over 1 s with velocity pins written every physics step, then released to fall for 0.5 s. The old
    # shrink + bump field relaxed back to ~98 % flat; laying random folds by hand jittered (> 4 m/s after 8 s).
    if spec.task == "BED-02" and a.presettle_s > 0:
        for _ in range(int(a.presettle_s * 15)):
            hold_step(act)
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

                def lift_path(p0, s_, drift=drift, lift=lift):
                    t = p0.clone()
                    t[:, :2] += drift * s_
                    t[:, 2] += lift * s_
                    return t

                drops.append((o.name, ids, n0[ids].clone(), lift_path, 1.0))
    if spec.task == "BED-03" and spec.difficulty == "second":
        # the first garment is already in the basket (D-49): spawned flat over it, it lay across the rim, so it is put
        # in like the oracle does - the point nearest its centre carried up to 0.95 m over the basket centre (the
        # garment hangs into the basket), then released
        bxy = torch.as_tensor(spec.obj("laundry_basket").xyz[:2], device=U.device) + U.scene.env_origins[0][:2]
        for o in spec.objects:
            if o.kind in ("tshirt", "towel") and o.xyz[2] > 0.3:
                n0 = U.scene[o.name].data.nodal_pos_w.torch[0]
                k = int((n0[:, :2] - n0[:, :2].mean(dim=0)).norm(dim=-1).argmin())
                ids = torch.nonzero((n0 - n0[k]).norm(dim=-1) < 0.03).flatten()
                goal = torch.cat([bxy.to(n0.dtype), torch.tensor([0.95], device=U.device, dtype=n0.dtype)])

                def put_path(p0, s_, goal=goal, c=n0[k].clone()):
                    return p0 + (goal - c) * s_

                drops.append((o.name, ids, n0[ids].clone(), put_path, 1.5))
    if spec.task == "BED-01" and spec.difficulty in ("medium", "hard"):
        # crumple the duvet physically like a person bundling it: three of its corners carried in 40 cm arcs onto one
        # spot (medium: near the bed middle; hard: on the open long edge, so the bundle hangs over the side). Two random
        # points pulled together left the 2.0 x 1.55 m duvet still covering the mattress (start score 0.49, 7/10
        # "spread", 2026-10-22).
        for _ in range(int(1.0 * 15)):
            hold_step(act)
        mat = load_layout()["parts_aabb"]["Bed/Mattress"]
        org = U.scene.env_origins[0]
        for o in spec.objects:
            if o.kind == "duvet":
                n0 = U.scene[o.name].data.nodal_pos_w.torch[0]
                c0 = n0.mean(dim=0)
                cy = 0.5 * (mat[0][1] + mat[1][1])
                if spec.difficulty == "medium":
                    g = [0.5 * (mat[0][0] + mat[1][0]) + float(rng.uniform(-0.15, 0.15)), cy + float(rng.uniform(-0.3, 0.1))]
                else:
                    g = [mat[0][0] + 0.05, cy + float(rng.uniform(-0.3, 0.3))]
                g = torch.as_tensor(g, device=U.device, dtype=n0.dtype) + org[:2]
                d = n0[:, :2] - c0[:2]
                corners = [int(torch.argmax(d @ torch.as_tensor(v, device=U.device, dtype=n0.dtype)))
                           for v in ((1, 1), (1, -1), (-1, 1), (-1, -1))]
                for k in [corners[i] for i in rng.permutation(4)[:3]]:
                    ids = torch.nonzero((n0 - n0[k]).norm(dim=-1) < 0.04).flatten()
                    pull = (g - n0[k, :2]) * float(rng.uniform(0.75, 0.95))

                    def gather(p0, s_, pull=pull):
                        t = p0.clone()
                        t[:, :2] += pull * s_
                        t[:, 2] += 0.4 * math.sin(math.pi * s_)  # up and back down on the gathered spot
                        return t

                    drops.append((o.name, ids, n0[ids].clone(), gather, 2.5))
    if spec.task == "BED-01" and spec.difficulty == "easy":
        # fold the duvet like a person (2026-10-22): pin its head-side edge, carry it in a 35 cm high arc (4 s) to the foot
        # side of the mattress middle, release. Writing mirrored positions instead left the render mesh wrong (a
        # strip, or nothing until the cloth moved) and crumpled the 65 x 65 duvet into a ball.
        mat = load_layout()["parts_aabb"]["Bed/Mattress"]
        fold = 0.5 * (mat[0][1] + mat[1][1]) + float(U.scene.env_origins[0][1])
        # the pillow lies on the duvet's head edge and rode along with it, dragging the whole duvet off the foot end:
        # park it on the floor beyond the bed while folding (its pose is not saved; the episode spawns it from the
        # scene JSON, clear of the folded duvet)
        if "pillow" in [o.name for o in spec.objects]:
            pl = U.scene["pillow"]
            pp = pl.data.nodal_pos_w.torch.clone()
            pp[0, :, 0] += 1.2
            pp[0, :, 2] += 0.01 - pp[0, :, 2].min()
            pl.write_nodal_pos_to_sim_index(pp)
            pl.write_nodal_velocity_to_sim_index(torch.zeros_like(pp))
        for k in range(int(2.0 * 15)):  # let the spawned sheet land on the mattress first (folding it mid-fall
            hold_step(act)              # bunched the duvet into a 1.0 x 0.5 m heap)
            if TRACE and k % 3 == 0:
                _trace(U, spec, f"presettle {k / 15:.2f}s")
        for o in spec.objects:
            if o.kind == "duvet":
                n0 = U.scene[o.name].data.nodal_pos_w.torch[0]
                ids = torch.nonzero(n0[:, 1] > n0[:, 1].max() - 0.02).flatten()

                def fold_path(p0, s_, fold=fold):
                    t = p0.clone()
                    ang = math.pi * s_
                    r = p0[:, 1] - fold  # rotate the edge about the fold line, 2 cm extra lift for the layer below
                    t[:, 1] = fold + r * math.cos(ang)
                    t[:, 2] = p0[:, 2] + 0.35 * math.sin(ang) + 0.02 * s_  # a flat 35 cm high arc
                    return t

                drops.append((o.name, ids, n0[ids].clone(), fold_path, 4.0))
    if drops:
        sim, dt = U.sim, U.physics_dt
        t_max = max(d[4] for d in drops)
        n_all = int((t_max + 0.5) / dt)
        for i in range(n_all):
            for name, ids, p0, fn, secs in drops:
                n_d = int(secs / dt)
                if i >= n_d:
                    continue
                s_ = min(1.0, (i + 1) / (0.85 * n_d))  # the last 15 %: hold at the target, then release
                s_ = s_ * s_ * (3 - 2 * s_)
                obj = U.scene[name]
                pos = obj.data.nodal_pos_w.torch
                vel = obj.data.nodal_vel_w.torch.clone()
                vel[0, ids] = (fn(p0, s_) - pos[0, ids]) / dt
                obj.write_nodal_velocity_to_sim_index(vel)
            U.scene.write_data_to_sim()
            sim.step(render=False)
            U.scene.update(dt)
            if TRACE and i % 48 == 0:
                _trace(U, spec, f"drop {i * dt:.2f}s")
    steps = int((a.settle_s + (2.0 if drops else 0.0)) * 15)  # a dropped heap keeps collapsing for ~5 s
    for k in range(steps):
        hold_step(act)
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
        vn = U.scene[o.name].data.nodal_vel_w.torch[0].norm(dim=-1)
        vel = vn.max().item()
        if vel > 1.0:  # where the fastest node is (debug for jitter at contacts)
            j = int(vn.argmax())
            print(f"[states-fast] {o.name}: {vel:.1f} m/s at {np.round(nodes[j], 3).tolist()}, "
                  f"nodes > 0.5 m/s: {int((vn > 0.5).sum())}", flush=True)
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
