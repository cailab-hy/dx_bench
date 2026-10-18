"""Towel fold springback test (D-U8): does a folded towel stay folded after the assisted grasp releases it?

    ~/miniforge3/envs/dx_arena/bin/python tools/cloth_spike/towel_fold.py --tag base
    ... --E 3e5 --thickness 0.001 --bend 0.0 --tag soft

No robot (default --sequence right): a virtual TCP grasps the towel like `cloth_assist` (nodes in a --radius cylinder around the TCP axis,
5 cm below to 3 cm above, pinned by velocity writes) 4 cm in from the right short edge, lifts 3 cm, carries it on the
fold arc of `oracle_bed02` (height = half the fold distance + 3 cm) to 3 cm past the centre line, lowers it onto the
towel, holds, releases and lets the towel settle. Towel = the benchmark small towel (`assets/objects.py` rect_cloth:
0.5 x 0.3 m, 2.3 cm nodes, 16 iterations, self collision) on a desk (top 0.725 m), 480 Hz.
The fold predicate is BED-02's `fold_left` proxy: long extent <= 0.75 x flat length.
Writes reports/towel_fold/runs/<tag>.json.
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--E", type=float, default=1.0e6, help="Young's modulus (Pa)")
ap.add_argument("--thickness", type=float, default=0.0015, help="surface thickness (m)")
ap.add_argument("--bend", type=float, default=None, help="surface_bend_stiffness (unset = PhysX default)")
ap.add_argument("--bend_damping", type=float, default=0.1)
ap.add_argument("--poisson", type=float, default=0.25)
ap.add_argument("--density", type=float, default=120.0)
ap.add_argument("--radius", type=float, default=0.025, help="assist capture radius (m)")
ap.add_argument("--grasp", default="mid", choices=["mid", "corner", "two"],
                help="pinch the short edge at its middle, at one corner, or at both corners at once (two hands)")
ap.add_argument("--sequence", default="right", help="fold moves in order: right,left (short edges), half2 (near long "
                "edge by both corners -> far long edge), half1 (same, middle only)")
ap.add_argument("--arc_h", type=float, default=None, help="fold arc height (m); default max(0.10, D/2 + 0.03)")
ap.add_argument("--small", type=int, default=1, help="1 = small towel 0.5 x 0.3 (BED-02 eval), 0 = 0.7 x 0.45")
ap.add_argument("--hz", type=int, default=480)
ap.add_argument("--settle_s", type=float, default=4.0)
ap.add_argument("--tag", default="base")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
import warp as wp  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg, DeformableObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab_physx.sim.schemas.schemas_cfg import PhysxCollisionCfg  # noqa: E402

OUT = Path(os.path.expanduser("~/dx_bench/reports/towel_fold/runs"))
OUT.mkdir(parents=True, exist_ok=True)
TOP = 0.725
SIZE = (0.5, 0.3) if args.small else (0.7, 0.45)
DT = 1.0 / args.hz


def T(x):
    if hasattr(x, "torch"):
        return x.torch
    return wp.to_torch(x) if isinstance(x, wp.array) else x


def build():
    mat = dict(density=args.density, static_friction=0.72, dynamic_friction=0.6, youngs_modulus=args.E,
               poissons_ratio=args.poisson, surface_thickness=args.thickness, bend_damping=args.bend_damping,
               elasticity_damping=0.1)
    if args.bend is not None:
        mat["surface_bend_stiffness"] = args.bend
    diag = math.hypot(*SIZE)
    spawn = sim_utils.MeshRectangleCfg(
        size=SIZE, edge_refinement=max(1.0, diag / (1.45 * 0.023)),
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(solver_position_iteration_count=16, self_collision=True),
        physics_material=sim_utils.PhysxSurfaceDeformableBodyMaterialCfg(**mat),
        collision_props=[PhysxCollisionCfg(contact_offset=0.008, rest_offset=0.002)],
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.9, 0.7)))

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
        light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2500.0))
        desk = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Desk",
            spawn=sim_utils.CuboidCfg(size=(0.8, 1.0, TOP), collision_props=sim_utils.CollisionPropertiesCfg(),
                                      physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.6,
                                                                                      dynamic_friction=0.5)),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, TOP / 2)))
        towel = DeformableObjectCfg(prim_path="{ENV_REGEX_NS}/Towel", spawn=spawn,
                                    init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, TOP + 0.01)))

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args.device, dt=DT, render_interval=32))
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=3.0))
    sim.reset()
    return sim, scene


def extent(n):  # long (x) / short (y) extent of the node cloud and max height above the desk
    return (float(n[:, 0].max() - n[:, 0].min()), float(n[:, 1].max() - n[:, 1].min()), float(n[:, 2].max() - TOP))


def main():
    sim, scene = build()
    towel = scene["towel"]
    log, t = [], 0.0

    def step(n=1):
        nonlocal t
        for _ in range(n):
            sim.step()
            scene.update(DT)
            t += DT

    def nodes():
        return T(towel.data.nodal_pos_w)[0] - scene.env_origins[0]

    step(int(1.5 / DT))  # settle flat
    n0 = nodes()
    flat = extent(n0)
    # stretch proxy: each node's 4 nearest neighbours in the flat state; strain = |d / d0 - 1| over those pairs
    d0m = torch.cdist(n0, n0)
    d0m.fill_diagonal_(1e9)
    nn = d0m.topk(4, largest=False).indices
    d0 = (n0[:, None, :] - n0[nn]).norm(dim=-1)

    def strain(n):
        d = (n[:, None, :] - n[nn]).norm(dim=-1)
        e = (d / d0 - 1.0).abs()
        return round(float(e.max()), 3), round(float(e.mean()), 4)

    max_strain = [0.0, 0.0]
    surf = float(n0[:, 2].max())
    speed = 0.012 * 15  # m/s of path, as oracle_bed02 (1.2 cm per 15 Hz step)

    def plan(move, n):
        """Grasp points (all ride one TCP path) and the reference put point for a fold move on the current cloth."""
        x_lo, x_hi, y_lo, y_hi = (float(n[:, 0].min()), float(n[:, 0].max()), float(n[:, 1].min()),
                                  float(n[:, 1].max()))
        xm, ym = float(n[:, 0].mean()), float(n[:, 1].mean())
        if move in ("right", "left"):  # short edge (x_hi / x_lo) -> 3 cm past the centre line
            x = x_hi - 0.04 if move == "right" else x_lo + 0.04
            ys = {"mid": [ym], "corner": [y_lo + 0.03], "two": [y_lo + 0.03, y_hi - 0.03]}[args.grasp]
            pts = [(x, y) for y in ys]
            put = (xm - 0.03 if move == "right" else xm + 0.03, ys[0])
        elif move in ("half2", "half1"):  # near long edge (y_lo) -> far long edge, both corners or the middle
            pts = [(x_lo + 0.04, y_lo + 0.04), (x_hi - 0.04, y_lo + 0.04)] if move == "half2" else [(xm, y_lo + 0.04)]
            put = (pts[0][0], y_hi - 0.03)
        else:
            raise ValueError(move)
        return pts, put

    def fold(move):
        n = nodes()
        pts, put_xy = plan(move, n)
        def local_top(x, y, r=0.04):  # cloth surface height around (x, y), not the global maximum (crumples)
            d = (n[:, :2] - torch.tensor([x, y], device=n.device)).norm(dim=-1)
            return float(n[d < r, 2].max()) if bool((d < r).any()) else float(n[:, 2].min())

        pick = torch.tensor([pts[0][0], pts[0][1], local_top(*pts[0])], device=n.device)
        put = torch.tensor([put_xy[0], put_xy[1], local_top(*put_xy) + 0.012], device=n.device)
        sel, offs = [], []
        for (x, y) in pts:  # node offsets relative to the reference TCP (first grasp point)
            p_ = torch.tensor([x, y, local_top(x, y)], device=n.device)
            loc = n - p_
            m = torch.nonzero((loc[:, :2].norm(dim=-1) < args.radius) & (loc[:, 2] > -0.05) & (loc[:, 2] < 0.03)).flatten()
            sel.append(m)
            offs.append(loc[m] + (p_ - pick))
        ids, off = torch.cat(sel), torch.cat(offs)
        D = float((put[:2] - pick[:2]).norm())
        h = args.arc_h if args.arc_h else max(0.10, 0.5 * D + 0.03)
        path = [pick + torch.tensor([0, 0, 0.03 * s], device=n.device) for s in torch.linspace(0, 1, int(0.3 / DT))]
        for s in torch.linspace(0, 1, max(2, int((D + 2 * h) / speed / DT))):
            s = float(s)
            q = pick + (put - pick) * s
            q[2] = pick[2] + 0.03 + (put[2] - pick[2] - 0.03) * s + h * math.sin(math.pi * s)
            path.append(q)
        path += [put] * int(0.3 / DT)
        prev = None
        for i, tgt_tcp in enumerate(path):
            pos = T(towel.data.nodal_pos_w)
            vel = T(towel.data.nodal_vel_w).clone()
            tgt = tgt_tcp + scene.env_origins[0] + off
            v_ff = (tgt - prev) / DT if prev is not None else torch.zeros_like(tgt)
            prev = tgt.clone()
            vel[0, ids] = (tgt - pos[0, ids]) / DT + v_ff
            towel.write_nodal_velocity_to_sim_index(vel)
            step()
            if i % int(0.05 / DT) == 0:
                st = strain(nodes())
                max_strain[0], max_strain[1] = max(max_strain[0], st[0]), max(max_strain[1], st[1])
        held = extent(nodes())
        return {"move": move, "grasp_points": len(pts), "grasped_nodes": int(len(ids)), "held_extent": held}

    moves = [m for m in args.sequence.split(",") if m]
    steps_log, n_held = [], None
    for k, mv in enumerate(moves):
        r = fold(mv)
        if k == len(moves) - 1:
            n_held = nodes().cpu().numpy()
        step(int((args.settle_s if k == len(moves) - 1 else 1.5) / DT))  # release (stop writing) and settle
        r["after_release_extent"] = extent(nodes())
        steps_log.append(r)
        log.append(r)
    n_end = nodes()
    finite = bool(torch.isfinite(n_end).all())
    end = extent(n_end)
    L, W = SIZE
    pred = {"fold_left": end[0] <= 0.75 * L, "fold_right": end[0] <= 0.5 * L + 0.05, "fold_half": end[1] <= 0.6 * W,
            "size_ok": max(end[0], end[1]) <= 0.37 and min(end[0], end[1]) <= 0.32}
    res = {"tag": args.tag, "sequence": moves, "grasp": args.grasp,
           "material": {"E": args.E, "thickness": args.thickness, "bend": args.bend, "bend_damping": args.bend_damping,
                        "poisson": args.poisson, "density": args.density},
           "radius": args.radius, "towel": SIZE, "hz": args.hz, "flat_extent": flat, "moves": steps_log,
           "end_extent": end, "predicates_end": pred, "carry_strain_max_mean": max_strain, "finite": finite}
    held0 = steps_log[0]["held_extent"]
    if moves == ["right"]:  # single side fold: springback summary as before
        res.update({"held_extent": held0, "springback_m": round(end[0] - held0[0], 4),
                    "folded_while_held": held0[0] <= 0.75 * L, "folded_end": end[0] <= 0.75 * L})
    (OUT / f"{args.tag}.json").write_text(json.dumps(res, indent=1))
    import numpy as np
    np.savez_compressed(OUT / f"{args.tag}_nodes.npz", flat=n0.cpu().numpy(), held=n_held, end=n_end.cpu().numpy())
    print(f"[towel_fold] {json.dumps({k: res[k] for k in ('tag', 'sequence', 'end_extent', 'predicates_end', 'carry_strain_max_mean', 'finite')})}", flush=True)
    app.close()


if __name__ == "__main__":
    main()
