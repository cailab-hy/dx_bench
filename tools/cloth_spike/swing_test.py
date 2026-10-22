"""How much does a carried cloth keep swinging? (2026-10-22, user: sim cloth flaps like a flag, real cloth barely swings)

    ~/miniforge3/envs/dx_arena/bin/python tools/cloth_spike/swing_test.py --cloth towel --ld 0 --bd 0.1 --ed 0.1 --tag x

No robot. The benchmark cloth (towel 0.8 x 0.4 m 120 g E 1e5 / wine tee 140 g E 3e4 / duvet 2.0 x 1.55 m 1.2 kg E 1e5)
lies flat on the floor; a virtual pinch (nodes within 2.5 cm of a corner / the tee's shoulder, velocity pins like the
assisted grasp) lifts it 0.6 m in 1.2 s, carries it 0.4 m sideways in 1.2 s (the robot's arm / base speed) and stops.
Metrics over the 4 s hold after the stop: swing of the free part (centroid of the nodes > 20 cm below the pinch,
horizontal distance from its final rest position): peak, number of half-swings > 2 cm, time until it stays < 1 cm;
mean node speed in the last second. Damping: --ld linear damping (1/s, a crude air drag, deformable body),
--bd / --ed material bend / elasticity damping. --video saves a side view.
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--cloth", default="towel", choices=["towel", "tee", "duvet"])
ap.add_argument("--ld", type=float, default=0.0, help="linear damping (1/s)")
ap.add_argument("--bd", type=float, default=0.1, help="material bend damping")
ap.add_argument("--ed", type=float, default=0.1, help="material elasticity damping")
ap.add_argument("--hz", type=int, default=480)
ap.add_argument("--video", action="store_true")
ap.add_argument("--motion", default="robot", choices=["smooth", "robot"],
                help="robot: after the lift, turn 90 deg about an axis 0.5 m away (0.3 m/s) with the target updated at "
                     "15 Hz like the policy, then stop; smooth: 0.4 m straight carry")
ap.add_argument("--lift", type=float, default=0.6, help="lift height (m)")
ap.add_argument("--fric", type=float, default=None, help="cloth friction override (static; dynamic = 0.83x)")
ap.add_argument("--tag", default=None)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.enable_cameras = bool(args.video)
app = AppLauncher(args).app

import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg, DeformableObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab_physx.renderers import IsaacRtxRendererCfg  # noqa: E402
from isaaclab_physx.sim.schemas.schemas_cfg import PhysxCollisionCfg  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "dxbench"))
from dxbench.assets.thick_cloth import grid_rectangle_spawn, split_single_mesh_cloth  # noqa: E402

TAG = args.tag or f"{args.cloth}_ld{args.ld:g}_bd{args.bd:g}_ed{args.ed:g}"
OUT = REPO / "reports" / "swing" / TAG
OUT.mkdir(parents=True, exist_ok=True)
DT = 1.0 / args.hz
HALF = 0.0025


def T(x):
    if hasattr(x, "torch"):
        return x.torch
    return wp.to_torch(x) if isinstance(x, wp.array) else x


def cloth_cfg():
    props = sim_utils.DeformableBodyPropertiesCfg(solver_position_iteration_count=16, self_collision=True,
                                                  linear_damping=args.ld)
    if args.cloth == "tee":
        return sim_utils.UsdFileCfg(usd_path=str(REPO / "dxbench/dxbench/assets/objects/tshirt_m95_wine.usda"))
    size, mass, E, th, rgb = ((0.8, 0.4), 0.120, 1.0e5, 0.0015, (0.4, 0.33, 0.42)) if args.cloth == "towel" else \
        ((2.0, 1.55), 1.2, 1.0e5, 0.002, (0.85, 0.85, 0.84))
    cfg = sim_utils.MeshRectangleCfg(
        size=size, edge_refinement=1.0, deformable_props=props,
        physics_material=sim_utils.PhysxSurfaceDeformableBodyMaterialCfg(
            density=mass / (size[0] * size[1] * th), static_friction=args.fric or 0.9,
            dynamic_friction=0.83 * (args.fric or 0.9), youngs_modulus=E,
            poissons_ratio=0.25, surface_thickness=th, bend_damping=args.bd, elasticity_damping=args.ed),
        collision_props=[PhysxCollisionCfg(contact_offset=0.008, rest_offset=HALF)],
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb))
    cfg.func = grid_rectangle_spawn(0.023)
    return cfg


def build():
    @configclass
    class SceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.6, dynamic_friction=0.5)))
        light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))
        cloth = DeformableObjectCfg(prim_path="{ENV_REGEX_NS}/Cloth", spawn=cloth_cfg(),
                                    init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.01)))
        cam = (CameraCfg(prim_path="{ENV_REGEX_NS}/cam", update_period=0.0, width=640, height=480, data_types=["rgb"],
                         spawn=sim_utils.PinholeCameraCfg(focal_length=14.0, clipping_range=(0.05, 10.0)),
                         renderer_cfg=IsaacRtxRendererCfg()) if args.video else None)

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args.device, dt=DT, render_interval=32))
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=5.0))
    from pxr import Usd

    stage = sim_utils.get_current_stage()
    if args.cloth == "tee":  # the tee USD: set the damping on its prims (it carries its own material)
        root = stage.GetPrimAtPath("/World/envs/env_0/Cloth")
        mesh = next(p for p in Usd.PrimRange(root) if p.HasAPI("OmniPhysicsSurfaceDeformableSimAPI"))
        split_single_mesh_cloth(stage, str(mesh.GetPath()), HALF)
        mesh.GetAttribute("physxCollision:restOffset").Set(HALF)
        mesh.GetAttribute("physxDeformableBody:linearDamping").Set(float(args.ld))
        for p in Usd.PrimRange(root):
            for name, v in (("physxDeformableMaterial:bendDamping", args.bd),
                            ("physxDeformableMaterial:elasticityDamping", args.ed)):
                a = p.GetAttribute(name)
                if a and a.IsValid():
                    a.Set(float(v))
    sim.reset()
    return sim, scene


def main():
    sim, scene = build()
    cl = scene["cloth"]
    cam = scene["cam"] if args.video else None
    o = scene.env_origins[0]
    frames = []

    def nodes():
        return T(cl.data.nodal_pos_w)[0] - o

    def step(k, record=False):
        sim.step(render=bool(record))
        scene.update(DT)
        if record:
            frames.append(T(cam.data.output["rgb"])[0, ..., :3].cpu().numpy().astype(np.uint8))

    if cam is not None:
        cam.set_world_poses_from_view((o + torch.tensor([0.3, -2.0, 0.9], device=o.device))[None],
                                      (o + torch.tensor([0.25, 0.25, 0.45], device=o.device))[None])
    for _ in range(int(1.0 / DT)):
        step(0)
    n0 = nodes().clone()
    if args.cloth == "tee":
        pick = torch.tensor([-0.20, float(n0[:, 1].max()) - 0.04], device=o.device)  # a shoulder
    else:
        pick = n0[:, :2].min(0).values + 0.02  # a corner
    ids = torch.nonzero((n0[:, :2] - pick).norm(dim=-1) < 0.025).flatten()
    p0 = n0[ids].clone()
    lift, carry, hold = 1.2, (2.6 if args.motion == "robot" else 1.2), 4.0
    ctrl = int(round(1.0 / 15 / DT))
    N1, N2, N3 = int(lift / DT), int(carry / DT), int(hold / DT)
    every = max(1, round(args.hz / 15))
    track, prev, carry_rel = [], None, []
    plog = []  # all node positions every step from 0.2 s before the lift end: finite-difference analysis
    for k in range(N1 + N2 + N3):
        if k < N1:
            s = (k + 1) / N1
            off = torch.tensor([0.0, 0.0, args.lift * (s * s * (3 - 2 * s))], device=o.device)
        elif k < N1 + N2:
            if args.motion == "robot":  # 15 Hz zero-order hold of a constant-speed turn (like the base / arm commands)
                kk = (k - N1) // ctrl * ctrl
                ang = (math.pi / 2) * min(1.0, (kk + ctrl) / N2)
                off = torch.tensor([0.5 * math.sin(ang), 0.5 * (1 - math.cos(ang)), args.lift], device=o.device)
            else:
                s = (k - N1 + 1) / N2
                off = torch.tensor([0.4 * (s * s * (3 - 2 * s)), 0.0, args.lift], device=o.device)
        else:
            off = torch.tensor([0.5, 0.5, args.lift] if args.motion == "robot" else [0.4, 0.0, args.lift], device=o.device)
        tgt = p0 + off
        pos = T(cl.data.nodal_pos_w)
        vel = T(cl.data.nodal_vel_w).clone()
        v_ff = (tgt - prev) / DT if prev is not None else torch.zeros_like(tgt)
        prev = tgt.clone()
        vel[0, ids] = (tgt + o - pos[0, ids]) / DT + v_ff
        cl.write_nodal_velocity_to_sim_index(vel)
        step(k, record=args.video and k % every == 0)
        if k >= N1 - int(0.2 / DT):
            plog.append(nodes().clone())
        if N1 <= k < N1 + N2 and k % 4 == 0:
            n = nodes()
            v = T(cl.data.nodal_vel_w)[0]
            free = n[:, 2] < float(tgt[:, 2].mean()) - 0.2
            if free.any():
                carry_rel.append(float((v[free] - v[ids].mean(0)).norm(dim=-1).mean()))
        if k >= N1 + N2 and k % 4 == 0:
            n = nodes()
            hang = n[n[:, 2] < float(tgt[:, 2].mean()) - 0.2]
            if len(hang):
                track.append([k * DT, *hang[:, :2].mean(0).tolist(), float(T(cl.data.nodal_vel_w)[0].norm(dim=-1).mean())])
    P = torch.stack(plog).cpu().numpy()  # (T, N, 3), every physics step
    pin = P[:, ids.cpu().numpy()].mean(1)
    rel = P - pin[:, None]  # node position relative to the pinch
    t0 = int(0.2 / DT)
    carry_sl, hold_sl = slice(t0, t0 + N2), slice(t0 + N2, None)
    free0 = P[t0, :, 2] < pin[t0, 2] - 0.2  # hanging nodes at the lift end

    def fd(lag, sl):  # mean speed of those nodes relative to the pinch, differenced over lag steps
        r = rel[sl][:, free0]
        return float(np.linalg.norm(r[lag:] - r[:-lag], axis=-1).mean() / (lag * DT))
    k30 = int(round(args.hz / 30))
    fdm = {"rel_v_carry_step": fd(1, carry_sl), "rel_v_carry_30hz": fd(k30, carry_sl),
           "rel_v_carry_5hz": fd(int(args.hz / 5), carry_sl),
           "rel_v_hold_step": fd(1, hold_sl), "rel_v_hold_30hz": fd(k30, hold_sl),
           "n_hanging": int(free0.sum()), "n_nodes": int(P.shape[1])}
    np.save(OUT / "rel_hanging.npy", rel[:, free0][:, ::max(1, int(free0.sum()) // 40)])
    tr = np.array(track)
    rest = tr[-int(0.5 / (4 * DT)):, 1:3].mean(0)  # where the free part ends up
    dist = np.linalg.norm(tr[:, 1:3] - rest, axis=1)
    t = tr[:, 0] - tr[0, 0]
    # half-swings: sign changes of the x offset with |offset| > 2 cm between them
    xo = tr[:, 1] - rest[0]
    half, last = 0, 0.0
    for v in xo:
        if abs(v) > 0.02:
            if last and np.sign(v) != np.sign(last):
                half += 1
            last = v
    settle = float(t[np.nonzero(dist > 0.01)[0][-1]]) if (dist > 0.01).any() else 0.0
    res = {"args": vars(args), "peak_swing_m": float(dist.max()), "half_swings_gt_2cm": half,
           "settle_s_1cm": settle, "mean_speed_last_s": float(tr[t > t[-1] - 1.0, 3].mean()),
           "carry_rel_speed": float(np.mean(carry_rel)) if carry_rel else None,
           "carry_rel_speed_p90": float(np.percentile(carry_rel, 90)) if carry_rel else None, **fdm}
    (OUT / "result.json").write_text(json.dumps(res, indent=1))
    np.save(OUT / "track.npy", tr)
    if frames:
        import imageio

        imageio.mimsave(OUT / "side.mp4", frames, fps=15, codec="libx264", quality=8, macro_block_size=1)
    print(f"[swing] {TAG}: {json.dumps({k: v for k, v in res.items() if k != 'args'})}", flush=True)
    app.close()


if __name__ == "__main__":
    main()
