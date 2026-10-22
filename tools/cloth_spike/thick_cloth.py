"""Cloth thickness spike (2026-10-22): does a 5 mm render shell + 2.5 mm rest offset fix the paper-thin look and the
towel layers that vanish into each other / the desk?

    ~/miniforge3/envs/dx_arena/bin/python tools/cloth_spike/thick_cloth.py --variant thick --tag thick

No robot. A towel (0.8 x 0.4 m, 120 g) on a desk is folded in half (one short edge pinched, carried over the other,
released), a second towel on the floor is crumpled (centre lifted 30 cm, dropped), and the M(95) T-shirt is dropped
next to it. Variants:
  base  : current assets (render mesh = sheet, rest offset 2 mm)
  thick : render shell +-2.5 mm along the normals (assets/thick_cloth.py), rest offset 2.5 mm on every cloth
Saves close-up renders (grazing, oblique, top) and layer metrics to reports/cloth_thickness/<tag>/.
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--variant", default="thick", choices=["base", "thick"])
ap.add_argument("--half", type=float, default=0.0025, help="half thickness (m) of the shell / rest offset")
ap.add_argument("--surface_thickness", type=float, default=0.0015, help="FEM surface thickness (bending, mass)")
ap.add_argument("--hz", type=int, default=480)
ap.add_argument("--video", action="store_true")
ap.add_argument("--reload", action="store_true", help="also test writing a saved folded state back (episode path)")
ap.add_argument("--tee_rest", type=float, default=None, help="tee rest offset (default: --half)")
ap.add_argument("--tee_filter", type=float, default=None, help="tee self-collision filter distance")
ap.add_argument("--tag", default=None)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402
from PIL import Image  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg, DeformableObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab_physx.renderers import IsaacRtxRendererCfg  # noqa: E402
from isaaclab_physx.sim.schemas.schemas_cfg import PhysxCollisionCfg  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "dxbench"))
from dxbench.assets.thick_cloth import split_single_mesh_cloth, thicken_render_mesh  # noqa: E402

TAG = args.tag or args.variant
OUT = REPO / "reports" / "cloth_thickness" / TAG
OUT.mkdir(parents=True, exist_ok=True)
TOP = 0.725
SIZE = (0.80, 0.40)
DT = 1.0 / args.hz
LAVENDER = tuple(((c / 255 + 0.055) / 1.055) ** 2.4 for c in (172, 148, 175))
PINK = tuple(((c / 255 + 0.055) / 1.055) ** 2.4 for c in (227, 201, 210))
REST = args.half if args.variant == "thick" else 0.002


def T(x):
    if hasattr(x, "torch"):
        return x.torch
    return wp.to_torch(x) if isinstance(x, wp.array) else x


def towel_cfg(rgb):
    density = 0.120 / (SIZE[0] * SIZE[1] * args.surface_thickness)
    return sim_utils.MeshRectangleCfg(
        size=SIZE, edge_refinement=max(1.0, math.hypot(*SIZE) / (1.45 * 0.023)),
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(solver_position_iteration_count=16, self_collision=True),
        physics_material=sim_utils.PhysxSurfaceDeformableBodyMaterialCfg(
            density=density, static_friction=0.9, dynamic_friction=0.75, youngs_modulus=1.0e5, poissons_ratio=0.25,
            surface_thickness=args.surface_thickness, bend_damping=0.1, elasticity_damping=0.1),
        collision_props=[PhysxCollisionCfg(contact_offset=0.008, rest_offset=REST)],
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb, roughness=0.9))


def build():
    @configclass
    class SceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(
            color=(0.62, 0.6, 0.58), physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.9,
                                                                                   dynamic_friction=0.75)))
        light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))
        sun = AssetBaseCfg(prim_path="/World/sun", spawn=sim_utils.DistantLightCfg(intensity=2500.0, angle=1.0),
                           init_state=AssetBaseCfg.InitialStateCfg(rot=(0.2, 0.3, 0.0, 0.93)))
        desk = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Desk",
            spawn=sim_utils.CuboidCfg(size=(1.2, 0.9, TOP), collision_props=sim_utils.CollisionPropertiesCfg(),
                                      visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.42, 0.38)),
                                      physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.9,
                                                                                      dynamic_friction=0.75)),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, TOP / 2)))
        towel = DeformableObjectCfg(prim_path="{ENV_REGEX_NS}/Towel", spawn=towel_cfg(LAVENDER),
                                    init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, TOP + 0.005)))
        floor_towel = DeformableObjectCfg(prim_path="{ENV_REGEX_NS}/FloorTowel", spawn=towel_cfg(PINK),
                                          init_state=DeformableObjectCfg.InitialStateCfg(pos=(1.3, 0.0, 0.005)))
        tee = DeformableObjectCfg(
            prim_path="{ENV_REGEX_NS}/Tee",
            spawn=sim_utils.UsdFileCfg(usd_path=str(REPO / "dxbench/dxbench/assets/objects/tshirt_m95_green.usda")),
            init_state=DeformableObjectCfg.InitialStateCfg(pos=(1.3, 0.8, 0.02)))
        cam = CameraCfg(prim_path="{ENV_REGEX_NS}/cam", update_period=0.0, width=960, height=720, data_types=["rgb"],
                        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.02, 6.0)),
                        renderer_cfg=IsaacRtxRendererCfg())

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args.device, dt=DT, render_interval=32))
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=4.0))
    if args.variant == "thick":
        import isaaclab.sim.utils as su
        from pxr import Sdf, Usd, UsdPhysics

        stage = sim_utils.get_current_stage()
        for name in ("Towel", "FloorTowel"):
            assert thicken_render_mesh(stage, f"/World/envs/env_0/{name}", args.half), name
        tee_root = stage.GetPrimAtPath("/World/envs/env_0/Tee")
        mesh = next(p for p in Usd.PrimRange(tee_root) if p.HasAPI("OmniPhysicsSurfaceDeformableSimAPI"))
        split_single_mesh_cloth(stage, str(mesh.GetPath()), args.half)
        mesh.GetAttribute("physxCollision:restOffset").Set(args.tee_rest if args.tee_rest is not None else args.half)
        if args.tee_filter is not None:
            mesh.CreateAttribute("physxDeformableBody:selfCollisionFilterDistance", Sdf.ValueTypeNames.Float).Set(
                args.tee_filter)
        print("[thick_cloth] tee attrs", {a.GetName(): a.Get() for a in mesh.GetAttributes()
                                         if "physx" in a.GetName()}, flush=True)
    sim.reset()
    return sim, scene


def main():
    sim, scene = build()
    towel, ft, tee, cam = scene["towel"], scene["floor_towel"], scene["tee"], scene["cam"]
    o = scene.env_origins[0]
    frames = []

    def step(n=1, render=False):
        for i in range(n):
            sim.step(render=render and i == n - 1)
            scene.update(DT)

    def nodes(obj):
        return T(obj.data.nodal_pos_w)[0] - o

    def view(eye, tgt):
        cam.set_world_poses_from_view((o + torch.tensor(eye, device=o.device, dtype=torch.float32))[None],
                                      (o + torch.tensor(tgt, device=o.device, dtype=torch.float32))[None])

    def snap(name, eye, tgt):
        view(eye, tgt)
        for _ in range(3):
            step(1, render=True)
        Image.fromarray(T(cam.data.output["rgb"])[0, ..., :3].cpu().numpy().astype(np.uint8)).save(OUT / f"{name}.png")

    def pin_move(obj, ids, start, path, secs, hold=1.5):
        steps = int(secs / DT)
        prev = None
        for k in range(steps + int(hold / DT)):
            s = min(1.0, (k + 1) / steps)
            s = s * s * (3 - 2 * s)
            tgt = start + torch.tensor(path(s), device=start.device, dtype=start.dtype)
            pos = T(obj.data.nodal_pos_w)
            vel = T(obj.data.nodal_vel_w).clone()
            v_ff = (tgt - prev) / DT if prev is not None else torch.zeros_like(tgt)
            prev = tgt.clone()
            if k < steps:
                vel[0, ids] = (tgt + o - pos[0, ids]) / DT + v_ff
                obj.write_nodal_velocity_to_sim_index(vel)
            step(1)

    step(int(1.5 / DT))
    n0 = nodes(towel).clone()
    # fold the desk towel in half across its long axis: pinch the +x short edge, carry it over to -x, release
    x_lo, x_hi = float(n0[:, 0].min()), float(n0[:, 0].max())
    ids = torch.nonzero(n0[:, 0] > x_hi - 0.02).flatten()
    st = n0[ids].clone()
    L = x_hi - x_lo
    pin_move(towel, ids, st, lambda s: (-L * s + 0.0 * s, 0.0, 0.12 * math.sin(math.pi * s)), 2.5, hold=0.0)
    step(int(2.0 / DT))
    # crumple the floor towel: lift its centre 30 cm, drop
    f0 = nodes(ft).clone()
    c = f0.mean(0)
    fid = torch.nonzero((f0[:, :2] - c[:2]).norm(dim=-1) < 0.03).flatten()
    pin_move(ft, fid, f0[fid].clone(), lambda s: (0.05 * s, 0.0, 0.30 * s), 1.0, hold=0.0)
    step(int(3.0 / DT))

    n = nodes(towel)
    f = nodes(ft)
    t = nodes(tee)
    # layer gap of the folded towel: for nodes of the folded-over half, z distance to the nearest node of the lower half
    top_half = n0[:, 0] > (x_lo + x_hi) / 2
    from scipy.spatial import cKDTree

    lo_pts = n[~top_half].cpu().numpy()
    hi_pts = n[top_half].cpu().numpy()
    d, j = cKDTree(lo_pts[:, :2]).query(hi_pts[:, :2])
    over = d < 0.01
    gap = hi_pts[over, 2] - lo_pts[j[over], 2]
    res = {"args": vars(args), "rest_offset": REST,
           "towel_min_z_above_desk_mm": float((n[:, 2].min() - TOP) * 1000),
           "fold_layer_gap_mm": {"median": float(np.median(gap) * 1000) if len(gap) else None,
                                 "p5": float(np.percentile(gap, 5) * 1000) if len(gap) else None,
                                 "frac_below_3mm": float((gap < 0.003).mean()) if len(gap) else None},
           "floor_towel_min_z_mm": float(f[:, 2].min() * 1000), "tee_min_z_mm": float(t[:, 2].min() * 1000),
           "tee_max_z_mm": float(t[:, 2].max() * 1000), "tee_max_speed": float(T(tee.data.nodal_vel_w)[0].norm(dim=-1).max()),
           "towel_max_speed": float(T(towel.data.nodal_vel_w)[0].norm(dim=-1).max())}
    if args.reload:
        # the episode path: write a saved (folded) state into the sim, as load_cloth_states does
        folded = T(towel.data.nodal_pos_w).clone()
        flat = n0 + o
        p_ = T(towel.data.nodal_pos_w).clone(); p_[0] = flat
        towel.write_nodal_pos_to_sim_index(p_); towel.write_nodal_velocity_to_sim_index(torch.zeros_like(p_))
        step(int(1.0 / DT))
        snap("reload_flat", (0.55, -0.6, TOP + 0.45), (0.0, 0.0, TOP))
        towel.write_nodal_pos_to_sim_index(folded); towel.write_nodal_velocity_to_sim_index(torch.zeros_like(folded))
        step(6)
        snap("reload_folded_6", (0.55, -0.6, TOP + 0.45), (0.0, 0.0, TOP))
        step(int(1.0 / DT))
        snap("reload_folded_1s", (0.55, -0.6, TOP + 0.45), (0.0, 0.0, TOP))
    snap("desk_grazing", (0.75, -0.55, TOP + 0.06), (0.0, 0.0, TOP + 0.01))
    snap("desk_oblique", (0.55, -0.6, TOP + 0.45), (0.0, 0.0, TOP))
    snap("desk_top", (0.02, 0.0, TOP + 0.9), (0.0, 0.0, TOP))
    snap("floor_oblique", (1.95, -0.55, 0.45), (1.3, 0.2, 0.02))
    snap("floor_grazing", (1.9, -0.25, 0.05), (1.3, 0.2, 0.01))
    # close-up of the folded edge and the free edge on the desk, and a far view (head-camera distance, z-fighting)
    xe = float(n[:, 0].max())
    snap("desk_macro_fold", (xe + 0.16, -0.12, TOP + 0.035), (xe - 0.02, 0.0, TOP + 0.004))
    snap("desk_macro_edge", (0.05, -0.36, TOP + 0.03), (0.05, -0.18, TOP + 0.003))
    snap("far", (3.2, -2.4, 1.7), (0.6, 0.3, 0.3))
    (OUT / "result.json").write_text(json.dumps(res, indent=1))
    print(f"[thick_cloth] {TAG}: {json.dumps({k: v for k, v in res.items() if k != 'args'})}", flush=True)
    app.close()


if __name__ == "__main__":
    main()
