"""Match the lab duvet (2.0 x 1.55 m, ~5 mm) to the video ~/Downloads/bed.mp4 (2026-10-22).

    ~/miniforge3/envs/dx_arena/bin/python tools/cloth_spike/duvet_match.py --E 1e5 --tag e1e5

No robot. A bed box like the lab bed (mattress 0.98 x 1.88 m, top 0.53 m, sides straight to the floor, headboard)
and the duvet dropped flat from 3 cm above it, its head edge 0.25 m below the headboard (pillow zone) so 0.37 m hang
over the foot end and 0.285 m over each side - the end state of the video (24-25 s). Optionally (--flick) one short
edge is pinched, lifted 0.8 m and swung back down as in the video (9.6-12 s). Saves renders from the video camera
angle and drape metrics (side / foot hang depth and flare) to reports/duvet_match/<tag>/.
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--E", type=float, default=1.0e6, help="Young's modulus (Pa) at the FEM thickness")
ap.add_argument("--thickness", type=float, default=0.002, help="FEM surface thickness (m)")
ap.add_argument("--mass", type=float, default=1.2)
ap.add_argument("--mu", type=float, default=0.6)
ap.add_argument("--damping", type=float, default=0.0, help="linear damping (1/s), a crude air drag")
ap.add_argument("--hz", type=int, default=480)
ap.add_argument("--settle", type=float, default=5.0)
ap.add_argument("--spacing", type=float, default=0.023)
ap.add_argument("--grid", type=float, default=0.0, help="regular node grid spacing (thick_cloth.grid_rectangle_spawn)")
ap.add_argument("--dx_room", action="store_true", help="the whole DX care room (as the task loads it); duvet at the task pose")
ap.add_argument("--no_filter", action="store_true", help="InteractiveScene filter_collisions=False")
ap.add_argument("--bed_global", action="store_true", help="--dx_bed at /World/DXBed (outside the env), like the room's bed")
ap.add_argument("--empty_room", action="store_true", help="also reference the care room, Building and Furniture inactive")
ap.add_argument("--far", action="store_true", help="--dx_bed placed where the care-room bed is (y ~ 20 m from the origin)")
ap.add_argument("--move_dx_cloth", action="store_true", help="move the DX blanket / pillow / T-shirt 10 m away instead of deactivating")
ap.add_argument("--static_furniture", action="store_true", help="all DX furniture rigid bodies static, joints off")
ap.add_argument("--room_off", default="", help="comma list of room prim paths to deactivate (bisection)")
ap.add_argument("--no_cameras", action="store_true", help="no rendering experience, no camera (bisection)")
ap.add_argument("--dx_bed", action="store_true", help="the DX Bed.usda (blanket / pillow inactive) instead of the box bed")
ap.add_argument("--robot", action="store_true", help="add the RB-Y1 articulation 2 m away (render-bug bisection)")
ap.add_argument("--write_pillow", action="store_true", help="with --load and --pillow: also write the pillow state")
ap.add_argument("--task_layout", action="store_true", help="duvet 2.0 x 1.55 m spawned with yaw pi/2 like the task (vertex order)")
ap.add_argument("--no_replicate", action="store_true", help="InteractiveScene replicate_physics=False (as Arena)")
ap.add_argument("--task_caps", action="store_true", help="GPU buffer sizes of the task env (rby1_envs.physics_callback)")
ap.add_argument("--load", default=None, help="write these saved nodes (npy, env frame) at the start, like an episode")
ap.add_argument("--bind_step", action="store_true", help="step once before writing --load")
ap.add_argument("--footboard", action="store_true", help="a 2.5 cm thick, 0.58 m tall footboard like the DX bed")
ap.add_argument("--convex", action="store_true", help="bed collider as a convex-hull mesh (like the DX mattress)")
ap.add_argument("--pillow", action="store_true", help="add the volume-deformable pillow on the floor, 1.5 m away")
ap.add_argument("--fold", action="store_true", help="after settling, fold the head half onto the foot half (BED-01 easy)")
ap.add_argument("--fold_s", type=float, default=4.0)
ap.add_argument("--fold_z", type=float, default=0.35)
ap.add_argument("--tag", default=None)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
NO_CAM = "--no_cameras" in __import__("sys").argv
args.enable_cameras = not NO_CAM
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
from dxbench.assets.thick_cloth import thicken_render_mesh  # noqa: E402

TAG = args.tag or f"e{args.E:.0e}".replace("+0", "")
OUT = REPO / "reports" / "duvet_match" / TAG
OUT.mkdir(parents=True, exist_ok=True)
W, L, TOP = 0.98, 1.88, 0.53   # mattress
SIZE = (1.55, 2.0)             # duvet: x across the bed, y along it
if args.task_layout:
    SIZE = (2.0, 1.55)  # spawned with yaw pi / 2 below, like objects.duvet in the task
HALF = 0.0025
DT = 1.0 / args.hz


def T(x):
    if hasattr(x, "torch"):
        return x.torch
    return wp.to_torch(x) if isinstance(x, wp.array) else x


def _rby1():
    from dxbench.embodiments.rby1_cfg import make_rby1_cfg

    return make_rby1_cfg()


def build():
    density = args.mass / (SIZE[0] * SIZE[1] * args.thickness)
    duvet = sim_utils.MeshRectangleCfg(
        size=SIZE, edge_refinement=max(1.0, math.hypot(*SIZE) / (1.45 * args.spacing)),
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(solver_position_iteration_count=16, self_collision=True,
                                                               linear_damping=args.damping or None),
        physics_material=sim_utils.PhysxSurfaceDeformableBodyMaterialCfg(
            density=density, static_friction=args.mu, dynamic_friction=0.85 * args.mu, youngs_modulus=args.E,
            poissons_ratio=0.3, surface_thickness=args.thickness, bend_damping=0.1, elasticity_damping=0.1),
        collision_props=[PhysxCollisionCfg(contact_offset=0.008, rest_offset=HALF)],
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.85, 0.84), roughness=0.95))
    if args.grid:
        from dxbench.assets.thick_cloth import grid_rectangle_spawn

        duvet.edge_refinement = 1.0
        duvet.func = grid_rectangle_spawn(args.grid)
    white = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.74, 0.72))
    rmat = sim_utils.RigidBodyMaterialCfg(static_friction=args.mu, dynamic_friction=0.85 * args.mu)

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(color=(0.55, 0.55, 0.53)))
        light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=1500.0))
        sun = AssetBaseCfg(prim_path="/World/sun", spawn=sim_utils.DistantLightCfg(intensity=1800.0, angle=2.0),
                           init_state=AssetBaseCfg.InitialStateCfg(rot=(0.25, 0.25, 0.0, 0.93)))
        bed = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Bed",
                           spawn=(sim_utils.MeshCuboidCfg if args.convex else sim_utils.CuboidCfg)(
                               size=(W, L, TOP), collision_props=sim_utils.CollisionPropertiesCfg(),
                               visual_material=white, physics_material=rmat),
                           init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, TOP / 2)))
        head = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Head",
                            spawn=sim_utils.CuboidCfg(size=(W + 0.04, 0.04, 0.9), collision_props=sim_utils.CollisionPropertiesCfg(),
                                                      visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.6, 0.55))),
                            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, L / 2 + 0.02, 0.45)))
        if args.pillow:
            pil = DeformableObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pillow",
                spawn=sim_utils.MeshCuboidCfg(
                    size=(0.66, 0.42, 0.13), edge_refinement=4.0,
                    deformable_props=sim_utils.DeformableBodyPropertiesCfg(solver_position_iteration_count=16),
                    physics_material=sim_utils.DeformableBodyMaterialCfg(youngs_modulus=4000.0, poissons_ratio=0.3,
                                                                         density=35.0, static_friction=0.7,
                                                                         dynamic_friction=0.6),
                    collision_props=[PhysxCollisionCfg(contact_offset=0.01, rest_offset=0.002)]),
                init_state=DeformableObjectCfg.InitialStateCfg(pos=(1.6, 0.0, 0.08)))
        if args.robot:
            robot = _rby1().replace(prim_path="{ENV_REGEX_NS}/Robot")
            robot.init_state.pos = (-2.0, 0.0, robot.init_state.pos[2])
        duv = DeformableObjectCfg(prim_path="{ENV_REGEX_NS}/Duvet", spawn=duvet,
                                  init_state=DeformableObjectCfg.InitialStateCfg(
                                      pos=(1.981, 19.8885, 0.55) if (args.dx_room or args.far) else (0.0, L / 2 - 0.25 - SIZE[1] / 2, TOP + 0.03),
                                      rot=(0.0, 0.0, 0.7071068, 0.7071068) if args.task_layout else (0.0, 0.0, 0.0, 1.0)))
        if args.footboard:
            foot = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Foot",
                                spawn=sim_utils.CuboidCfg(size=(W + 0.05, 0.025, 0.58),
                                                          collision_props=sim_utils.CollisionPropertiesCfg(),
                                                          visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.6, 0.55))),
                                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -L / 2 - 0.0125, 0.29)))
        cam = None if NO_CAM else CameraCfg(prim_path="{ENV_REGEX_NS}/cam", update_period=0.0, width=1280, height=720, data_types=["rgb"],
                        spawn=sim_utils.PinholeCameraCfg(focal_length=14.0, clipping_range=(0.05, 10.0)),
                        renderer_cfg=IsaacRtxRendererCfg())

    scfg = sim_utils.SimulationCfg(device=args.device, dt=DT, render_interval=32)
    if args.task_caps:
        from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg

        scfg.physics = scfg.physics or PhysxCfg()
        for k, v in (("gpu_max_rigid_contact_count", 2**23), ("gpu_max_rigid_patch_count", 2**21),
                     ("gpu_max_soft_body_contacts", 2**21), ("gpu_heap_capacity", 2**27),
                     ("gpu_found_lost_pairs_capacity", 2**22)):
            setattr(scfg.physics, k, v)
    sim = sim_utils.SimulationContext(scfg)
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=5.0, replicate_physics=not args.no_replicate,
                                      filter_collisions=not args.no_filter))
    assert thicken_render_mesh(sim_utils.get_current_stage(), "/World/envs/env_0/Duvet", HALF)
    if args.empty_room:  # the room reference with everything in it switched off (render bisection)
        from pxr import UsdGeom

        st = sim_utils.get_current_stage()
        room = UsdGeom.Xform.Define(st, "/World/DX_Hub_Building_add_obj")
        room.GetPrim().GetReferences().AddReference(os.path.expanduser(
            "~/P2-DX_project/DX/USD/DX_Hub_Building_add_obj_v1/DX_Hub_Building_add_obj_v1_CareRoom.usda"))
        for q in ("Building", "Furniture"):
            st.GetPrimAtPath(f"/World/DX_Hub_Building_add_obj/{q}").SetActive(False)
    if args.dx_room:
        from pxr import UsdGeom, UsdPhysics

        st = sim_utils.get_current_stage()
        for q in ("/World/envs/env_0/Bed", "/World/envs/env_0/Head", "/World/envs/env_0/Foot"):
            if st.GetPrimAtPath(q).IsValid():
                st.GetPrimAtPath(q).SetActive(False)
        room = UsdGeom.Xform.Define(st, "/World/DX_Hub_Building_add_obj")
        room.GetPrim().GetReferences().AddReference(os.path.expanduser(
            "~/P2-DX_project/DX/USD/DX_Hub_Building_add_obj_v1/DX_Hub_Building_add_obj_v1_CareRoom.usda"))
        F = "/World/DX_Hub_Building_add_obj/Furniture/CareRoom"
        for rel in ("Bed/Blanket", "Bed/Pillow", "TShirt"):
            if args.move_dx_cloth:  # keep them, but 10 m away (stale-render test)
                xf = UsdGeom.Xformable(st.GetPrimAtPath(f"{F}/{rel}"))
                ops = [op for op in xf.GetOrderedXformOps() if op.GetOpName() == "xformOp:translate"]
                if ops:
                    v = ops[0].Get()
                    ops[0].Set(type(v)(v[0] + 10.0, v[1], v[2]))
                else:
                    xf.AddTranslateOp(opSuffix="dxmove").Set((10.0, 0.0, 0.0))
            else:
                st.GetPrimAtPath(f"{F}/{rel}").SetActive(False)
        for rel in ("Bed/Frame", "Bed/Mattress"):
            UsdPhysics.RigidBodyAPI(st.GetPrimAtPath(f"{F}/{rel}")).CreateRigidBodyEnabledAttr(False)
        UsdPhysics.Joint(st.GetPrimAtPath(f"{F}/Bed/Joints/mattress_to_frame")).CreateJointEnabledAttr(False)
        fl = st.GetPrimAtPath("/World/DX_Hub_Building_add_obj/Building/Collision/floor_325")
        if fl.IsValid():
            UsdPhysics.CollisionAPI.Apply(fl).CreateCollisionEnabledAttr(False)
        if args.static_furniture:
            from pxr import Usd

            nb = nj = 0
            for q in Usd.PrimRange(st.GetPrimAtPath(F)):
                if q.HasAPI(UsdPhysics.RigidBodyAPI):
                    UsdPhysics.RigidBodyAPI(q).CreateRigidBodyEnabledAttr(False)
                    nb += 1
                if q.IsA(UsdPhysics.Joint):
                    UsdPhysics.Joint(q).CreateJointEnabledAttr(False)
                    nj += 1
            print(f"[duvet_match] static furniture: {nb} bodies, {nj} joints", flush=True)
        for q in [x for x in args.room_off.split(",") if x]:
            if st.GetPrimAtPath(q).IsValid():
                st.GetPrimAtPath(q).SetActive(False)
    if args.dx_bed:
        from pxr import Gf, UsdGeom, UsdPhysics

        st = sim_utils.get_current_stage()
        st.GetPrimAtPath("/World/envs/env_0/Bed").SetActive(False)
        st.GetPrimAtPath("/World/envs/env_0/Head").SetActive(False)
        if st.GetPrimAtPath("/World/envs/env_0/Foot").IsValid():
            st.GetPrimAtPath("/World/envs/env_0/Foot").SetActive(False)
        BP = "/World/DXBed" if args.bed_global else "/World/envs/env_0/DXBed"
        b = UsdGeom.Xform.Define(st, BP)
        b.GetPrim().GetReferences().AddReference(
            os.path.expanduser("~/P2-DX_project/DX/USD/DX_Hub_Building_add_obj_v1/assets/Bed/Bed.usda"))
        # the DX bed's mattress top centre is at (0, -0.0775, 0.54) in its frame; put it where the box bed was
        b.AddTranslateOp().Set(Gf.Vec3d(1.981, 19.9685 + 0.0775, 0.0) if args.far else Gf.Vec3d(0.0, 0.0775, -0.01))
        for rel in ("Blanket", "Pillow"):
            q = st.GetPrimAtPath(f"{BP}/{rel}")
            if q.IsValid():
                q.SetActive(False)
        for rel in ("Frame", "Mattress"):
            q = st.GetPrimAtPath(f"{BP}/{rel}")
            if q.IsValid() and q.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(q).CreateRigidBodyEnabledAttr(False)
        j = st.GetPrimAtPath(f"{BP}/Joints/mattress_to_frame")
        if j.IsValid():
            UsdPhysics.Joint(j).CreateJointEnabledAttr(False)
    sim.reset()
    return sim, scene


def main():
    sim, scene = build()
    duv = scene["duv"]
    cam = None if NO_CAM else scene["cam"]
    o = scene.env_origins[0]

    def step(n=1, render=False):
        for i in range(n):
            sim.step(render=render and i == n - 1)
            scene.update(DT)

    def snap(name, eye, tgt):
        cam.set_world_poses_from_view((o + torch.tensor(eye, device=o.device, dtype=torch.float32))[None],
                                      (o + torch.tensor(tgt, device=o.device, dtype=torch.float32))[None])
        for _ in range(3):
            step(1, render=True)
        Image.fromarray(T(cam.data.output["rgb"])[0, ..., :3].cpu().numpy().astype(np.uint8)).save(OUT / f"{name}.png")

    if args.load:
        if args.bind_step:
            step(1)
        if args.pillow and args.write_pillow:  # like Arena's per-object reset: write the pillow's nodal state too
            pil = scene["pil"]
            ps = T(pil.data.default_nodal_state_w).clone()
            pil.write_nodal_state_to_sim_index(ps)
        saved = torch.as_tensor(np.load(args.load), dtype=torch.float32, device=o.device) + o
        pp = T(duv.data.nodal_pos_w).clone()
        pp[0] = saved
        duv.write_nodal_pos_to_sim_index(pp)
        duv.write_nodal_velocity_to_sim_index(torch.zeros_like(pp))
        step(6)
        snap("top_view", (0.01, 0.0, 3.0), (0.0, 0.0, 0.5))
        app.close()
        return
    step(int(args.settle / DT))
    if args.fold:
        n0 = T(duv.data.nodal_pos_w)[0].clone()
        ids = torch.nonzero(n0[:, 1] > n0[:, 1].max() - 0.02).flatten()
        p0 = n0[ids].clone()
        fold = 19.9685 if (args.dx_room or args.far) else float(o[1])  # mattress middle
        N = int(args.fold_s / DT)
        for i in range(N):
            s_ = min(1.0, (i + 1) / (0.85 * N))
            s_ = s_ * s_ * (3 - 2 * s_)
            r = p0[:, 1] - fold
            t = p0.clone()
            t[:, 1] = fold + r * math.cos(math.pi * s_)
            t[:, 2] = p0[:, 2] + args.fold_z * math.sin(math.pi * s_) + 0.02 * s_
            pos = T(duv.data.nodal_pos_w)
            vel = T(duv.data.nodal_vel_w).clone()
            vel[0, ids] = (t - pos[0, ids]) / DT
            duv.write_nodal_velocity_to_sim_index(vel)
            step(1)
        step(int(3.0 / DT))
    n = (T(duv.data.nodal_pos_w)[0] - o).cpu().numpy()
    side = n[np.abs(n[:, 0]) > W / 2 + 0.01]
    foot = n[n[:, 1] < -L / 2 - 0.01]
    res = {"args": vars(args),
           "side_drop": float(TOP - side[:, 2].min()) if len(side) else 0.0,       # how far the side hang reaches down
           "side_flare": float(np.abs(side[:, 0]).max() - W / 2) if len(side) else 0.0,  # outward from the side face
           "foot_drop": float(TOP - foot[:, 2].min()) if len(foot) else 0.0,
           "foot_flare": float(-L / 2 - foot[:, 1].min()) if len(foot) else 0.0,
           "top_rough_mm": float(np.std(n[(np.abs(n[:, 0]) < W / 2 - 0.05) & (np.abs(n[:, 1]) < L / 2 - 0.3), 2]) * 1000),
           "max_speed": float(T(duv.data.nodal_vel_w)[0].norm(dim=-1).max()),
           "fast_nodes": int((T(duv.data.nodal_vel_w)[0].norm(dim=-1) > 0.5).sum()),
           "extent": (n.max(0) - n.min(0)).round(3).tolist(), "y_range": [float(n[:, 1].min()), float(n[:, 1].max())]}
    # the video camera: standing at the bed's foot-left corner, ~1.6 m high, looking at the bed middle
    if NO_CAM:
        print(f"[duvet_match] {TAG}: {json.dumps({k: v for k, v in res.items() if k != 'args'})}", flush=True)
        app.close()
        return
    snap("video_view", (-1.3, -1.9, 1.55), (0.15, 0.1, 0.35))
    snap("side_view", (1.5, -0.3, 0.45), (0.0, -0.3, 0.3))
    if args.dx_room or args.far:  # the room ceiling is at ~2.7 m: look from 2.2 m above the bed middle
        snap("top_view", (1.99, 19.97, 2.2), (1.98, 19.97, 0.5))
        snap("far_side", (3.4, 19.3, 0.75), (1.98, 19.4, 0.5))
    else:
        snap("top_view", (0.01, 0.0, 3.0), (0.0, 0.0, 0.5))
    np.save(OUT / "nodes.npy", n)
    (OUT / "result.json").write_text(json.dumps(res, indent=1))
    print(f"[duvet_match] {TAG}: {json.dumps({k: v for k, v in res.items() if k != 'args'})}", flush=True)
    app.close()


if __name__ == "__main__":
    main()
