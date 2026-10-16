"""T4: place RB-Y1 in the DX care room (bedroom) and run 10 s on GPU physics with the deformables.

    python tools/rby1_check/t4_bedroom.py [--yaw 0] [--duration 10] [--viz kit]

Order matters: open the scene stage first, then create the Isaac Lab SimulationContext on cuda:0
(that is where the physics device gets set; doing it before a stage exists crashes Kit).
The scene's own /World/PhysicsScene (GPU dynamics, TGS) is reused instead of adding a second one.
Outputs reports/rby1_check/t4/{t4_<load>.json, head_rgb_<load>.png, head_depth_<load>.png, overview_<load>.png}.
Isaac Lab 3.0 quaternions are (x, y, z, w).
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--yaw", type=float, default=0.0, help="robot heading in degrees (0 = +x)")
parser.add_argument("--duration", type=float, default=10.0)
parser.add_argument("--dt", type=int, default=200, help="physics Hz")
parser.add_argument("--mode", default="implicit", choices=["implicit", "ideal_pd"])
parser.add_argument("--load", default="open", choices=["open", "reference"],
                    help="open = open the *_scene.usda stage, then add the robot (physics OK, but cloned envs are not "
                         "rendered, see REPORT); reference = Isaac Lab stage + CareRoom.usda as a global asset (Arena style)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

from common import CARE_SCENE, CARE_SPAWN, kit_log_path, out_dir, save_json  # noqa: E402

OUT = out_dir("t4")
FU = "/World/DX_Hub_Building_add_obj/Furniture/CareRoom"
DEFORM = {"pillow": f"{FU}/Bed/Pillow", "blanket": f"{FU}/Bed/Blanket", "tshirt": f"{FU}/TShirt/Shirt"}
LOG0 = kit_log_path()
LOG0_SIZE = LOG0.stat().st_size if LOG0 else 0


def T(x):
    import warp as wp

    if hasattr(x, "torch"):
        return x.torch
    if isinstance(x, wp.array):
        return wp.to_torch(x)
    return x


def world_bounds(stage, path, purposes=("default", "render")):
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), list(purposes), useExtentsHint=False)
    r = cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
    return np.array(r.GetMin()), np.array(r.GetMax())


def box_gap(a, b):
    """Signed separation of two AABBs (negative = overlap depth along the least-overlapping axis)."""
    (amin, amax), (bmin, bmax) = a, b
    sep = np.maximum(bmin - amax, amin - bmax)  # per axis
    return float(sep.max())


def qmul(a, b):
    """Quaternion product, (x, y, z, w)."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def look_rot(eye, target):
    """(x, y, z, w) for an Isaac Lab camera in 'world' convention (+x forward, +z up)."""
    d = np.asarray(target) - np.asarray(eye)
    yw, pt = math.atan2(d[1], d[0]), math.atan2(-d[2], math.hypot(d[0], d[1]))
    qy = np.array([0, 0, math.sin(yw / 2), math.cos(yw / 2)])
    qp = np.array([0, math.sin(pt / 2), 0, math.cos(pt / 2)])
    return tuple(float(v) for v in qmul(qy, qp))


def main():
    t0 = time.time()
    ctx = omni.usd.get_context()
    scene_ps = []
    if args.load == "open":
        ctx.open_stage(str(CARE_SCENE))
        scene_ps = [str(p.GetPath()) for p in ctx.get_stage().Traverse() if p.IsA(UsdPhysics.Scene)]

    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sensors import CameraCfg
    from isaaclab.utils import configclass

    from rby1_cfg_draft import apply_original_materials, make_rby1_cfg

    yaw = math.radians(args.yaw)
    rcfg = make_rby1_cfg(mode=args.mode)
    rcfg.init_state.pos = CARE_SPAWN
    rcfg.init_state.rot = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))  # (x, y, z, w)

    # overview camera inside the room, south-east of the robot (the west wall is at x = -4.2). It must live under
    # the env namespace: Isaac Lab 3.0 renders env prims only for cameras of the same env (global cameras miss them)
    eye = (-0.4, 16.9, 2.1)
    target = tuple(float(v) for v in np.array(CARE_SPAWN) + np.array([0.0, 0.2, 0.7]))

    from isaaclab.assets import AssetBaseCfg

    from common import DX_DIR

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        robot = rcfg
        overview_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/T4_overview", width=1280, height=720, data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=12.0, horizontal_aperture=20.955, clipping_range=(0.03, 100.0)),
            offset=CameraCfg.OffsetCfg(pos=eye, rot=look_rot(eye, target), convention="world"))
        head_cam = CameraCfg(prim_path="{ENV_REGEX_NS}/Robot/link_head_2/first_person_camera", spawn=None,
                             width=640, height=480, data_types=["rgb", "distance_to_image_plane"],
                             update_latest_camera_pose=True)

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        device="cuda:0", dt=1.0 / args.dt, render_interval=max(1, args.dt // 30),
        physics_prim_path=scene_ps[0] if scene_ps else "/physicsScene"))
    if args.load == "reference":  # same prim paths as in the *_scene.usda stage
        SceneCfg.dx = AssetBaseCfg(prim_path="/World/DX_Hub_Building_add_obj", spawn=sim_utils.UsdFileCfg(
            usd_path=str(DX_DIR / "DX_Hub_Building_add_obj_v1_CareRoom.usda")))
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=0.0))
    apply_original_materials()
    stage = ctx.get_stage()
    for _ in range(20):
        app.update()

    # ---- static clearance (zero pose, before stepping): robot link AABBs vs furniture AABBs
    robot_root = "/World/envs/env_0/Robot"
    furn = [p for p in stage.GetPrimAtPath(FU).GetChildren()]
    furn_b = {p.GetName(): world_bounds(stage, str(p.GetPath())) for p in furn}
    proxies = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)  # colliders sit inside instanceable prims
    links = [p for p in Usd.PrimRange(stage.GetPrimAtPath(robot_root), proxies) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    has_col = {str(p.GetPath()): any(q.HasAPI(UsdPhysics.CollisionAPI) for q in Usd.PrimRange(p, proxies)) for p in links}
    clear = []
    for p in links:
        lb = world_bounds(stage, str(p.GetPath()))
        for fn, fb in furn_b.items():
            clear.append({"link": p.GetName(), "furniture": fn, "gap_m": box_gap(lb, fb),
                          "link_has_collider": has_col[str(p.GetPath())]})
    nearest = sorted(clear, key=lambda c: c["gap_m"])[:8]
    robot_b = world_bounds(stage, robot_root)
    robot_vis = UsdGeom.Imageable(stage.GetPrimAtPath(robot_root)).ComputeVisibility()

    sim.reset()
    from isaaclab_physx.physics.physx_manager import PhysxManager

    phys_dev = PhysxManager.get_physics_sim_device()
    robot = scene["robot"]
    names = list(robot.joint_names)
    bodies = list(robot.body_names)
    q_tgt = T(robot.data.default_joint_pos).clone()
    q_tgt[:, names.index("head_1")] = math.radians(15)  # look slightly down at the bed
    base_i = bodies.index("base")

    def deform_state():
        """Nodal positions through the PhysX tensor views (same API Isaac Lab's DeformableObject uses)."""
        import warp as wp

        out = {}
        view = PhysxManager.get_physics_sim_view()
        for k, p in DEFORM.items():
            prim = stage.GetPrimAtPath(p)
            if not prim.IsValid():
                out[k] = "missing"
                continue
            has_tet = any(q.GetTypeName() == "TetMesh" for q in Usd.PrimRange(prim))
            try:
                v = (view.create_volume_deformable_body_view(p) if has_tet else view.create_surface_deformable_body_view(p))
                pos = v.get_simulation_nodal_positions()
                pos = (wp.to_torch(pos) if isinstance(pos, wp.array) else torch.as_tensor(pos)).reshape(-1, 3).cpu().numpy()
                out[k] = {"type": "volume" if has_tet else "surface", "nodes": int(len(pos)),
                          "finite": bool(np.isfinite(pos).all()), "z_min": float(pos[:, 2].min()),
                          "z_max": float(pos[:, 2].max()), "centroid": pos.mean(axis=0).tolist()}
            except Exception as e:  # noqa: BLE001
                out[k] = {"error": repr(e)}
        return out

    def_before = deform_state()
    p_base0 = (T(robot.data.body_link_pos_w)[0, base_i]).clone()
    steps = int(args.duration * args.dt)
    rec_err, rec_disp, rec_tilt = [], [], []
    body_ids = [names.index(n) for n in names if not n.endswith("wheel") and "gripper" not in n]
    for k in range(steps):
        robot.set_joint_position_target_index(target=q_tgt)
        scene.write_data_to_sim()
        sim.step(render=(k % max(1, args.dt // 30) == 0))
        scene.update(1.0 / args.dt)
        if k % max(1, args.dt // 50) == 0:
            q = T(robot.data.joint_pos)
            rec_err.append(float((q[0, body_ids] - q_tgt[0, body_ids]).abs().max()))
            rec_disp.append(float((T(robot.data.body_link_pos_w)[0, base_i] - p_base0).norm()))
            qb = T(robot.data.body_link_quat_w)[0, base_i]  # (x, y, z, w)
            rec_tilt.append(math.degrees(math.acos(max(-1.0, min(1.0, float(1 - 2 * (qb[0] ** 2 + qb[1] ** 2)))))))
    def_after = deform_state()
    for _ in range(4):  # settle renderer
        sim.render()
        scene.update(0.0)

    # ---- head camera image + bed visibility (project bed AABB corners)
    cam = scene["head_cam"]
    rgb = T(cam.data.output["rgb"])[0].cpu().numpy()[..., :3].astype(np.uint8)
    dep = T(cam.data.output["distance_to_image_plane"])[0].cpu().numpy().squeeze(-1)
    K = T(cam.data.intrinsic_matrices)[0].cpu().numpy()
    pos = T(cam.data.pos_w)[0].cpu().numpy()
    x, y, z, w = T(cam.data.quat_w_ros)[0].cpu().numpy()  # (x, y, z, w); ROS: +z forward, +x right, +y down
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    bmin, bmax = furn_b["Bed"]
    corners = np.array([[a, b, c] for a in (bmin[0], bmax[0]) for b in (bmin[1], bmax[1]) for c in (bmin[2], bmax[2])])
    pc = (corners - pos) @ R  # world -> camera (R columns = camera axes in world)
    in_front = pc[:, 2] > 0.05
    uv = (K @ pc.T).T
    uv = uv[:, :2] / np.maximum(uv[:, 2:3], 1e-6)
    inside = in_front & (uv[:, 0] >= 0) & (uv[:, 0] < 640) & (uv[:, 1] >= 0) & (uv[:, 1] < 480)
    centre = ((bmin + bmax) / 2 - pos) @ R
    cuv = (K @ centre)[:2] / max(centre[2], 1e-6)
    bed_visible = bool(centre[2] > 0 and 0 <= cuv[0] < 640 and 0 <= cuv[1] < 480)
    img = Image.fromarray(rgb)
    d = ImageDraw.Draw(img)
    if in_front.any():
        u = np.clip(uv[in_front], -1e4, 1e4)
        d.rectangle([float(u[:, 0].min()), float(u[:, 1].min()), float(u[:, 0].max()), float(u[:, 1].max())],
                    outline=(255, 140, 0), width=3)
        d.text((8, 8), "Bed AABB (projected)", fill=(255, 140, 0))
    img.save(OUT / f"head_rgb_{args.load}.png")
    Image.fromarray(rgb).save(OUT / f"head_rgb_raw_{args.load}.png")
    fin = np.isfinite(dep) & (dep > 0)
    dv = np.zeros(dep.shape, np.uint8)
    if fin.any():
        dd = np.clip(dep, 0, np.percentile(dep[fin], 99))
        dv = (255 * (1 - dd / max(1e-6, dd.max()))).astype(np.uint8)
        dv[~fin] = 0
    Image.fromarray(dv).save(OUT / f"head_depth_{args.load}.png")

    # ---- overview screenshot (Isaac Lab camera spawned at a fixed world pose)
    try:
        ov = T(scene["overview_cam"].data.output["rgb"])[0].cpu().numpy()[..., :3].astype(np.uint8)
        Image.fromarray(ov).save(OUT / f"overview_{args.load}.png")
        overview = str(OUT / f"overview_{args.load}.png")
    except Exception as e:  # noqa: BLE001
        overview = f"failed: {e!r}"

    log = kit_log_path()
    warns = []
    if log and log.exists():
        with open(log, errors="replace") as fh:
            if log == LOG0:
                fh.seek(LOG0_SIZE)
            warns = [ln.strip() for ln in fh if "[Warning]" in ln or "[Error]" in ln]
    deform_ok = all(isinstance(def_after.get(k), dict) and def_after[k].get("finite") for k in DEFORM)
    res = {
        "scene": str(CARE_SCENE), "load": args.load, "spawn": CARE_SPAWN, "yaw_deg": args.yaw, "physics_device": phys_dev,
        "physics_scene_reused": scene_ps, "dt": 1.0 / args.dt, "duration_s": args.duration,
        "robot_aabb": [robot_b[0].tolist(), robot_b[1].tolist()], "robot_visibility": str(robot_vis),
        "static_nearest_furniture": nearest,
        "static_overlaps": [c for c in clear if c["gap_m"] < 0],
        "links_without_collider": [p.GetName() for p in links if not has_col[str(p.GetPath())]],
        "base_disp_max_m": max(rec_disp), "base_tilt_max_deg": max(rec_tilt),
        "joint_err_max_rad": max(rec_err), "joint_err_final_rad": rec_err[-1],
        "deformables_before": def_before, "deformables_after": def_after,
        "head_camera": {"pos_w": pos.tolist(), "bed_centre_visible": bed_visible, "bed_corners_in_image": int(inside.sum()),
                        "bed_centre_px": cuv.tolist(), "rgb_mean": float(rgb.mean())},
        "images": {"head_rgb": str(OUT / f"head_rgb_{args.load}.png"), "head_depth": str(OUT / f"head_depth_{args.load}.png"),
                   "overview": overview},
        "kit_warning_count": len(warns), "kit_warnings": warns[:300], "wall_s": time.time() - t0,
    }
    crit = {"gpu_physics": "cuda" in str(phys_dev),
            "no_static_overlap(collider links)": not any(c["link_has_collider"] for c in res["static_overlaps"]),
            "robot_stable(base<1cm, tilt<2deg)": res["base_disp_max_m"] < 0.01 and res["base_tilt_max_deg"] < 2.0,
            "deformables_finite": deform_ok, "bed_in_head_camera": bed_visible}
    res["criteria"] = crit
    res["pass"] = all(crit.values())
    save_json(OUT / f"t4_{args.load}.json", res)
    print(f"[T4] pass={res['pass']} criteria={crit}")
    print(json.dumps({k: res[k] for k in ("base_disp_max_m", "base_tilt_max_deg", "joint_err_max_rad", "head_camera",
                                         "robot_aabb", "robot_visibility")}, indent=1))


if __name__ == "__main__":
    main()
    app.close()
