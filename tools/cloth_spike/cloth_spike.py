"""Stage 1 cloth spike: can the RB-Y1 gripper handle the duvet, a T-shirt sleeve and a floor shirt?

    python tools/cloth_spike/cloth_spike.py --test duvet  --grip friction --dt 240
    python tools/cloth_spike/cloth_spike.py --test sleeve --grip assist   --dt 480 --cameras
    python tools/cloth_spike/cloth_spike.py --test floor  --grip assist   --cloth_iters 32 --friction 1.0

Tests (robot base fixed at the origin facing +x, preset "tuned" gains, gravity feedforward):
  duvet  : 1.5 x 1.4 m surface cloth (DX BlanketPhysics material) on a 1.0 x 2.0 m mattress, near edge
           hanging; grasp the flap 25 cm from the corner and pull it 0.5 m (0.3 back, 0.4 up).
  sleeve : DX TShirt.usda on a desk (top 0.725 m), grasp the right sleeve end and fold it over the body.
  floor  : DX TShirt.usda on the floor in front of the robot, grasp near the shoulder and lift 0.3 m.
Grasp modes:
  friction : fingers only.
  assist   : once the fingers are closed, nodes within --assist_r of the TCP are pinned to the TCP
             (their positions/velocities are overwritten every physics step; surface deformables
             have no kinematic targets in PhysX).
After the manipulation the scene keeps running until --duration (default 60 s) to check stability.
Writes reports/cloth_spike/runs/<run>.json (+ snapshots with --cameras).
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
sys.path.insert(0, os.path.expanduser("~/dx_bench/tools/rby1_check"))

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--test", required=True, choices=["duvet", "sleeve", "floor"])
ap.add_argument("--grip", default="friction", choices=["friction", "assist", "pad"],
                help="pad: the BED-task pad-box grasp (D-34/35): fingertips onto the cloth, only cloth between the "
                     "pads (+-2.5 cm, <= 4 mm beyond the tips) captured when closing, pulled into the pads")
ap.add_argument("--dt", type=int, default=240, help="physics Hz")
ap.add_argument("--duration", type=float, default=60.0)
ap.add_argument("--cameras", action="store_true", help="render the 3 robot cameras at 15 Hz (+ observer snapshots)")
ap.add_argument("--self_collision", type=int, default=1)
ap.add_argument("--res_cm", type=float, default=2.3, help="duvet node spacing (cm): 4.7 | 2.3 | 1.2")
ap.add_argument("--torso", default=None, help="override torso preset: upright|lean|crouch|deep")
ap.add_argument("--thickness", type=float, default=None, help="cloth surface thickness (m), default = asset value")
ap.add_argument("--friction", type=float, default=None, help="cloth static friction (dynamic = 0.85x), default = asset")
ap.add_argument("--cloth_iters", type=int, default=16, help="deformable solver position iterations")
ap.add_argument("--assist_r", type=float, default=0.05)
ap.add_argument("--pin_mode", default="velocity", choices=["position", "velocity"],
                help="velocity (default, stable) | position (blows up large moving cloth, kept for the record)")
ap.add_argument("--assist_filter", type=int, default=1, help="assist: filter right-finger/cloth collisions")
ap.add_argument("--pin_gain", type=float, default=1.0, help="assist: fraction of the position error removed per step")
ap.add_argument("--pad_capture_c", type=float, default=0.025, help="pad: capture half width along the closing axis")
ap.add_argument("--pad_support_r", type=float, default=0.0,
                help="pad: also hold nodes within this radius of the pad centre, in the pad height band (load sharing)")
ap.add_argument("--pad_support_gain", type=float, default=0.5)
ap.add_argument("--tag", default="")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
if args.cameras:
    args.enable_cameras = True
app = AppLauncher(args).app
# TCP height above the cloth top at the grasp: fingertips (TCP - 3.1 cm) 2 mm above it, or 2 mm into it for pad mode
GRIP_DZ = 0.031 - 0.002 if args.grip == "pad" else 0.033

import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from pxr import Sdf  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg, DeformableObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from isaaclab_physx.sim.schemas.schemas_cfg import PhysxCollisionCfg  # noqa: E402

from rby1_cfg_draft import apply_original_materials, make_rby1_cfg  # noqa: E402

DX_ASSETS = Path(os.path.expanduser("~/P2-DX_project/DX/USD/DX_Hub_Building_add_obj_v1/assets"))
TSHIRT = DX_ASSETS / "TShirt" / "TShirt.usda"
OUT = Path(os.path.expanduser("~/dx_bench/reports/cloth_spike"))
(OUT / "runs").mkdir(parents=True, exist_ok=True)
(OUT / "images").mkdir(parents=True, exist_ok=True)
DEV = args.device
DT = 1.0 / args.dt
CTRL_HZ = 15
RUN = (f"{args.test}_{args.grip}_dt{args.dt}_sc{args.self_collision}_it{args.cloth_iters}"
       + (f"_res{args.res_cm:g}" if args.test == "duvet" else "")
       + (f"_th{args.thickness:g}" if args.thickness else "") + (f"_mu{args.friction:g}" if args.friction else "")
       + ("_cam" if args.cameras else "") + (f"_{args.tag}" if args.tag else ""))

# torso presets (rad) — torso_1/2/3 combined (0-stage: never bend torso_1 alone)
TORSO = {"upright": (0.0, 0.0, 0.0), "lean": (math.radians(45), math.radians(-70), math.radians(45)),
         "crouch": (math.radians(60), math.radians(-90), math.radians(60)),
         "deep": (math.radians(90), math.radians(-130), math.radians(90))}
TCP_OFF = (0.0, 0.0, -0.1045)  # in ee frame (0-stage report)


def T(x):
    if hasattr(x, "torch"):
        return x.torch
    if isinstance(x, wp.array):
        return wp.to_torch(x)
    return x


def wxyz(q):
    return q[..., [3, 0, 1, 2]]


def quat_apply(q, v):  # q wxyz (N,4), v (N,3)
    w, xyz = q[:, :1], q[:, 1:]
    t = 2 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def quat_to_mat(q):  # wxyz torch (4,)
    w, x, y, z = q.tolist()
    return torch.tensor([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], device=q.device)


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], device=DEV)


# ------------------------------------------------------------------------------------------ test layouts
def layout():
    """Supports, cloth spawn and the TCP waypoints for each test (world frame, robot at origin)."""
    if args.test == "duvet":
        top = 0.53
        mat = dict(size=(1.0, 2.0, top), pos=(0.95, 0.0, top / 2))  # near edge at x = 0.45
        # duvet lying flat with its near-right corner 5 cm inside the mattress edge (x 0.50, y -0.45);
        # the far side drapes over the mattress. Grasp = pinch the corner from above, then pull it 0.5 m
        # (0.3 m back towards the robot + 0.4 m up).
        cloth = dict(kind="rect", size=(1.5, 1.4), pos=(1.25, 0.25, top + 0.01))
        return dict(supports=[mat], cloth=cloth, torso="crouch", R=rot_z(0.0), corner=True, hold_close=1.0)
    if args.test == "sleeve":
        top = 0.725
        # desk near edge 0.45 m in front of the base centre (the leaning torso must clear it)
        desk = dict(size=(0.6, 1.2, top), pos=(0.75, 0.0, top / 2))
        cloth = dict(kind="tshirt", pos=(0.70, 0.0, top + 0.004), yaw=90.0)  # sleeves along world y
        return dict(supports=[desk], cloth=cloth, torso="lean", R=rot_z(math.radians(90)), sleeve=True,
                    hold_close=1.0)
    # floor
    # floor shirt: near hem ~0.2 m in front of the base front face (front ~0.32 m), i.e. inside the
    # 15-50 cm floor-pick band of the 0-stage URDF analysis
    cloth = dict(kind="tshirt", pos=(0.85, -0.15, 0.004), yaw=90.0)
    return dict(supports=[], cloth=cloth, torso="deep", R=rot_z(math.radians(90)), floor=True, hold_close=1.0)


LAY = layout()
if args.torso:
    LAY["torso"] = args.torso


def build():
    robot_cfg = make_rby1_cfg(preset="tuned")
    robot_cfg.spawn.articulation_props.fix_root_link = True
    t1, t2, t3 = TORSO[LAY["torso"]]
    robot_cfg.init_state.joint_pos = {".*_wheel": 0.0, "torso_[045]": 0.0, "torso_1": t1, "torso_2": t2,
                                      "torso_3": t3, "left_arm_.*": 0.0, "right_arm_[246]": 0.0,
                                      # IK seed (0-stage T3d): elbow bent, forearm forward
                                      "right_arm_0": -0.5, "right_arm_1": -0.2, "right_arm_3": -1.6,
                                      "right_arm_5": 0.3, "head_.*": 0.0,
                                      "gripper_finger_[lr]1": -0.05, "gripper_finger_[lr]2": 0.05}

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
        light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2500.0))
        robot = robot_cfg

    for i, s in enumerate(LAY["supports"]):
        setattr(SceneCfg, f"support{i}", AssetBaseCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Support{i}",
            spawn=sim_utils.CuboidCfg(size=s["size"], collision_props=sim_utils.CollisionPropertiesCfg(),
                                      physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.6,
                                                                                      dynamic_friction=0.5),
                                      visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.85, 0.8))),
            init_state=AssetBaseCfg.InitialStateCfg(pos=s["pos"])))
    c = LAY["cloth"]
    yaw = math.radians(c.get("yaw", 0.0))
    rot = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
    deform_props = sim_utils.DeformableBodyPropertiesCfg(
        solver_position_iteration_count=args.cloth_iters, self_collision=bool(args.self_collision))
    if c["kind"] == "rect":
        diag = math.hypot(*c["size"])
        mu = args.friction or 0.6
        # trimesh subdivide_to_size halves edges until the (diagonal) edges are <= max_edge, so the node
        # spacing is size / 2^k: 1.5 m -> 4.7 / 2.3 / 1.2 cm. max_edge = 1.45 x the requested spacing.
        spawn = sim_utils.MeshRectangleCfg(
            size=c["size"], edge_refinement=diag / (1.45 * args.res_cm / 100.0), deformable_props=deform_props,
            physics_material=sim_utils.PhysxSurfaceDeformableBodyMaterialCfg(
                density=100.0, static_friction=mu, dynamic_friction=0.85 * mu, youngs_modulus=1.0e6,
                poissons_ratio=0.3, surface_thickness=args.thickness or 0.002, bend_damping=0.1,
                elasticity_damping=0.1),
            collision_props=[PhysxCollisionCfg(contact_offset=0.012, rest_offset=0.005)],
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.7, 0.9)))
    else:
        # no deformable_props: the asset's Shirt mesh already carries the deformable body API (adding
        # props here would create a second simulation mesh at the root); iterations/self-collision are
        # authored on the Shirt prim below
        spawn = sim_utils.UsdFileCfg(usd_path=str(TSHIRT))
    SceneCfg.cloth = DeformableObjectCfg(prim_path="{ENV_REGEX_NS}/Cloth", spawn=spawn,
                                         init_state=DeformableObjectCfg.InitialStateCfg(pos=c["pos"], rot=rot))
    if args.cameras:
        from isaaclab.sensors import CameraCfg

        for k, rel in {"head": "link_head_2/first_person_camera", "left": "left_gripper/ee_left/left_hand_camera",
                       "right": "right_gripper/ee_right/right_hand_camera"}.items():
            setattr(SceneCfg, f"cam_{k}", CameraCfg(prim_path=f"{{ENV_REGEX_NS}}/Robot/{rel}", spawn=None,
                                                    width=640, height=480, data_types=["rgb"],
                                                    update_period=1.0 / CTRL_HZ, update_latest_camera_pose=True))
        eye, tgt = (1.9, -1.6, 1.5), (0.5, -0.2, 0.5)
        d = [t - e for t, e in zip(tgt, eye)]
        yw, pt = math.atan2(d[1], d[0]), math.atan2(-d[2], math.hypot(d[0], d[1]))
        cy, sy, cp, sp = math.cos(yw / 2), math.sin(yw / 2), math.cos(pt / 2), math.sin(pt / 2)
        SceneCfg.cam_obs = CameraCfg(prim_path="{ENV_REGEX_NS}/Observer", width=960, height=720, data_types=["rgb"],
                                     update_period=1.0 / CTRL_HZ,
                                     spawn=sim_utils.PinholeCameraCfg(focal_length=14.0, horizontal_aperture=20.955),
                                     offset=CameraCfg.OffsetCfg(pos=eye, rot=(-sy * sp, cy * sp, sy * cp, cy * cp),
                                                                convention="world"))
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        device=DEV, dt=DT, render_interval=max(1, args.dt // CTRL_HZ)))
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=4.0))
    apply_original_materials(caster_mu=0.0)
    if c["kind"] == "tshirt" and (args.thickness or args.friction):
        stage = sim_utils.get_current_stage()
        for p in stage.Traverse():
            if p.GetName() == "TShirtPhysics":
                if args.thickness:
                    p.GetAttribute("omniphysics:surfaceThickness").Set(float(args.thickness))
                if args.friction:
                    p.GetAttribute("omniphysics:staticFriction").Set(float(args.friction))
                    p.GetAttribute("omniphysics:dynamicFriction").Set(float(0.85 * args.friction))
    if args.grip == "assist" and args.assist_filter:
        # assist pins move nodes rigidly with the TCP; finger contacts would fight them, so the right
        # fingers do not collide with the cloth in assist mode (a sim-only crutch, see DECISIONS)
        from pxr import UsdPhysics

        stage = sim_utils.get_current_stage()
        for f in ("ee_finger_r1", "ee_finger_r2"):
            fp = stage.GetPrimAtPath(f"/World/envs/env_0/Robot/right_gripper/{f}")
            UsdPhysics.FilteredPairsAPI.Apply(fp).CreateFilteredPairsRel().AddTarget("/World/envs/env_0/Cloth")
    if c["kind"] == "tshirt":
        stage = sim_utils.get_current_stage()
        for p in stage.Traverse():
            if "OmniPhysicsDeformableBodyAPI" in p.GetAppliedSchemas():
                p.CreateAttribute("physxDeformableBody:selfCollision", Sdf.ValueTypeNames.Bool).Set(bool(args.self_collision))
                p.CreateAttribute("physxDeformableBody:solverPositionIterationCount",
                                  Sdf.ValueTypeNames.Int).Set(int(args.cloth_iters))
    sim.reset()
    return sim, scene


# ------------------------------------------------------------------------------------------ robot helpers
class Bot:
    def __init__(self, sim, scene):
        self.sim, self.scene = sim, scene
        self.r = scene["robot"]
        self.cloth = scene["cloth"]
        self.names = list(self.r.joint_names)
        self.bodies = list(self.r.body_names)
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.q_tgt = T(self.r.data.default_joint_pos).clone()
        body = [n for n in self.names if not n.endswith("wheel") and "gripper" not in n]
        self.body_ids = [self.idx[n] for n in body]
        self.arm = [self.idx[f"right_arm_{i}"] for i in range(7)]
        self.f = {n: self.idx[f"gripper_finger_{n}"] for n in ("r1", "r2", "l1", "l2")}
        self.t = 0.0
        self.pins = None  # (node ids, offsets in TCP frame)
        self.pad = None   # pad mode: {"ids", "loc0" (grip frame w, c, a), "t", "prev", "out_max", ...}
        self.tcp_off = torch.tensor([TCP_OFF], device=DEV)
        self.steps = 0
        self.wall = 0.0

    def ee(self):
        bi = self.bodies.index("ee_right")
        p = T(self.r.data.body_link_pos_w)[:, bi]
        q = wxyz(T(self.r.data.body_link_quat_w)[:, bi])
        return p, q

    def tcp(self):
        p, q = self.ee()
        return p + quat_apply(q, self.tcp_off), q

    def nodes(self):
        return T(self.cloth.data.nodal_pos_w)[0]

    def grip(self, open_m):
        self.q_tgt[:, self.f["r1"]] = -open_m
        self.q_tgt[:, self.f["r2"]] = open_m
        self.q_tgt[:, self.f["l1"]] = -0.05
        self.q_tgt[:, self.f["l2"]] = 0.05

    def step(self, n=1):
        for _ in range(n):
            t0 = time.perf_counter()
            self.r.set_joint_position_target_index(target=self.q_tgt)
            g = T(self.r.data.gravity_compensation_forces)
            ff = torch.zeros_like(self.q_tgt)
            ff[:, self.body_ids] = g[:, g.shape[-1] - len(self.names):][:, self.body_ids]
            self.r.set_joint_effort_target_index(target=ff)
            if self.pins is not None:
                self._apply_pins()
            if self.pad is not None:
                self._apply_pad()
            self.scene.write_data_to_sim()
            self.sim.step(render=args.cameras and self.steps % max(1, args.dt // CTRL_HZ) == 0)
            self.scene.update(DT)
            self.t += DT
            self.steps += 1
            self.wall += time.perf_counter() - t0

    def _apply_pins(self):
        """Pin = consistent position + velocity writes for the grasped node cluster.

        Targets start at the nodes' own positions (no jump) and follow the TCP rigidly; velocities are
        the TCP-induced target velocity. Writing a single node to a distant position with zero velocity
        blows the cloth up, and velocity-only pins lag by ~12 cm under the duvet's weight (REPORT)."""
        ids, off = self.pins
        p_tcp, q = self.tcp()
        R = quat_to_mat(q[0])
        tgt = p_tcp[0] + off @ R.T
        v = (tgt - self._pin_prev) / DT if self._pin_prev is not None else torch.zeros_like(tgt)
        self._pin_prev = tgt.clone()
        pos = T(self.cloth.data.nodal_pos_w).clone()
        vel = T(self.cloth.data.nodal_vel_w).clone()
        if args.pin_mode == "velocity":
            vel[0, ids] = args.pin_gain * (tgt - pos[0, ids]) / DT + v
        else:
            if os.environ.get("DEBUG_PINS") and self.steps % 60 == 0:
                print(f"[pin] step {self.steps} tgt0 {tgt[0].tolist()} node0 {pos[0, ids[0]].tolist()} n={len(ids)}")
            pos[0, ids] = tgt
            vel[0, ids] = v
            self.cloth.write_nodal_pos_to_sim_index(pos)
        self.cloth.write_nodal_velocity_to_sim_index(vel)

    # ---- pad mode (same rules as dxbench/embodiments/mdp/pad_grasp.py, right hand)
    PAD_W, PAD_H, TIP_MARGIN, CAPTURE_C, INSET, SNAP_S = 0.032, 0.02, 0.004, 0.025, 0.002, 0.2

    def grip_frame(self):
        p, q = self.ee()
        R = quat_to_mat(q[0])
        o = p[0] + R @ torch.tensor([0.0, 0.0, -0.1355], device=DEV)
        G = torch.stack([R[:, 1], R[:, 0], -R[:, 2]], dim=1)  # columns w, c, a
        return o, G

    def pad_capture(self):
        o, G = self.grip_frame()
        q = T(self.r.data.joint_pos)[0]
        half = float((q[self.f["r1"]].abs() + q[self.f["r2"]].abs()) / 2)
        loc = (self.nodes() - o) @ G
        fm = (loc[:, 0].abs() <= self.PAD_W / 2 + 0.002) & (loc[:, 1].abs() <= min(half + 0.002, args.pad_capture_c))
        m = fm & (loc[:, 2] >= -self.PAD_H) & (loc[:, 2] <= self.TIP_MARGIN)
        under = fm & (loc[:, 2] >= -self.PAD_H)
        ids = torch.nonzero(m).flatten()
        info = {"half_gap": half, "tip_to_cloth": float(loc[under, 2].min()) if under.any() else None,
                "nodes": int(len(ids))}
        if len(ids):
            self.pad = {"ids": ids, "loc0": loc[ids].clone(), "t": 0, "prev": None, "out_max": 0.0, "out_sum": 0.0,
                        "out_n": 0, "sup": None}
            if args.pad_support_r > 0:
                sm = ((loc[:, :2].norm(dim=-1) <= args.pad_support_r) & (loc[:, 2] >= -self.PAD_H)
                      & (loc[:, 2] <= self.TIP_MARGIN) & ~m)
                sid = torch.nonzero(sm).flatten()
                if len(sid):
                    self.pad["sup"] = (sid, loc[sid].clone())
                info["support_nodes"] = int(len(sid))
        return info

    def _apply_pad(self):
        P_ = self.pad
        o, G = self.grip_frame()
        loc0 = P_["loc0"]
        wl = self.PAD_W / 2 - self.INSET
        goal = torch.stack([loc0[:, 0].clamp(-wl, wl), loc0[:, 1],
                            loc0[:, 2].clamp(-self.PAD_H + self.INSET, -self.INSET)], -1)
        P_["t"] += 1
        s = min(1.0, P_["t"] / max(1, round(self.SNAP_S / DT)))
        s = s * s * (3 - 2 * s)
        tl = loc0 + s * (goal - loc0)
        tgt = o + tl @ G.T
        v = (tgt - P_["prev"]) / DT if P_["prev"] is not None else torch.zeros_like(tgt)
        P_["prev"] = tgt.clone()
        pos = T(self.cloth.data.nodal_pos_w)
        vel = T(self.cloth.data.nodal_vel_w).clone()
        vel[0, P_["ids"]] = (tgt - pos[0, P_["ids"]]) / DT + v
        if P_["sup"] is not None:  # support ring: keep its offset in the grip frame, softer gain
            sid, sl0 = P_["sup"]
            st_ = o + sl0 @ G.T
            sv = (st_ - P_["sprev"]) / DT if P_.get("sprev") is not None else torch.zeros_like(st_)
            P_["sprev"] = st_.clone()
            vel[0, sid] = args.pad_support_gain * (st_ - pos[0, sid]) / DT + sv
        self.cloth.write_nodal_velocity_to_sim_index(vel)
        if s >= 1.0:
            cur = (pos[0, P_["ids"]] - o) @ G
            out = torch.stack([cur[:, 0].abs() - self.PAD_W / 2, cur[:, 1].abs() - loc0[:, 1].abs(), cur[:, 2],
                               -self.PAD_H - cur[:, 2]], -1).clamp(min=0.0).amax(-1)
            P_["out_max"] = max(P_["out_max"], float(out.max()))
            P_["out_sum"] += float(out.mean())
            P_["out_n"] += 1

    def grasp_region(self, r):
        """Nodes inside the finger region: within r of the TCP approach axis, between 5 cm below and
        3 cm above the TCP (fingertips are 3.1 cm below it). Returns (ids, offsets in the TCP frame)."""
        p_tcp, q = self.tcp()
        R = quat_to_mat(q[0])
        off = (self.nodes() - p_tcp[0]) @ R
        m = (off[:, :2].norm(dim=-1) < r) & (off[:, 2] > -0.05) & (off[:, 2] < 0.03)
        ids = torch.nonzero(m).flatten()
        if len(ids) == 0:
            ids = torch.argsort((self.nodes() - p_tcp[0]).norm(dim=-1))[:3]
        return ids, off[ids]

    def attach(self, r):
        ids, off = self.grasp_region(r)
        self.pins = (ids, off)
        self._pin_prev = None
        return ids

    # kinematic IK (teleport the right arm, no physics step) -> joint solution
    def _fk_set(self, q):
        self.r.write_joint_position_to_sim_index(position=q)
        self.sim.forward() if hasattr(self.sim, "forward") else None
        self.scene.update(0.0)

    def _ik_err(self, p_des, R_des, w_rot):
        p_tcp, qe = self.tcp()
        Rc = quat_to_mat(qe[0])
        ep = torch.tensor([p_des], device=DEV) - p_tcp
        Re = R_des @ Rc.T
        ang = math.acos(max(-1.0, min(1.0, (float(torch.trace(Re)) - 1) / 2)))
        if ang < 1e-6:
            er = torch.zeros(1, 3, device=DEV)
        elif ang > math.pi - 1e-3:  # 180 deg: pick any perpendicular axis
            er = torch.tensor([[0.0, 0.0, ang]], device=DEV)
        else:
            ax = torch.tensor([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]], device=DEV)
            er = (ax / (2 * math.sin(ang)) * ang).unsqueeze(0)
        cost = float(ep.norm()) + w_rot * ang
        return ep, er, ang, cost, p_tcp

    def solve(self, p_des, R_des, iters=400, w_rot=0.3):
        """Damped least squares with a backtracking line search (step accepted only if the cost drops)."""
        q = T(self.r.data.joint_pos).clone()
        bi = self.bodies.index("ee_right") - (1 if self.r.is_fixed_base else 0)  # fixed base: root not in J
        lo = T(self.r.data.joint_pos_limits)[0, self.arm, 0]
        hi = T(self.r.data.joint_pos_limits)[0, self.arm, 1]
        self._fk_set(q)
        ep, er, ang, cost, p_tcp = self._ik_err(p_des, R_des, w_rot)
        lam = 0.05
        for _ in range(iters):
            if float(ep.norm()) < 1e-4 and ang < 1e-3:
                break
            J = T(self.r.data.body_link_jacobian_w)[:, bi]
            off = J.shape[-1] - len(self.names)
            Ja = J[:, :, [off + i for i in self.arm]]
            r = p_tcp - self.ee()[0]
            Jv, Jw = Ja[:, :3], Ja[:, 3:]
            rx = torch.zeros(1, 3, 3, device=DEV)
            rx[:, 0, 1], rx[:, 0, 2], rx[:, 1, 0] = -r[:, 2], r[:, 1], r[:, 2]
            rx[:, 1, 2], rx[:, 2, 0], rx[:, 2, 1] = -r[:, 0], -r[:, 1], r[:, 0]
            W = torch.tensor([1, 1, 1, w_rot, w_rot, w_rot], device=DEV).view(1, 6, 1)
            Jt = W * torch.cat([Jv - rx @ Jw, Jw], dim=1)
            e = W * torch.cat([ep, er], dim=1).unsqueeze(-1)
            dq = (Jt.transpose(1, 2) @ torch.linalg.solve(Jt @ Jt.transpose(1, 2)
                                                          + lam ** 2 * torch.eye(6, device=DEV), e)).squeeze(-1)
            step, accepted = 1.0, False
            while step > 1e-3:
                qn = q.clone()
                qn[:, self.arm] = torch.max(torch.min(q[:, self.arm] + step * dq.clamp(-0.3, 0.3), hi), lo)
                self._fk_set(qn)
                ep2, er2, ang2, cost2, p2 = self._ik_err(p_des, R_des, w_rot)
                if cost2 < cost:
                    q, ep, er, ang, cost, p_tcp, accepted = qn, ep2, er2, ang2, cost2, p2, True
                    break
                step *= 0.5
            if not accepted:
                lam *= 2.0
                self._fk_set(q)
                if lam > 5.0:
                    break
            else:
                lam = max(0.01, lam * 0.7)
        self._fk_set(q)
        return q[:, self.arm].clone(), float(ep.norm()), ang

    def move(self, q_goal, secs):
        q0 = self.q_tgt[:, self.arm].clone()
        n = max(1, int(secs / DT))
        for k in range(n):
            a = 0.5 * (1 - math.cos(math.pi * (k + 1) / n))
            self.q_tgt[:, self.arm] = q0 + (q_goal - q0) * a
            self.step(1)


# ------------------------------------------------------------------------------------------ run
def main():
    t_start = time.time()
    sim, scene = build()
    bot = Bot(sim, scene)
    bot.grip(0.05)
    q_home = T(bot.r.data.joint_pos).clone()
    # settle the cloth first (robot holds its pose)
    bot.step(int(1.5 / DT))
    nodes0 = bot.nodes().clone()
    info = {"num_nodes": int(nodes0.shape[0])}
    R = LAY["R"]
    # --- waypoints from the settled cloth
    if args.test == "duvet":
        ci = int(torch.argmin(nodes0[:, 0] + nodes0[:, 1]))  # near-right corner
        corner = nodes0[ci]
        sel = (nodes0[:, :2] - corner[:2]).norm(dim=-1) < 0.06
        g = (float(corner[0]) + 0.03, float(corner[1]) + 0.03, float(nodes0[sel, 2].max()) + GRIP_DZ)
        approach = [(g[0], g[1], g[2] + 0.12), g]
        carry = [(g[0] - 0.1 * k, g[1], g[2] + 0.4 / 3 * k) for k in (1, 2, 3)]
        target_ids = torch.nonzero(sel).flatten()
        info["grasp_point"] = g
    else:
        # sleeve: right sleeve end = the most negative-y nodes; floor: a shoulder/hem point near the robot
        if args.test == "sleeve":
            ymin = float(nodes0[:, 1].min())
            sel = nodes0[:, 1] < ymin + 0.04
        else:
            xmin = float(nodes0[:, 0].min())
            sel = (nodes0[:, 0] < xmin + 0.05) & (torch.abs(nodes0[:, 1] - float(nodes0[:, 1].mean())) < 0.08)
        tip = nodes0[sel].mean(dim=0)
        surf = float(nodes0[sel, 2].max())
        # fingertips (TCP - 0.031 m along the approach axis) 2 mm above the cloth surface
        g = (float(tip[0]), float(tip[1]) + (0.03 if args.test == "sleeve" else 0.0), surf + GRIP_DZ)
        approach = [(g[0], g[1], g[2] + 0.12), g]
        if args.test == "sleeve":
            cy = float(nodes0[:, 1].mean())
            carry = [(g[0], g[1], g[2] + 0.12), (g[0], (g[1] + cy) / 2, g[2] + 0.16),
                     (g[0], cy + 0.05, g[2] + 0.08), (g[0], cy + 0.05, g[2] + 0.02)]
        else:
            carry = []  # floor: lift by straightening the torso (deep -> crouch), arm joints fixed
        target_ids = torch.nonzero(sel).flatten()
        info["grasp_point"] = g
    # --- plan with kinematic IK (teleporting the arm only), then restore the state
    plan, ik_err = [], []
    for wpnt in approach + carry:
        qa, ep, er = bot.solve(wpnt, R)
        plan.append(qa)
        ik_err.append((ep, er))
        print(f"[ik] wp={tuple(round(v, 3) for v in wpnt)} pos_err={ep:.4f} rot_err={er:.4f}")
    bot.r.write_joint_position_to_sim_index(position=q_home)
    bot.r.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q_home))
    scene.update(0.0)
    info["ik_err_max_m"] = max(e[0] for e in ik_err)
    info["ik_err_max_rad"] = max(e[1] for e in ik_err)
    snaps = []

    def snap(tag):
        if args.cameras:
            img = T(scene["cam_obs"].data.output["rgb"])[0].cpu().numpy()[..., :3].astype(np.uint8)
            from PIL import Image

            fn = OUT / "images" / f"{RUN}_{tag}.png"
            Image.fromarray(img).save(fn)
            snaps.append(str(fn))

    # --- execute
    bot.step(int(0.3 / DT))
    na = len(approach)
    bot.move(plan[0], 2.0)
    bot.move(plan[1], 1.5)
    bot.step(int(0.3 / DT))
    p_tcp, _ = bot.tcp()
    info["tcp_at_grasp"] = p_tcp[0].tolist()
    info["tcp_grasp_err_m"] = float((p_tcp[0] - torch.tensor(approach[-1], device=DEV)).norm())
    if args.grip == "pad":  # capture on the close command, with the fingers still open (pad_grasp)
        info["pad_capture"] = bot.pad_capture()
    bot.grip(0.0)
    bot.step(int(LAY["hold_close"] / DT))
    q = T(bot.r.data.joint_pos)[0]
    info["finger_gap_m"] = float(q[bot.f["r2"]] - q[bot.f["r1"]])
    # grasped nodes = nodes within 3 cm of the TCP at closing (same set for both modes)
    p_tcp, qe = bot.tcp()
    grasped, off0 = bot.grasp_region(args.assist_r)
    info["grasped_nodes"] = int(len(grasped))
    info["nearest_node_m"] = float((bot.nodes() - p_tcp[0]).norm(dim=-1).min())
    if args.grip == "assist":
        info["pinned_nodes"] = int(len(bot.attach(args.assist_r)))
    snap("grasp")
    drift_max, drift_mean = 0.0, 0.0
    target_start = bot.nodes()[target_ids].mean(dim=0).clone() if len(target_ids) else None
    for k, qk in enumerate(plan[na:]):
        bot.move(qk, 1.5)
        p_tcp, qe = bot.tcp()
        Rk = quat_to_mat(qe[0])
        off = (bot.nodes()[grasped] - p_tcp[0]) @ Rk
        dd = (off - off0).norm(dim=-1)
        drift_max = max(drift_max, float(dd.max()))
        drift_mean = max(drift_mean, float(dd.mean()))
    if args.test == "floor":
        tid = [bot.idx[f"torso_{i}"] for i in (1, 2, 3)]
        t0 = bot.q_tgt[:, tid].clone()
        t1 = torch.tensor([TORSO["crouch"]], device=DEV)
        n = int(3.0 / DT)
        for k in range(n):
            a = 0.5 * (1 - math.cos(math.pi * (k + 1) / n))
            bot.q_tgt[:, tid] = t0 + (t1 - t0) * a
            bot.step(1)
            if k % 60 == 0:
                p_tcp, qe = bot.tcp()
                dd = ((bot.nodes()[grasped] - p_tcp[0]) @ quat_to_mat(qe[0]) - off0).norm(dim=-1)
                drift_max = max(drift_max, float(dd.max()))
                drift_mean = max(drift_mean, float(dd.mean()))
    bot.step(int(1.0 / DT))
    p_tcp, qe = bot.tcp()
    off = (bot.nodes()[grasped] - p_tcp[0]) @ quat_to_mat(qe[0])
    dd = (off - off0).norm(dim=-1)
    info["drift_end_mean_m"] = float(dd.mean())
    info["drift_end_median_m"] = float(dd.median())
    info["drift_max_m"] = max(drift_max, float(dd.max()))
    info["drift_mean_m"] = max(drift_mean, float(dd.mean()))
    info["tcp_travel_m"] = float((p_tcp[0] - torch.tensor(info["tcp_at_grasp"], device=DEV)).norm())
    if target_start is not None:
        info["target_nodes_travel_m"] = float((bot.nodes()[target_ids].mean(dim=0) - target_start).norm())
    snap("carry")
    if args.test == "sleeve":
        # fold quality: sleeve-end nodes should end near the body centre line
        info["sleeve_end_y"] = float(bot.nodes()[target_ids, 1].mean())
        info["body_centre_y"] = float(nodes0[:, 1].mean())
        bot.grip(0.05)
        bot.pins = None
        if bot.pad is not None:
            info["pad_out_max_m"] = bot.pad["out_max"]
            info["pad_out_mean_m"] = bot.pad["out_sum"] / max(1, bot.pad["out_n"])
        bot.pad = None
        bot.step(int(1.5 / DT))
        info["sleeve_end_y_after_release"] = float(bot.nodes()[target_ids, 1].mean())
        snap("release")
    if args.test == "floor":
        info["lift_m"] = float(bot.nodes()[grasped, 2].mean())
        info["tcp_z_end"] = float(bot.tcp()[0][0, 2])
    # --- hold until duration, monitor stability
    t_manip = bot.t
    finite, vmax, ext = True, 0.0, 0.0
    while bot.t < args.duration:
        bot.step(int(0.5 / DT))
        nod = bot.nodes()
        if not torch.isfinite(nod).all():
            finite = False
            break
        v = T(bot.cloth.data.nodal_vel_w)[0]
        vmax = max(vmax, float(v.norm(dim=-1).max()))
        ext = max(ext, float((nod.max(dim=0).values - nod.min(dim=0).values).max()))
    snap("end")
    info.update({"finite": finite, "max_node_speed_after_manip": vmax, "max_extent_m": ext,
                 "manip_time_s": t_manip, "sim_time_s": bot.t, "step_wall_s": bot.wall,
                 "rtf": bot.t / bot.wall if bot.wall else None, "snapshots": snaps})
    stable = finite and vmax < 5.0 and ext < 5.0
    held = info["drift_mean_m"] <= 0.02
    if args.grip == "pad":  # pinned nodes stay in the pad box (out of the box: mean <= 5 mm)
        if bot.pad is not None:
            info["pad_out_max_m"] = bot.pad["out_max"]
            info["pad_out_mean_m"] = bot.pad["out_sum"] / max(1, bot.pad["out_n"])
        held = info["pad_capture"]["nodes"] > 0 and info.get("pad_out_mean_m", 1.0) <= 0.005
    res = {"run": RUN, "args": vars(args), "info": info,
           "criteria": {"grasp_hold(mean drift<=2cm)": held, "stable_60s": stable,
                        "rtf>=0.5": (info["rtf"] or 0) >= 0.5},
           "wall_total_s": time.time() - t_start}
    res["pass"] = held and stable
    (OUT / "runs" / f"{RUN}.json").write_text(json.dumps(res, indent=2))
    if args.grip == "pad":
        print(f"[cloth] pad capture={info['pad_capture']} out_mean={info.get('pad_out_mean_m')} "
              f"out_max={info.get('pad_out_max_m')} target_travel={info.get('target_nodes_travel_m')}")
    print(f"[cloth] {RUN}: pass={res['pass']} drift_mean={info['drift_mean_m']:.4f} max={info['drift_max_m']:.4f} "
          f"stable={stable} rtf={info['rtf']:.2f} nodes={info['num_nodes']} ik={info['ik_err_max_m']:.4f}")


if __name__ == "__main__":
    main()
    app.close()
