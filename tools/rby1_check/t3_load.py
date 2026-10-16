"""T3: load RB-Y1 in Isaac Lab 3.0 (PhysX) and run motion tests.

    python tools/rby1_check/t3_load.py --test idle  --device cuda:0 --dt 500
    python tools/rby1_check/t3_load.py --test idle  --device cuda:0 --dt 200 --mode ideal_pd --effort usd
    python tools/rby1_check/t3_load.py --test crouch|sweep|gripper|drive|camera --device cuda:0
    python tools/rby1_check/t3_load.py --test idle --num_envs 4            # T3g cloning
    python tools/rby1_check/t3_load.py --test idle --device cpu            # T3g CPU vs GPU
    add --viz kit to watch (not headless)

Each run writes reports/rby1_check/t3/<test>_<mode>_<device>_dt<N>_n<envs>[_<tag>].json (+ png).
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")  # same convention as DX USD scripts

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--test", required=True,
                    choices=["idle", "crouch", "sweep", "gripper", "drive", "camera"])
parser.add_argument("--dt", type=int, default=500, help="physics rate in Hz (200 or 500)")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--mode", default="implicit", choices=["implicit", "ideal_pd"])
parser.add_argument("--effort", default="urdf", choices=["urdf", "usd"])
parser.add_argument("--pose", default="zero", choices=["zero", "ready"])
parser.add_argument("--duration", type=float, default=10.0)
parser.add_argument("--preset", default="original", choices=["original", "tuned"],
                    help="rby1_cfg_draft preset (tuned = viscous friction + DAMPING_ZETA07 + l2/r2 driven)")
parser.add_argument("--friction", default="full", choices=["full", "viscous", "none"],
                    help="joint friction: original motor model / viscous only / none")
parser.add_argument("--no_friction", action="store_true", help="alias for --friction none")
parser.add_argument("--damping_ratio", type=float, default=None,
                    help="override body damping so that zeta = value (uses mass-matrix diagonal + armature)")
parser.add_argument("--gravity_comp", action="store_true",
                    help="feed gravity-compensation torques as joint effort targets (body joints), like the SDK feedforward")
parser.add_argument("--grip_mimic", action="store_true", help="drive l2/r2 to mirror l1/r1 (software mimic)")
parser.add_argument("--crouch", default="60,-90,60", help="torso_1,torso_2,torso_3 in degrees")
parser.add_argument("--sweep_joints", default="", help="comma list (default: all body joints except wheels)")
parser.add_argument("--caster_mu", type=float, default=0.3, help="friction of the base caster cylinders (original 0.3)")
parser.add_argument("--fix_root", action="store_true", help="pin the base (used to measure torso tracking without tipping)")
parser.add_argument("--spawn", default="0,0,0", help="robot init position x,y,z (render/pose sync check)")
parser.add_argument("--arms_ready", action="store_true")
parser.add_argument("--usd", default=None)
parser.add_argument("--tag", default="")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.no_friction:
    args.friction = "none"
if args.test == "camera":
    args.enable_cameras = True
app = AppLauncher(args).app

import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from common import CPP_JOINT_NAMES, GEN, WHEEL_RADIUS, WHEEL_TRACK, kit_log_path, out_dir, save_json  # noqa: E402
from rby1_cfg_draft import EFFORT_URDF, apply_original_materials, make_rby1_cfg  # noqa: E402

DEV = args.device
DT = 1.0 / args.dt
OUT = out_dir("t3")
RUN = f"{args.test}_{args.mode}_{'gpu' if 'cuda' in DEV else 'cpu'}_dt{args.dt}_n{args.num_envs}" + (
    f"_{args.tag}" if args.tag else "")
LOG0 = kit_log_path()
LOG0_SIZE = LOG0.stat().st_size if LOG0 else 0
BODY = CPP_JOINT_NAMES[2:]  # torso, arms, head (position-controlled)


def wxyz(q):
    """Isaac Lab 3.0 quaternions are (x, y, z, w); this script computes in (w, x, y, z)."""
    return q[..., [3, 0, 1, 2]]


def T(x):
    """warp / ProxyArray / tensor -> torch tensor."""
    if hasattr(x, "torch"):
        return x.torch
    if isinstance(x, wp.array):
        return wp.to_torch(x)
    return x


# --------------------------------------------------------------------------------------------- scene
def _look_rot(eye, target):
    """(x, y, z, w) rotation for an Isaac Lab camera in 'world' convention (+x forward, +z up)."""
    d = [t - e for t, e in zip(target, eye)]
    yw, pt = math.atan2(d[1], d[0]), math.atan2(-d[2], math.hypot(d[0], d[1]))
    cy, sy, cp, sp = math.cos(yw / 2), math.sin(yw / 2), math.cos(pt / 2), math.sin(pt / 2)
    return (-sy * sp, cy * sp, sy * cp, cy * cp)  # q_yaw(z) * q_pitch(y)


def build_scene():
    robot_cfg = make_rby1_cfg(mode=args.mode, usd_path=args.usd, effort=args.effort,
                              friction=args.friction, preset=args.preset)
    if args.fix_root:
        robot_cfg.spawn.articulation_props.fix_root_link = True
    robot_cfg.init_state.pos = tuple(float(v) for v in args.spawn.split(","))
    if args.arms_ready:  # dxbench READY_ARMS (elbows bent, forearms forward)
        jp = {n: 0.0 for n in ["right_wheel", "left_wheel"] + [f"torso_{i}" for i in range(6)] + ["head_0", "head_1"]
              + [f"{s}_arm_{i}" for s in ("right", "left") for i in range(7)]}
        jp.update({"right_arm_0": -0.5, "right_arm_1": -0.2, "right_arm_3": -1.6, "right_arm_5": 0.3,
                   "left_arm_0": -0.5, "left_arm_1": 0.2, "left_arm_3": -1.6, "left_arm_5": 0.3,
                   "gripper_finger_[lr]1": 0.0, "gripper_finger_[lr]2": 0.0})
        robot_cfg.init_state.joint_pos = jp

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
        light = AssetBaseCfg(prim_path="/World/light",
                             spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)))
        robot = robot_cfg

    if args.test == "gripper":
        SceneCfg.cube = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube",
            spawn=sim_utils.CuboidCfg(
                size=(0.05, 0.05, 0.05),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.2, 0.2)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(3.0, 0.0, 0.025)),
        )
        SceneCfg.pedestal = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Pedestal",
            spawn=sim_utils.CuboidCfg(
                size=(0.03, 0.03, 0.6),  # narrower than the cube so the fingertips clear it
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.8)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(3.0, 1.0, 0.3)),
        )
    if args.test == "camera":
        from isaaclab.sensors import CameraCfg

        cams = {"head": "link_head_2/first_person_camera",
                "left_wrist": "left_gripper/ee_left/left_hand_camera",
                "right_wrist": "right_gripper/ee_right/right_hand_camera"}
        for k, rel in cams.items():
            setattr(SceneCfg, f"cam_{k}", CameraCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{rel}", spawn=None, width=640, height=480,
                data_types=["rgb", "distance_to_image_plane"], update_period=0.0,
                update_latest_camera_pose=True,  # default False keeps the pose from initialisation
            ))
        # observer camera looking at the robot from the front-left (checks that the robot itself renders)
        SceneCfg.cam_observer = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Observer", width=960, height=720, data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=14.0, horizontal_aperture=20.955, clipping_range=(0.03, 50.0)),
            offset=CameraCfg.OffsetCfg(pos=(2.4, 1.6, 1.7), rot=_look_rot((2.4, 1.6, 1.7), (0.0, 0.0, 0.8)),
                                       convention="world"))
        # props so that every camera sees structure (not a blank floor)
        for i, (pos, col, size) in enumerate([
            ((1.2, 0.0, 0.5), (0.8, 0.2, 0.2), (0.3, 1.0, 1.0)),
            ((0.7, -0.4, 0.15), (0.2, 0.7, 0.2), (0.2, 0.2, 0.3)),
            ((0.7, 0.4, 0.15), (0.2, 0.3, 0.8), (0.2, 0.2, 0.3)),
            ((0.5, 0.0, 0.05), (0.9, 0.8, 0.1), (0.1, 0.6, 0.1)),
        ]):
            setattr(SceneCfg, f"prop{i}", AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Prop{i}",
                spawn=sim_utils.CuboidCfg(size=size, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=col),
                                          collision_props=sim_utils.CollisionPropertiesCfg()),
                init_state=AssetBaseCfg.InitialStateCfg(pos=pos)))

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=DEV, dt=DT, render_interval=max(1, args.dt // 30)))
    scene = InteractiveScene(SceneCfg(num_envs=args.num_envs, env_spacing=3.0))
    n_mat = apply_original_materials(caster_mu=args.caster_mu)
    sim.reset()
    return sim, scene, n_mat


# --------------------------------------------------------------------------------------------- helpers
class Rec:
    """Records per-step state on the sim device, converted to numpy at the end."""

    def __init__(self):
        self.d: dict[str, list] = {}

    def add(self, **kw):
        for k, v in kw.items():
            self.d.setdefault(k, []).append(v.detach().clone() if torch.is_tensor(v) else v)

    def np(self, k):
        v = self.d[k]
        return torch.stack(v).cpu().numpy() if torch.is_tensor(v[0]) else np.asarray(v)


class Bot:
    def __init__(self, sim, scene):
        self.sim, self.scene = sim, scene
        self.r = scene["robot"]
        self.names = list(self.r.joint_names)
        self.bodies = list(self.r.body_names)
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.n = scene.num_envs
        self.t = 0.0
        self.q_tgt = T(self.r.data.default_joint_pos).clone()
        self.v_tgt = torch.zeros_like(self.q_tgt)
        self.origins = scene.env_origins.clone()
        self.wheel_pos_tgt = torch.zeros(self.n, 2, device=DEV)
        self.body_ids = [self.idx[n] for n in BODY]

    def j(self, names):
        return [self.idx[n] for n in names]

    def b(self, name):
        return self.bodies.index(name)

    def set_target(self, q=None, v=None):
        if q is not None:
            self.q_tgt[:] = q
        if v is not None:
            self.v_tgt[:] = v

    def step(self, n=1, cb=None):
        for _ in range(n):
            if cb:
                cb(self)
            self.r.set_joint_position_target_index(target=self.q_tgt)
            self.r.set_joint_velocity_target_index(target=self.v_tgt)
            if args.gravity_comp:
                g = self.gravity()
                ff = torch.zeros_like(self.q_tgt)
                ids = self.body_ids
                ff[:, ids] = g[:, g.shape[-1] - len(self.names):][:, ids]
                self.r.set_joint_effort_target_index(target=ff)
            self.scene.write_data_to_sim()
            self.sim.step(render=args.test == "camera" or bool(args.visualizer))
            self.scene.update(DT)
            self.t += DT

    # state
    def q(self):
        return T(self.r.data.joint_pos)

    def dq(self):
        return T(self.r.data.joint_vel)

    def root_pos(self):  # the mobile base link, independent of which link PhysX made the root
        return self.body_pos("base")

    def root_quat(self):
        return self.body_quat("base")

    def body_pos(self, name):
        return T(self.r.data.body_link_pos_w)[:, self.b(name)] - self.origins

    def body_quat(self, name):  # (w, x, y, z)
        return wxyz(T(self.r.data.body_link_quat_w)[:, self.b(name)])

    def applied(self):
        return T(self.r.data.applied_torque)

    def projected(self):
        try:
            return T(self.r.root_view.get_dof_projected_joint_forces())
        except Exception:  # noqa: BLE001
            return torch.full_like(self.q(), float("nan"))

    def gravity(self):
        return T(self.r.data.gravity_compensation_forces)

    def mass_diag(self):
        M = T(self.r.data.mass_matrix)
        off = M.shape[-1] - len(self.names)
        return torch.diagonal(M, dim1=-2, dim2=-1)[:, off:]

    def write_state(self, q, root_pos=None):
        self.r.write_joint_position_to_sim_index(position=q)
        self.r.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q))
        pose = T(self.r.data.default_root_pose).clone()
        pose[:, :3] += self.origins
        if root_pos is not None:
            pose[:, :3] += root_pos
        self.r.write_root_pose_to_sim_index(root_pose=pose)
        self.r.write_root_velocity_to_sim_index(root_velocity=torch.zeros(self.n, 6, device=DEV))
        self.scene.reset()
        self.q_tgt[:] = q
        self.v_tgt.zero_()


def tilt_deg(quat):
    """Angle between body z-axis and world z (quat wxyz, torch)."""
    w, x, y, z = quat.unbind(-1)
    zz = 1 - 2 * (x * x + y * y)
    return torch.rad2deg(torch.acos(zz.clamp(-1, 1)))


def yaw(quat):
    w, x, y, z = quat.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def nan_check(bot, rec_q):
    q = rec_q
    return bool(np.isnan(q).any() or np.abs(q).max() > 1e3 or np.isnan(bot.root_pos().cpu().numpy()).any())


def per_joint(names, arr):
    return {n: float(v) for n, v in zip(names, arr)}


def ramp(t, T0):
    """Smooth 0->1 over T0 seconds (half-cosine)."""
    if t >= T0:
        return 1.0
    return 0.5 * (1 - math.cos(math.pi * t / T0))


def ready_pose(bot):
    """Read non-zero joint state values from rby1_ready_reach_table.usd (if any)."""
    from pxr import Usd

    stage = Usd.Stage.Open(str(GEN / "rby1_ready_reach_table.usd"))
    found = {}
    for p in stage.Traverse():
        a = p.GetAttribute("state:angular:physics:position")
        if a and a.HasAuthoredValue() and p.GetName() in bot.idx:
            found[p.GetName()] = math.radians(a.Get())
    return found


def damping_ratio_report(bot, q0):
    """zeta = (D + viscous) / (2 sqrt(K (M_ii + armature))) per body joint at the current pose."""  # noqa: D401
    Md = bot.mass_diag()[0].cpu().numpy()
    K = T(bot.r.data.joint_stiffness)[0].cpu().numpy()
    D = T(bot.r.data.joint_damping)[0].cpu().numpy()
    arm = T(bot.r.data.joint_armature)[0].cpu().numpy()
    visc = T(bot.r.data.joint_viscous_friction_coeff)[0].cpu().numpy()
    out = {}
    for n in BODY:
        i = bot.idx[n]
        # implicit drives live in the solver; ideal_pd keeps gains in the actuator model
        k, d = K[i], D[i]
        if args.mode == "ideal_pd":
            from rby1_cfg_draft import DAMPING, KP
            k, d = KP[n], DAMPING[n]
        m = Md[i] + arm[i]  # the PhysX mass matrix diagonal does NOT include the joint armature
        out[n] = {"K": float(k), "D": float(d), "viscous": float(visc[i]), "M_ii": float(m), "armature": float(arm[i]),
                  "zeta_drive": float(d / (2 * math.sqrt(k * m))) if k > 0 and m > 0 else None,
                  "zeta_total": float((d + visc[i]) / (2 * math.sqrt(k * m))) if k > 0 and m > 0 else None,
                  "D_for_zeta_0.7": float(1.4 * math.sqrt(k * m)) if k > 0 else None,
                  "D_for_zeta_1.0": float(2.0 * math.sqrt(k * m)) if k > 0 else None,
                  "f_n_hz": float(math.sqrt(k / m) / (2 * math.pi)) if k > 0 and m > 0 else None}
    return out


def oscillation_metrics(t, err):
    """Per-joint: zero crossings of the error after 1 s and peak-to-peak in the last 3 s."""
    m = t > 1.0
    zc = (np.diff(np.sign(err[m]), axis=0) != 0).sum(axis=0)
    last = t > (t[-1] - 3.0)
    p2p = err[last].max(axis=0) - err[last].min(axis=0)
    return zc, p2p


def plot(path, t, series: dict, ylabel, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for k, v in series.items():
        ax.plot(t, v, label=k, lw=1)
    ax.set_xlabel("t [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    if len(series) <= 12:
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def effort_limit_vec(bot):
    import re

    lim = np.full(len(bot.names), np.nan)
    for n, i in bot.idx.items():
        for k, v in EFFORT_URDF.items():
            if re.fullmatch(k, n):
                lim[i] = v
    return lim


def apply_damping_ratio(bot, zeta):
    """Overwrite body-joint damping (implicit: solver drive; ideal_pd: actuator gains) for zeta.

    Uses the mass-matrix diagonal at the current pose plus the joint armature."""
    Md = bot.mass_diag()
    K = T(bot.r.data.joint_stiffness).clone()
    if args.mode == "ideal_pd":
        from rby1_cfg_draft import KP

        for n in BODY:
            K[:, bot.idx[n]] = KP[n]
    ids = bot.j(BODY)
    M = Md[:, ids] + T(bot.r.data.joint_armature)[:, ids]  # mass-matrix diagonal excludes armature
    D = 2 * zeta * torch.sqrt(K[:, ids] * M)
    if args.mode == "implicit":
        bot.r.write_joint_damping_to_sim_index(damping=D, joint_ids=ids)
    else:
        for act in bot.r.actuators.values():
            jn = [bot.names[i] for i in act.joint_indices] if hasattr(act, "joint_indices") else []
            for k, n in enumerate(jn):
                if n in BODY:
                    T(act.damping)[:, k] = D[:, BODY.index(n)]
    return {n: float(D[0, k]) for k, n in enumerate(BODY)}


# --------------------------------------------------------------------------------------------- tests
def hold_test(bot, q_goal, ramp_s, hold_s, name):
    """Ramp from current targets to q_goal, then hold. Returns metrics + recorded arrays."""
    rec = Rec()
    q_start = bot.q_tgt.clone()
    steps = int((ramp_s + hold_s) / DT)
    root0 = bot.root_pos().clone()
    lim = torch.tensor(effort_limit_vec(bot), device=DEV)
    for k in range(steps):
        a = ramp(bot.t if False else k * DT, ramp_s) if ramp_s > 0 else 1.0
        bot.q_tgt[:] = q_start + (q_goal - q_start) * a
        bot.step(1)
        if k % max(1, args.dt // 100) == 0:  # record at 100 Hz
            rec.add(t=k * DT, q=bot.q(), qt=bot.q_tgt.clone(), root=bot.root_pos(), tilt=tilt_deg(bot.root_quat()),
                    tau=bot.applied(), proj=bot.projected(), grav=bot.gravity())
    t = rec.np("t")
    q, qt = rec.np("q"), rec.np("qt")
    ids = bot.j(BODY)
    err = q[:, :, ids] - qt[:, :, ids]  # (T, envs, J)
    root = rec.np("root")
    disp = np.linalg.norm(root - root[0:1], axis=-1)  # (T, envs)
    settle = t > 0.5
    hold = t >= ramp_s + 1.0
    tau, proj, grav = rec.np("tau"), rec.np("proj"), rec.np("grav")
    limv = lim.cpu().numpy()
    res = {
        "nan": nan_check(bot, q),
        "max_err_rad_all": float(np.abs(err).max()),
        "max_err_rad_hold": float(np.abs(err[hold]).max()) if hold.any() else None,
        "max_err_by_joint_hold": per_joint(BODY, np.abs(err[hold]).max(axis=(0, 1))) if hold.any() else {},
        "rms_err_by_joint_hold": per_joint(BODY, np.sqrt((err[hold] ** 2).mean(axis=(0, 1)))) if hold.any() else {},
        "base_disp_m_total": float(disp.max()),
        "base_disp_m_after_0p5s": float(np.linalg.norm(root[settle] - root[settle][0:1], axis=-1).max()),
        "base_z_range_m": [float(root[..., 2].min()), float(root[..., 2].max())],
        "max_tilt_deg": float(rec.np("tilt").max()),
        "per_env_max_err_hold": [float(np.abs(err[hold][:, e]).max()) for e in range(bot.n)] if hold.any() else [],
    }
    zc, p2p = oscillation_metrics(t, err[:, 0])
    res["err_zero_crossings_after_1s"] = per_joint(BODY, zc)
    res["err_p2p_last3s_rad"] = per_joint(BODY, p2p)
    if hold.any():
        tau_h = np.abs(tau[hold][:, 0]).max(axis=0)
        proj_h = np.abs(proj[hold][:, 0]).max(axis=0)
        grav_h = np.abs(grav[hold][:, 0])
        grav_h = grav_h[:, -len(bot.names):].mean(axis=0) if grav_h.ndim == 2 else grav_h
        res["torque_hold"] = {n: {"applied_max": float(tau_h[bot.idx[n]]), "projected_max": float(proj_h[bot.idx[n]]),
                                  "gravity_comp": float(grav_h[bot.idx[n]]), "urdf_effort": float(limv[bot.idx[n]]),
                                  "ratio_projected": float(proj_h[bot.idx[n]] / limv[bot.idx[n]]),
                                  "ratio_gravity": float(grav_h[bot.idx[n]] / limv[bot.idx[n]])} for n in BODY}
    plot(OUT / f"{RUN}_{name}_err.png", t, {n: err[:, 0, k] for k, n in enumerate(BODY)
                                            if np.abs(err[:, 0, k]).max() > 0.2 * np.abs(err).max()},
         "q - q_target [rad]", f"{RUN} {name}: joint tracking error (largest joints)")
    return res, rec


def test_idle(bot):
    q0 = T(bot.r.data.default_joint_pos).clone()
    info = {}
    if args.pose == "ready":
        rp = ready_pose(bot)
        info["ready_pose_deg"] = {k: math.degrees(v) for k, v in rp.items()}
        for n, v in rp.items():
            q0[:, bot.idx[n]] = v
        bot.write_state(q0)
    info["damping_ratio"] = damping_ratio_report(bot, q0)
    if args.damping_ratio:
        info["damping_override"] = apply_damping_ratio(bot, args.damping_ratio)
    res, rec = hold_test(bot, q0, 0.0, args.duration, "idle")
    hold = rec.np("t") >= 1.0
    res["max_err_rad_hold"] = float(np.abs(rec.np("q")[hold][:, :, bot.j(BODY)] - rec.np("qt")[hold][:, :, bot.j(BODY)]).max())
    crit = {"no_nan": not res["nan"], "err<=0.02": res["max_err_rad_all"] <= 0.02,
            "base<=0.01m": res["base_disp_m_total"] <= 0.01}
    return {**info, **res, "criteria": crit, "pass": all(crit.values())}


def test_crouch(bot):
    a, b, c = [math.radians(float(x)) for x in args.crouch.split(",")]
    q = T(bot.r.data.default_joint_pos).clone()
    q[:, bot.idx["torso_1"]] = a
    q[:, bot.idx["torso_2"]] = b
    q[:, bot.idx["torso_3"]] = c
    res, rec = hold_test(bot, q, 4.0, 6.0, "crouch")
    tq = res.get("torque_hold", {})
    worst = max(tq.items(), key=lambda kv: kv[1]["ratio_projected"]) if tq else (None, {})
    crit = {"no_nan": not res["nan"], "err<=0.05": (res["max_err_rad_hold"] or 9) <= 0.05,
            "torque<=0.8*effort": all(v["ratio_projected"] <= 0.8 for v in tq.values()),
            "no_tip(tilt<5deg)": res["max_tilt_deg"] < 5.0}
    # end-effector heights tell how far down the crouch reaches
    res["ee_z_m"] = {s: float(bot.body_pos(f"ee_{s}")[0, 2]) for s in ("left", "right")}
    res["torso5_pos_m"] = bot.body_pos("link_torso_5")[0].tolist()
    return {"crouch_deg": args.crouch, **res, "worst_torque_joint": worst[0], "criteria": crit,
            "pass": all(crit.values())}


def test_sweep(bot):
    import re

    joints = [j for j in (args.sweep_joints.split(",") if args.sweep_joints else BODY) if j]
    lo = T(bot.r.data.joint_pos_limits)[0, :, 0].cpu().numpy()
    hi = T(bot.r.data.joint_pos_limits)[0, :, 1].cpu().numpy()
    q0 = T(bot.r.data.default_joint_pos).clone()
    period = 10.0
    out = {}
    body_names_lowest = [b for b in bot.bodies if not re.match(r"wheel_|base$", b)]
    for jn in joints:
        i = bot.idx[jn]
        bot.write_state(q0)
        bot.step(int(0.5 / DT))
        a_hi, a_lo = float(0.8 * hi[i]), float(0.8 * lo[i])
        rec = Rec()
        steps = int(period / DT)
        for k in range(steps):
            ph = k * DT / period  # 0..1 : 0 -> hi -> 0 -> lo -> 0
            s = math.sin(2 * math.pi * ph)
            amp = a_hi if s >= 0 else -a_lo
            bot.q_tgt[:, i] = s * amp
            bot.v_tgt[:, i] = amp * 2 * math.pi / period * math.cos(2 * math.pi * ph)  # velocity feedforward
            bot.step(1)
            if k % max(1, args.dt // 100) == 0:
                zmin = min(float(bot.body_pos(b)[0, 2]) for b in body_names_lowest)
                rec.add(t=k * DT, q=bot.q()[0, i], qt=bot.q_tgt[0, i], tilt=tilt_deg(bot.root_quat())[0],
                        qall=bot.q()[0], zmin=zmin, tau=bot.applied()[0, i])
        q, qt = rec.np("q"), rec.np("qt")
        e = q - qt
        others = [bot.idx[n] for n in BODY if n != jn]
        qall = rec.np("qall")
        out[jn] = {
            "range_cmd_rad": [float(a_lo), float(a_hi)],
            "rms_err_rad": float(np.sqrt((e ** 2).mean())), "max_err_rad": float(np.abs(e).max()),
            "overshoot_hi_rad": float(max(0.0, q.max() - qt.max())), "overshoot_lo_rad": float(max(0.0, qt.min() - q.min())),
            "reached_hi_rad": float(q.max()), "reached_lo_rad": float(q.min()),
            "max_tau_Nm": float(np.abs(rec.np("tau")).max()),
            "other_joints_max_dev_rad": float(np.abs(qall[:, others] - q0[0, others].cpu().numpy()).max()),
            "max_tilt_deg": float(rec.np("tilt").max()), "min_link_origin_z_m": float(rec.np("zmin").min()),
            "nan": bool(np.isnan(q).any() or np.abs(q).max() > 100),
        }
        out[jn]["flags"] = [f for f, c in [
            ("NaN/explode", out[jn]["nan"]), ("tip>10deg", out[jn]["max_tilt_deg"] > 10),
            ("link below floor (no collider)", out[jn]["min_link_origin_z_m"] < -0.02),
            ("max_err>0.1rad", out[jn]["max_err_rad"] > 0.1)] if c]
        print(f"[sweep] {jn}: rms {out[jn]['rms_err_rad']:.4f} max {out[jn]['max_err_rad']:.4f} "
              f"os {out[jn]['overshoot_hi_rad']:.4f}/{out[jn]['overshoot_lo_rad']:.4f} flags {out[jn]['flags']}")
    crit = {"no_nan": not any(v["nan"] for v in out.values()),
            "no_tip": not any(v["max_tilt_deg"] > 10 for v in out.values())}
    return {"period_s": period, "joints": out, "criteria": crit, "pass": all(crit.values())}


def test_drive(bot):
    """Wheel kinematics. Sign convention found in T3e: +wheel joint velocity drives the base BACKWARD
    (wheel joint frames are rotated 180 deg about x), so forward speed v maps to omega = -v / R.
    Unlimited revolute joints report positions wrapped to [-pi, pi]; angles are unwrapped per step."""
    res = {}
    wi = bot.j(["right_wheel", "left_wheel"])

    def wrap(a):
        return (a + math.pi) % (2 * math.pi) - math.pi

    for name, (vr, vl), dur in [("straight", (0.3, 0.3), 3.0),
                                ("spin", (0.5 * WHEEL_TRACK / 2, -0.5 * WHEEL_TRACK / 2), 3.0)]:
        bot.write_state(T(bot.r.data.default_joint_pos).clone())
        bot.step(int(0.5 / DT))
        w_cmd = -torch.tensor([vr / WHEEL_RADIUS, vl / WHEEL_RADIUS], device=DEV)
        p0, y0 = bot.root_pos()[0].clone(), yaw(bot.root_quat())[0].item()
        q_prev = bot.q()[:, wi].clone()
        meas = torch.zeros(bot.n, 2, device=DEV)  # unwrapped measured wheel angle since start
        tgt = torch.zeros(bot.n, 2, device=DEV)  # integrated command (unwrapped)
        yaw_unwrapped, y_prev = 0.0, y0
        max_wrap_jump = 0.0
        for k in range(int(dur / DT)):
            w = w_cmd * ramp(k * DT, 0.5)
            tgt += w * DT
            bot.v_tgt[:, wi] = w
            if args.mode == "ideal_pd":  # original: integrated position target + kp/kd, error taken unwrapped
                bot.q_tgt[:, wi] = bot.q()[:, wi] + (tgt - meas)
            bot.step(1)
            q_now = bot.q()[:, wi]
            d = q_now - q_prev
            max_wrap_jump = max(max_wrap_jump, float(d.abs().max()))
            meas += (d + math.pi) % (2 * math.pi) - math.pi
            q_prev = q_now.clone()
            y = yaw(bot.root_quat())[0].item()
            yaw_unwrapped += wrap(y - y_prev)
            y_prev = y
        bot.v_tgt[:, wi] = 0.0
        p1 = bot.root_pos()[0]
        d = (p1 - p0)[:2].cpu().numpy()
        heading = np.array([math.cos(y0), math.sin(y0)])
        th, cmd = meas[0], tgt[0]
        r = {"cmd_wheel_rad": cmd.tolist(), "meas_wheel_rad_unwrapped": th.tolist(),
             "raw_joint_pos_jump_max_rad": max_wrap_jump,
             "disp_xy_m": d.tolist(), "forward_m": float(d @ heading), "lateral_m": float(np.cross(heading, d)),
             "yaw_rad": yaw_unwrapped}
        v_c = -cmd * WHEEL_RADIUS  # forward travel of each wheel (r, l)
        v_m = -th * WHEEL_RADIUS
        if name == "straight":
            pc, pm = float(v_c.mean()), float(v_m.mean())
            r.update(pred_from_cmd_m=pc, pred_from_wheels_m=pm,
                     err_vs_cmd=abs(r["forward_m"] - pc) / abs(pc), err_vs_wheels=abs(r["forward_m"] - pm) / max(1e-6, abs(pm)),
                     wheel_tracking_err=float(((th - cmd).abs() / cmd.abs()).max()))
        else:
            pc = float((v_c[0] - v_c[1]) / WHEEL_TRACK)
            pm = float((v_m[0] - v_m[1]) / WHEEL_TRACK)
            r.update(pred_yaw_from_cmd=pc, pred_yaw_from_wheels=pm,
                     err_vs_cmd=abs(yaw_unwrapped - pc) / abs(pc), err_vs_wheels=abs(yaw_unwrapped - pm) / max(1e-6, abs(pm)),
                     wheel_tracking_err=float(((th - cmd).abs() / cmd.abs()).max()))
        res[name] = r
        print(f"[drive] {name}: {json.dumps(r)}")
    crit = {"straight<=10%": res["straight"]["err_vs_cmd"] <= 0.10, "spin<=10%": res["spin"]["err_vs_cmd"] <= 0.10}
    return {**res, "caster_mu": args.caster_mu, "criteria": crit, "pass": all(crit.values())}


# ---- gripper -----------------------------------------------------------------------------------------
def tcp_offset_from_usd(side="right"):
    """TCP in ee_<side> frame = centre of the finger collision boxes (closed fingers, from USD)."""
    from pxr import Usd, UsdGeom

    from common import ROBOT_USD

    stage = Usd.Stage.Open(str(args.usd or ROBOT_USD))
    root = stage.GetDefaultPrim()
    ee = root.GetPrimAtPath(f"{side}_gripper/ee_{side}")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"], useExtentsHint=False)
    s = side[0]
    pts = []
    for f in (f"ee_finger_{s}1", f"ee_finger_{s}2"):
        mesh = root.GetPrimAtPath(f"{side}_gripper/{f}/collisions/finger_mesh")
        w = cache.ComputeWorldBound(mesh).ComputeAlignedRange()
        pts += [w.GetMin(), w.GetMax()]
    ee_w = np.array(UsdGeom.Xformable(ee).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    R, t = ee_w[:3, :3].T, ee_w[3, :3]
    lo = np.min(np.array(pts), axis=0)
    hi = np.max(np.array(pts), axis=0)
    c_world = (lo + hi) / 2
    tip_world = None
    off = R.T @ (c_world - t)
    # approach axis: from ee origin towards the finger box centre (dominant ee axis)
    ax = np.zeros(3)
    ax[int(np.argmax(np.abs(off)))] = np.sign(off[int(np.argmax(np.abs(off)))])
    ext = R.T @ (hi - lo)
    return off, ax, np.abs(ext), tip_world


def quat_apply(q, v):
    """q (wxyz) torch (N,4), v (N,3)."""
    w, xyz = q[:, :1], q[:, 1:]
    t = 2 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def ik_step(bot, ee_name, arm_ids, tcp_off, ax_local, p_des, ax_des, gain=1.0, lam=0.05):
    """One damped-least-squares step on position of the TCP + alignment of the approach axis."""
    bi = bot.b(ee_name)
    J = T(bot.r.data.body_link_jacobian_w)[:, bi]  # (N, 6, dof[+6])
    off = J.shape[-1] - len(bot.names)
    Ja = J[:, :, [off + i for i in arm_ids]]
    q_ee = bot.body_quat(ee_name)
    p_ee = bot.body_pos(ee_name)
    r = quat_apply(q_ee, tcp_off.expand(bot.n, 3))
    p_tcp = p_ee + r
    a = quat_apply(q_ee, ax_local.expand(bot.n, 3))
    e_p = p_des - p_tcp
    e_r = torch.cross(a, ax_des.expand(bot.n, 3), dim=-1)
    # TCP linear velocity = v + w x r  ->  J_tcp = Jv - [r]x Jw
    Jv, Jw = Ja[:, :3], Ja[:, 3:]
    rx = torch.zeros(bot.n, 3, 3, device=DEV)
    rx[:, 0, 1], rx[:, 0, 2], rx[:, 1, 0] = -r[:, 2], r[:, 1], r[:, 2]
    rx[:, 1, 2], rx[:, 2, 0], rx[:, 2, 1] = -r[:, 0], -r[:, 1], r[:, 0]
    Jt = torch.cat([Jv - rx @ Jw, Jw], dim=1)
    e = torch.cat([e_p, e_r], dim=1).unsqueeze(-1)
    JJt = Jt @ Jt.transpose(1, 2) + (lam ** 2) * torch.eye(6, device=DEV)
    dq = (Jt.transpose(1, 2) @ torch.linalg.solve(JJt, e)).squeeze(-1)
    return dq * gain, p_tcp, float(e_p.norm(dim=-1).max()), float(e_r.norm(dim=-1).max())


def test_gripper(bot):
    res = {}
    fl = bot.j(["gripper_finger_r1", "gripper_finger_r2", "gripper_finger_l1", "gripper_finger_l2"])
    r1, r2, l1, l2 = fl

    def set_grip(open_m, mimic=args.grip_mimic):
        # l1/r1: 0 closed, -0.05 open ; l2/r2 mirror range (0..0.05)
        bot.q_tgt[:, r1] = -open_m
        bot.q_tgt[:, l1] = -open_m
        if mimic:
            bot.q_tgt[:, r2] = open_m
            bot.q_tgt[:, l2] = open_m

    if args.grip_mimic:  # give the passive fingers the same drive as the driven ones
        bot.r.write_joint_stiffness_to_sim_index(stiffness=torch.full((bot.n, 2), 100.0, device=DEV), joint_ids=[r2, l2])
        bot.r.write_joint_damping_to_sim_index(damping=torch.full((bot.n, 2), 1.0, device=DEV), joint_ids=[r2, l2])
        bot.r.write_joint_effort_limit_to_sim_index(limits=torch.full((bot.n, 2), 75.0, device=DEV), joint_ids=[r2, l2])

    # 1) open / close in the air: symmetry
    sym = []
    for target in (0.05, 0.0, 0.05, 0.0):
        set_grip(target)
        bot.step(int(1.5 / DT))
        q = bot.q()[0]
        sym.append({"cmd_open_m": target, "r1": float(q[r1]), "r2": float(q[r2]), "l1": float(q[l1]), "l2": float(q[l2]),
                    "asym_r_m": float(abs(q[r1] + q[r2])), "asym_l_m": float(abs(q[l1] + q[l2]))})
    res["open_close"] = sym
    res["max_asym_m"] = max(max(s["asym_r_m"], s["asym_l_m"]) for s in sym)

    # 2) grasp a 5 cm cube with the right hand and lift 20 cm
    off, ax, ext, _ = tcp_offset_from_usd("right")
    res["tcp_offset_in_ee_right_m"] = off.tolist()
    res["approach_axis_in_ee_right"] = ax.tolist()
    res["finger_box_extent_m"] = ext.tolist()
    tcp_off = torch.tensor(off, device=DEV, dtype=torch.float32)
    ax_local = torch.tensor(ax, device=DEV, dtype=torch.float32)
    ax_down = torch.tensor([0.0, 0.0, -1.0], device=DEV)
    arm = bot.j([f"right_arm_{i}" for i in range(7)])
    cube_c = torch.tensor([[0.50, -0.25, 0.95]], device=DEV).repeat(bot.n, 1)
    ped = bot.scene["pedestal"]
    cube = bot.scene["cube"]

    def place(obj, pos, yaw=0.0):
        pose = T(obj.data.default_root_pose).clone()
        pose[:, :3] = pos + bot.origins
        pose[:, 3:] = torch.tensor([0.0, 0, math.sin(yaw / 2), math.cos(yaw / 2)], device=DEV)  # (x, y, z, w)
        obj.write_root_pose_to_sim_index(root_pose=pose)
        obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros(bot.n, 6, device=DEV))

    def solve_ik(p_des, iters=1000):
        """Kinematic IK: teleport the arm joints (no physics step) until the TCP reaches p_des."""
        q = bot.q().clone()
        for _ in range(iters):
            dq, p_tcp, e_p, e_r = ik_step(bot, "ee_right", arm, tcp_off, ax_local, p_des, ax_down, gain=0.5, lam=0.02)
            q[:, arm] += dq.clamp(-0.05, 0.05)
            bot.r.write_joint_position_to_sim_index(position=q)
            bot.sim.forward() if hasattr(bot.sim, "forward") else None
            bot.scene.update(0.0)
        return q[:, arm].clone(), e_p, e_r

    def move_arm(q_goal, secs, settle=1.0):
        q_from = bot.q_tgt[:, arm].clone()
        for k in range(int(secs / DT)):
            bot.q_tgt[:, arm] = q_from + (q_goal - q_from) * ramp(k * DT, secs)
            bot.step(1)
        bot.step(int(settle / DT))

    def tcp_now():
        q_ee = bot.body_quat("ee_right")
        return bot.body_pos("ee_right") + quat_apply(q_ee, tcp_off.expand(bot.n, 3)), q_ee

    # seed: elbow bent, forearm forward (the zero pose is a straight-arm singularity)
    seed = {"right_arm_0": -0.5, "right_arm_1": -0.2, "right_arm_3": -1.6, "right_arm_5": 0.3}
    q_hold = bot.q().clone()
    for n, v in seed.items():
        q_hold[:, bot.idx[n]] = v
    bot.write_state(q_hold)
    # plan the three arm configurations before the cube exists
    q_pre, e1p, e1r = solve_ik(cube_c + torch.tensor([0, 0, 0.12], device=DEV))
    q_grasp, e2p, e2r = solve_ik(cube_c)
    q_lift, e3p, e3r = solve_ik(cube_c + torch.tensor([0, 0, 0.20], device=DEV))
    res["ik_err"] = {"pre_pos_m": e1p, "pre_axis": e1r, "grasp_pos_m": e2p, "grasp_axis": e2r, "lift_pos_m": e3p, "lift_axis": e3r}
    bot.write_state(q_hold)
    bot.step(int(0.5 / DT))
    set_grip(0.05)
    move_arm(q_pre, 2.0)
    move_arm(q_grasp, 1.5)
    p_tcp, q_ee = tcp_now()
    res["reach_err"] = {"grasp_pos_m": float((p_tcp - cube_c).norm(dim=-1).max())}
    # put pedestal + cube where the TCP actually is (removes arm-tracking error from the grasp test)
    R = quat_to_mat(q_ee[0].cpu().numpy())
    yaw_ee = math.atan2(R[1, 0], R[0, 0])
    c = p_tcp.clone()
    place(ped, torch.stack([c[:, 0], c[:, 1], c[:, 2] - 0.025 - 0.3], dim=1))
    place(cube, c, yaw=yaw_ee)
    bot.step(int(0.3 / DT))
    set_grip(0.0)
    bot.step(int(1.5 / DT))
    q = bot.q()[0]
    res["closed_on_cube"] = {"r1": float(q[r1]), "r2": float(q[r2]), "l1": float(q[l1]), "l2": float(q[l2])}

    def rel_cube():
        p, _ = tcp_now()
        return (T(cube.data.root_link_pos_w) - bot.origins) - p, p

    rel0, tcp0 = rel_cube()
    cube_z0 = float((T(cube.data.root_link_pos_w) - bot.origins)[0, 2])
    move_arm(q_lift, 2.0, settle=0.0)
    place(ped, torch.tensor([[3.0, 1.0, 0.3]], device=DEV).repeat(bot.n, 1))  # remove support
    rel_lift, _ = rel_cube()
    hold = int(10.0 / DT)
    rec = Rec()
    for k in range(hold):
        bot.step(1)
        if k % max(1, args.dt // 20) == 0:
            rr, pt = rel_cube()
            rec.add(t=k * DT, rel=rr[0], z=(T(cube.data.root_link_pos_w) - bot.origins)[0, 2])
    rel = rec.np("rel")
    slip = np.linalg.norm(rel - rel[0:1], axis=-1)
    res["lift"] = {"cube_z_before_m": cube_z0,
                   "cube_z_after_lift_m": float(rec.np("z")[0]), "cube_z_end_m": float(rec.np("z")[-1]),
                   "cube_rise_m": float(rec.np("z")[-1] - cube_z0),
                   "slip_during_lift_m": float(np.linalg.norm((rel_lift - rel0)[0].cpu().numpy())),
                   "slip_hold_10s_m": float(slip.max()),
                   "dropped": bool(rec.np("z")[-1] < cube_z0 - 0.05)}
    plot(OUT / f"{RUN}_slip.png", rec.np("t"), {"slip [m]": slip}, "m", f"{RUN}: cube slip relative to TCP (hold)")
    crit = {"symmetric(<=2mm)": res["max_asym_m"] <= 0.002,
            "lift>=0.18m": res["lift"]["cube_rise_m"] >= 0.18,
            "slip<=1cm": (not res["lift"]["dropped"]) and res["lift"]["slip_hold_10s_m"] <= 0.01}
    return {**res, "criteria": crit, "pass": all(crit.values())}


# ---- camera ------------------------------------------------------------------------------------------
def test_camera(bot):
    from PIL import Image

    cam_json = json.loads((GEN / "camera_frames.json").read_text())
    img_dir = out_dir("images")
    # look slightly down so the head camera sees the props
    q = T(bot.r.data.default_joint_pos).clone()
    q[:, bot.idx["head_1"]] = math.radians(20)
    for s, sign in (("right", -1), ("left", 1)):  # bring wrists forward so the wrist cameras see props
        q[:, bot.idx[f"{s}_arm_1"]] = sign * math.radians(10)
        q[:, bot.idx[f"{s}_arm_3"]] = math.radians(-90)
    bot.write_state(q)
    bot.step(int(1.0 / DT))
    out = {}
    key_map = {"head": "first_person_camera", "left_wrist": "left_hand_camera", "right_wrist": "right_hand_camera"}
    parent = {"head": "link_head_2", "left_wrist": "ee_left", "right_wrist": "ee_right"}
    for k in ("head", "left_wrist", "right_wrist"):
        cam = bot.scene[f"cam_{k}"]
        rgb = T(cam.data.output["rgb"])[0].cpu().numpy()[..., :3].astype(np.uint8)
        dep = T(cam.data.output["distance_to_image_plane"])[0].cpu().numpy().squeeze(-1)
        Image.fromarray(rgb).save(img_dir / f"t3f_{k}_rgb.png")
        np.save(img_dir / f"t3f_{k}_depth.npy", dep)
        finite = np.isfinite(dep) & (dep > 0)
        dvis = np.zeros_like(dep)
        if finite.any():
            d = np.clip(dep, 0, np.percentile(dep[finite], 99))
            dvis = (255 * (1 - d / max(1e-6, d.max()))).astype(np.uint8)
            dvis[~finite] = 0
        Image.fromarray(dvis.astype(np.uint8)).save(img_dir / f"t3f_{k}_depth.png")
        # orientation: camera world pose (OpenGL/USD convention) vs camera_frames.json relative to parent link
        jm = np.array(cam_json[key_map[k]]["matrix"])  # Gf row-vector convention
        R_rel, t_rel = jm[:3, :3].T, jm[3, :3]
        pq = bot.body_quat(parent[k])[0].cpu().numpy()
        pp = bot.body_pos(parent[k])[0].cpu().numpy()
        R_exp = quat_to_mat(pq) @ R_rel
        p_exp = quat_to_mat(pq) @ t_rel + pp
        cq = wxyz(T(cam.data.quat_w_opengl)[0]).cpu().numpy()
        cp = T(cam.data.pos_w)[0].cpu().numpy() - bot.origins[0].cpu().numpy()
        R_act = quat_to_mat(cq)
        ang = math.degrees(math.acos(np.clip((np.trace(R_exp.T @ R_act) - 1) / 2, -1, 1)))
        K = T(cam.data.intrinsic_matrices)[0].cpu().numpy()
        out[k] = {"rgb_mean": float(rgb.mean()), "rgb_std": float(rgb.std()), "depth_valid_frac": float(finite.mean()),
                  "depth_range_m": [float(dep[finite].min()), float(dep[finite].max())] if finite.any() else None,
                  "orient_err_deg": ang, "pos_err_m": float(np.linalg.norm(cp - p_exp)),
                  "view_dir_world(-Z)": (-R_act[:, 2]).tolist(), "K": K.tolist(),
                  "files": [str(img_dir / f"t3f_{k}_{s}.png") for s in ("rgb", "depth")]}
        print(f"[camera] {k}: mean {rgb.mean():.1f} std {rgb.std():.1f} depth_valid {finite.mean():.2f} "
              f"orient_err {ang:.2f} deg pos_err {out[k]['pos_err_m']*1000:.1f} mm")
    obs = T(bot.scene["cam_observer"].data.output["rgb"])[0].cpu().numpy()[..., :3].astype(np.uint8)
    Image.fromarray(obs).save(img_dir / "t3f_observer_rgb.png")
    out_obs = str(img_dir / "t3f_observer_rgb.png")
    crit = {f"{k}_not_black": v["rgb_mean"] > 5 and v["rgb_std"] > 2 for k, v in out.items()}
    crit.update({f"{k}_orient<=1deg": v["orient_err_deg"] <= 1.0 for k, v in out.items()})
    crit.update({f"{k}_depth_valid": v["depth_valid_frac"] > 0.3 for k, v in out.items()})
    return {**out, "observer_image": out_obs, "criteria": crit, "pass": all(crit.values())}


# --------------------------------------------------------------------------------------------- main
def main():
    t_wall = time.time()
    sim, scene, n_mat = build_scene()
    bot = Bot(sim, scene)
    try:
        from isaaclab_physx.physics.physx_manager import PhysxManager

        phys_dev = PhysxManager.get_physics_sim_device()
    except Exception as e:  # noqa: BLE001
        phys_dev = f"unknown ({e})"
    meta = {
        "run": RUN, "args": vars(args), "physics_device": phys_dev, "dt": DT, "joint_names": bot.names,
        "body_names": bot.bodies, "root_body": bot.bodies[0], "is_fixed_base": bool(bot.r.is_fixed_base), "num_joints": bot.r.num_joints,
        "materials_bound": n_mat,
        "body_mass_kg": per_joint(bot.bodies, T(bot.r.data.body_mass)[0].cpu().numpy()),
        "total_mass_kg": float(T(bot.r.data.body_mass)[0].sum()),
        "body_pos_start": {n: (T(bot.r.data.body_link_pos_w)[0, i] - bot.origins[0]).tolist() for i, n in enumerate(bot.bodies)},
        "joint_limits_rad": {n: T(bot.r.data.joint_pos_limits)[0, i].tolist() for n, i in bot.idx.items()},
        "joint_effort_limits": per_joint(bot.names, T(bot.r.data.joint_effort_limits)[0].cpu().numpy()),
        "joint_vel_limits": per_joint(bot.names, T(bot.r.data.joint_vel_limits)[0].cpu().numpy()),
        "joint_stiffness": per_joint(bot.names, T(bot.r.data.joint_stiffness)[0].cpu().numpy()),
        "joint_damping": per_joint(bot.names, T(bot.r.data.joint_damping)[0].cpu().numpy()),
        "joint_armature": per_joint(bot.names, T(bot.r.data.joint_armature)[0].cpu().numpy()),
    }
    print(f"[T3] {RUN}: physics device {phys_dev}, {bot.r.num_joints} joints, fixed_base={bot.r.is_fixed_base}, "
          f"mass {meta['total_mass_kg']:.1f} kg")
    if args.damping_ratio and args.test != "idle":
        meta["damping_override"] = apply_damping_ratio(bot, args.damping_ratio)
    fn = {"idle": test_idle, "crouch": test_crouch, "sweep": test_sweep, "gripper": test_gripper,
          "drive": test_drive, "camera": test_camera}[args.test]
    try:
        result = fn(bot)
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        result = {"error": repr(e), "traceback": traceback.format_exc(), "pass": False}
    meta["wall_s"] = time.time() - t_wall
    log = kit_log_path()
    warns = []
    if log and log.exists():
        with open(log, errors="replace") as fh:
            if log == LOG0:
                fh.seek(LOG0_SIZE)
            warns = [ln.strip() for ln in fh if "[Warning]" in ln or "[Error]" in ln]
    meta["kit_warnings"] = warns[:300]
    meta["kit_warning_count"] = len(warns)
    save_json(OUT / f"{RUN}.json", {**meta, "result": result})
    print(f"[T3] {RUN}: pass={result.get('pass')} criteria={result.get('criteria')} -> {OUT / (RUN + '.json')}")


if __name__ == "__main__":
    main()
    app.close()
