"""Stage 2 regression: the 0-stage T3 checks re-run through the Arena RB-Y1 embodiments (15 Hz actions).

    python tools/rby1_check/t5_embodiment.py --test idle    --external_environment_class_path dxbench.environments.rby1_envs:DxFlatEnvironment dx_rby1_flat
    python tools/rby1_check/t5_embodiment.py --test crouch  --external_environment_class_path ... dx_rby1_flat
    python tools/rby1_check/t5_embodiment.py --test drive   --external_environment_class_path ... dx_rby1_flat
    python tools/rby1_check/t5_embodiment.py --test gripper --external_environment_class_path ... dx_rby1_flat --embodiment rby1_ik --props grasp_cube
    python tools/rby1_check/t5_embodiment.py --test floor   --external_environment_class_path ... dx_rby1_flat --embodiment rby1_ik
    python tools/rby1_check/t5_embodiment.py --test bedroom_camera --enable_cameras \
        --external_environment_class_path dxbench.environments.rby1_envs:DxBedroomEnvironment dx_bedroom
Writes reports/rby1_check/t5/<test>.json (+ images for bedroom_camera).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

from dxbench.utils.arena_runtime import ArenaRuntime

def _t5_args(p):
    p.add_argument("--test", required=True, choices=["idle", "crouch", "drive", "gripper", "floor", "bedroom_camera",
                                                     "torso_ramp"])
    p.add_argument("--ramp_to", default="deep", help="torso_ramp: preset name or 'a,b,c' degrees for torso_1..3")
    p.add_argument("--ramp_s", type=float, default=4.0)
    p.add_argument("--floor_presets", default="2:crouch,3:deep", help="torso preset steps:name list for --test floor")
    p.add_argument("--floor_aheads", default="0.15,0.20,0.25,0.30")
    p.add_argument("--t5_tag", default="", help="suffix of the result JSON name")


rt = ArenaRuntime(add_args=_t5_args)
env = rt.make_env()
args = rt.args

import numpy as np  # noqa: E402
import torch  # noqa: E402

from dxbench.embodiments.rby1_cfg import ARM_L, ARM_R, EFFORT, READY_ARMS, TORSO, TORSO_PRESETS, _sel  # noqa: E402

OUT = Path(os.path.expanduser("~/dx_bench/reports/rby1_check/t5"))
OUT.mkdir(parents=True, exist_ok=True)
U = env.unwrapped
R = U.scene["robot"]
DEV = U.device
N = U.num_envs
HZ = 15.0
BASE_FRONT_X = 0.32  # base outline 0.64 m long (URDF mesh), centred on the base origin


def T(x):
    import warp as wp

    if hasattr(x, "torch"):
        return x.torch
    return wp.to_torch(x) if isinstance(x, wp.array) else x


def jidx(names):
    ids, _ = R.find_joints(names, preserve_order=True)
    return list(ids.tolist() if hasattr(ids, "tolist") else ids)


def body_pos(name):
    return T(R.data.body_link_pos_w)[:, R.body_names.index(name)] - U.scene.env_origins


def body_quat(name):  # xyzw
    return T(R.data.body_link_quat_w)[:, R.body_names.index(name)]


def qapply(q, v):  # xyzw
    xyz, w = q[:, :3], q[:, 3:4]
    t = 2 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def qmat(q):  # xyzw (4,) -> 3x3
    x, y, z, w = q.tolist()
    return torch.tensor([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], device=DEV)


def tcp(side="right"):
    q = body_quat(f"ee_{side}")
    return body_pos(f"ee_{side}") + qapply(q, torch.tensor([[0.0, 0.0, -0.1045]], device=DEV).expand(N, 3)), q


def tilt_deg():
    x, y, z, w = body_quat("base")[0].tolist()
    return math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y)))))


def yaw():
    x, y, z, w = body_quat("base")[0].tolist()
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


# --------------------------------------------------------------------------- joint embodiment helpers
ARMS = ARM_R + ARM_L
ARM_READY = torch.tensor([[READY_ARMS.get(n, 0.0) for n in ARMS]], device=DEV)


def joint_action(arms=None, grip=(1.0, 1.0), torso=None, base=(0.0, 0.0)):
    a = torch.zeros(N, 24, device=DEV)
    a[:, :14] = ARM_READY if arms is None else arms
    a[:, 14:16] = torch.tensor(grip, device=DEV)
    a[:, 16:22] = 0.0 if torso is None else torso
    a[:, 22:24] = torch.tensor(base, device=DEV)
    return a


def torso_vec(preset):
    t1, t2, t3 = TORSO_PRESETS[preset]
    return torch.tensor([[0.0, t1, t2, t3, 0.0, 0.0]], device=DEV)


def ik_action(dr=None, dl=None, grip=(1.0, 1.0), torso_step=0.0, base=(0.0, 0.0)):
    a = torch.zeros(N, 17, device=DEV)
    if dr is not None:
        a[:, 0:6] = dr
    if dl is not None:
        a[:, 6:12] = dl
    a[:, 12:14] = torch.tensor(grip, device=DEV)
    a[:, 14] = torso_step
    a[:, 15:17] = torch.tensor(base, device=DEV)
    return a


def rot_err(R_des, q):  # axis-angle error (world) from current ee quat to R_des
    Rc = qmat(q[0])
    Re = R_des @ Rc.T
    ang = math.acos(max(-1.0, min(1.0, (float(torch.trace(Re)) - 1) / 2)))
    if ang < 1e-6:
        return torch.zeros(3, device=DEV), 0.0
    ax = torch.tensor([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]], device=DEV)
    return ax / (2 * math.sin(ang)) * ang, ang


def servo(p_des, R_des, steps, grip=(1.0, 1.0), side="right", kp=0.5, max_dp=0.02, max_dr=0.08, torso_step=0.0):
    """Relative-IK P servo of one TCP (base frame == world frame here: base at the origin, yaw 0)."""
    err = (9.0, 9.0)
    for _ in range(steps):
        p, q = tcp(side)
        dp = (torch.tensor(p_des, device=DEV) - p[0]) * kp
        dp = dp.clamp(-max_dp, max_dp)
        er, ang = rot_err(R_des, q)
        dr = (er * kp).clamp(-max_dr, max_dr)
        d = torch.cat([dp, dr]).unsqueeze(0)
        U_step(ik_action(dr=d if side == "right" else None, dl=d if side == "left" else None, grip=grip,
                         torso_step=torso_step))
        err = (float((torch.tensor(p_des, device=DEV) - p[0]).norm()), ang)
    return err


def U_step(a):
    obs, *_ = env.step(a)
    return obs


def projected_ratio(names):
    f = T(R.root_view.get_dof_projected_joint_forces())
    ids = jidx(names)
    lim = _sel(EFFORT, names)
    import re

    out = {}
    for n, i in zip(names, ids):
        e = next(v for k, v in lim.items() if re.fullmatch(k, n))
        out[n] = float(f[0, i].abs() / e)
    return out


# --------------------------------------------------------------------------- tests
def test_idle():
    env.reset()
    ids = jidx(ARMS + TORSO)
    tgt = torch.cat([ARM_READY, torch.zeros(1, 6, device=DEV)], dim=1)
    p0 = body_pos("base")[0].clone()
    errs = []
    for k in range(150):
        U_step(joint_action())
        if k >= 15:
            errs.append(float((T(R.data.joint_pos)[0, ids] - tgt[0]).abs().max()))
    disp = float((body_pos("base")[0] - p0).norm())
    res = {"max_joint_err_rad": max(errs), "base_disp_m": disp, "tilt_deg": tilt_deg()}
    res["criteria"] = {"err<=0.02": res["max_joint_err_rad"] <= 0.02, "base<=0.01m": disp <= 0.01}
    return res


def test_crouch():
    env.reset()
    ids = jidx(ARMS + TORSO)
    tv = torso_vec("crouch")
    for k in range(60):
        a = 0.5 * (1 - math.cos(math.pi * (k + 1) / 60))
        U_step(joint_action(torso=tv * a))
    errs, ratios = [], {}
    for k in range(90):
        U_step(joint_action(torso=tv))
        tgt = torch.cat([ARM_READY, tv], dim=1)
        errs.append(float((T(R.data.joint_pos)[0, ids] - tgt[0]).abs().max()))
        for n, r in projected_ratio(TORSO + ARMS).items():
            ratios[n] = max(ratios.get(n, 0.0), r)
    worst = max(ratios.items(), key=lambda kv: kv[1])
    res = {"max_joint_err_rad": max(errs[15:]), "worst_torque": worst, "tilt_deg": tilt_deg(),
           "ee_right_z": float(body_pos("ee_right")[0, 2])}
    res["criteria"] = {"err<=0.05": res["max_joint_err_rad"] <= 0.05, "torque<=0.8": worst[1] <= 0.8,
                       "no_tip": res["tilt_deg"] < 5}
    return res


def test_torso_ramp():
    """Ramp the torso (cosine, --ramp_s) from upright to a preset and hold 4 s: trace tilt, torso tracking error,
    projected torque / effort and COM x each step (diagnoses tipping in the deep floor posture)."""
    env.reset()
    if "," in args.ramp_to:
        d = [math.radians(float(x)) for x in args.ramp_to.split(",")]
        tv = torch.zeros(1, 6, device=DEV)
        tv[0, 1:4] = torch.tensor(d, device=DEV)
    else:
        tv = torso_vec(args.ramp_to)
    ids = jidx(TORSO)
    n_ramp = int(args.ramp_s * HZ)
    trace = []
    for k in range(n_ramp + 60):
        a = 0.5 * (1 - math.cos(math.pi * min(1.0, (k + 1) / n_ramp)))
        U_step(joint_action(torso=tv * a))
        q = T(R.data.joint_pos)[0, ids]
        pr = projected_ratio(TORSO)
        trace.append({"k": k, "alpha": a, "tilt": tilt_deg(), "com_x": com_x(),
                      "err": [float(x) for x in (q - tv[0] * a).tolist()], "q": [float(x) for x in q.tolist()],
                      "torque_ratio": {n: round(v, 3) for n, v in pr.items()}})
    tip_k = next((t["k"] for t in trace if t["tilt"] > 5), None)
    worst = max(((n, v) for t in trace for n, v in t["torque_ratio"].items()), key=lambda kv: kv[1])
    res = {"ramp_to": args.ramp_to, "tip_step": tip_k, "tip_alpha": trace[tip_k]["alpha"] if tip_k is not None else None,
           "final_tilt": trace[-1]["tilt"], "max_err": max(max(abs(e) for e in t["err"]) for t in trace[: tip_k or None]),
           "worst_torque": worst, "trace": trace[::3]}
    res["criteria"] = {"no_tip": tip_k is None, "torque<=1": worst[1] <= 1.0}
    return res


def _idle_action(base=(0.0, 0.0)):
    """Hold posture + base command for either embodiment (24-D joint / 17-D ik)."""
    if U.action_manager.total_action_dim == 17:
        return ik_action(base=base)
    return joint_action(base=base)


def test_drive():
    env.reset()
    out = {}
    for name, (v, w), dur in (("straight", (0.3, 0.0), 3.0), ("spin", (0.0, 0.5), 3.0)):
        env.reset()
        for _ in range(8):
            U_step(_idle_action())
        p0, y0 = body_pos("base")[0].clone(), yaw()
        yu, yp = 0.0, y0
        n = int(dur * HZ)
        for k in range(n):
            U_step(_idle_action(base=(v, w)))
            y = yaw()
            yu += (y - yp + math.pi) % (2 * math.pi) - math.pi
            yp = y
        d = (body_pos("base")[0] - p0)[:2]
        fwd = float(d[0] * math.cos(y0) + d[1] * math.sin(y0))
        r = {"forward_m": fwd, "lateral_m": float(-d[0] * math.sin(y0) + d[1] * math.cos(y0)), "yaw_rad": yu}
        if name == "straight":
            r["expected_m"] = v * dur
            r["err"] = abs(fwd - v * dur) / (v * dur)
        else:
            r["expected_rad"] = w * dur
            r["err"] = abs(yu - w * dur) / (w * dur)
        out[name] = r
    out["criteria"] = {"straight<=10%": out["straight"]["err"] <= 0.10, "spin<=10%": out["spin"]["err"] <= 0.10}
    return out


R_DOWN = torch.eye(3)  # ee frame aligned with the base: approach axis = ee -z = world down


def test_gripper():
    env.reset()
    Rd = R_DOWN.to(DEV)
    cube = U.scene["grasp_cube"]
    c0 = (T(cube.data.root_link_pos_w)[0] - U.scene.env_origins[0]).clone()
    e1 = servo((float(c0[0]), float(c0[1]), float(c0[2]) + 0.12), Rd, 90)
    e2 = servo((float(c0[0]), float(c0[1]), float(c0[2])), Rd, 60)
    for _ in range(25):  # close
        U_step(ik_action(grip=(0.0, 1.0)))
    p, q = tcp()
    rel0 = (T(cube.data.root_link_pos_w)[0] - U.scene.env_origins[0]) - p[0]
    z_before = float(T(cube.data.root_link_pos_w)[0, 2])
    e3 = servo((float(c0[0]), float(c0[1]), float(c0[2]) + 0.20), Rd, 150, grip=(0.0, 1.0))
    q_arm = T(R.data.joint_pos)[0, jidx(ARM_R)].tolist()
    lim = T(R.data.joint_pos_limits)[0, jidx(ARM_R)].tolist()
    slip = 0.0
    for _ in range(150):
        U_step(ik_action(grip=(0.0, 1.0)))
        p, q = tcp()
        rel = (T(cube.data.root_link_pos_w)[0] - U.scene.env_origins[0]) - p[0]
        slip = max(slip, float((rel - rel0).norm()))
    rise = float(T(cube.data.root_link_pos_w)[0, 2]) - z_before
    res = {"reach_err_pre": e1, "reach_err_grasp": e2, "reach_err_lift": e3, "cube_rise_m": rise, "slip_m": slip,
           "right_arm_q_at_lift": q_arm, "right_arm_limits": lim}
    res["criteria"] = {"lift>=0.12": rise >= 0.12, "slip<=1cm": slip <= 0.01}  # D-17: 0.18 m exceeds the down-wrist workspace
    return res


FRONT_AXLE_X = 0.228  # drive wheels (forward tipping line, 0-stage URDF analysis)


def com_x():
    m = T(R.data.body_mass)[0]
    c = T(R.data.body_com_pos_w)[0] - U.scene.env_origins[0]
    return float((m[:, None] * c).sum(0)[0] / m.sum())


def test_floor():
    """Floor-pick posture: torso preset + right-arm IK to fingertips 1.2 cm above the floor, 15-50 cm ahead of the
    base front. Tries presets x distances (nearest first) and keeps the first that does not tip."""
    Rd = R_DOWN.to(DEV)
    tries = []
    presets = [(int(x.split(":")[0]), x.split(":")[1]) for x in args.floor_presets.split(",")]
    aheads = [float(x) for x in args.floor_aheads.split(",")]
    for preset_steps, preset in presets:
        for ahead in aheads:
            env.reset()
            for k in range(preset_steps):
                U_step(ik_action(torso_step=1.0))
                U_step(ik_action(torso_step=0.0))
            for _ in range(int(6 * HZ)):
                U_step(ik_action())
            phase_tilt = {"after_preset": tilt_deg(), "com_after_preset": com_x()}
            target = (BASE_FRONT_X + ahead, -0.20, 0.031 + 0.012)
            pre = servo((target[0], target[1], target[2] + 0.15), Rd, 90)
            phase_tilt.update(after_pre=tilt_deg(), com_after_pre=com_x())
            err = servo(target, Rd, 90)
            phase_tilt.update(after_servo=tilt_deg(), com_after_servo=com_x())
            for _ in range(30):
                U_step(ik_action())
            p, _ = tcp()
            t = {"preset": preset, "ahead_target_m": ahead, "phase_tilt": phase_tilt, "servo_err": err,
                 "fingertip_z_m": float(p[0, 2]) - 0.031,
                 "ahead_of_base_front_m": float(p[0, 0]) - BASE_FRONT_X, "tilt_deg": tilt_deg(),
                 "com_x_m": com_x(), "com_margin_to_front_axle_m": FRONT_AXLE_X - com_x()}
            t["ok"] = t["fingertip_z_m"] <= 0.02 and 0.15 <= t["ahead_of_base_front_m"] <= 0.50 and t["tilt_deg"] < 5
            tries.append(t)
            print("[floor]", json.dumps(t))
    best = next((t for t in tries if t["ok"]), min(tries, key=lambda t: t["tilt_deg"]))
    res = {"tries": tries, "best": best}
    res["criteria"] = {"fingertip<=0.02m": best["fingertip_z_m"] <= 0.02,
                       "15-50cm_ahead": 0.15 <= best["ahead_of_base_front_m"] <= 0.50, "no_tip": best["tilt_deg"] < 5}
    return res


def test_bedroom_camera():
    from PIL import Image
    from pxr import Usd, UsdGeom

    from dxbench.paths import CARE_FURNITURE

    env.reset()
    for _ in range(20):
        obs = U_step(joint_action())
    img = obs["camera_obs"]["head_rgb"][0].cpu().numpy()[..., :3].astype(np.uint8)
    Image.fromarray(img).save(OUT / "bedroom_head_rgb.png")
    for k in ("left_wrist_rgb", "right_wrist_rgb"):
        Image.fromarray(obs["camera_obs"][k][0].cpu().numpy()[..., :3].astype(np.uint8)).save(OUT / f"bedroom_{k}.png")
    # bed AABB centre projected into the head camera
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"], useExtentsHint=False)
    rng = cache.ComputeWorldBound(stage.GetPrimAtPath(f"{CARE_FURNITURE}/Bed")).ComputeAlignedRange()
    c = (np.array(rng.GetMin()) + np.array(rng.GetMax())) / 2
    cam = U.scene["head"]
    K = cam.data.intrinsic_matrices.torch[0].cpu().numpy() if hasattr(cam.data.intrinsic_matrices, "torch") else cam.data.intrinsic_matrices[0].cpu().numpy()
    pos = T(cam.data.pos_w)[0].cpu().numpy()
    x, y, z, w = T(cam.data.quat_w_ros)[0].cpu().numpy()
    Rm = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                   [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                   [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    pc = (c - pos) @ Rm
    uv = (K @ pc)[:2] / max(pc[2], 1e-6)
    vis = bool(pc[2] > 0 and 0 <= uv[0] < img.shape[1] and 0 <= uv[1] < img.shape[0])
    res = {"bed_centre_px": uv.tolist(), "bed_visible": vis, "rgb_mean": float(img.mean()),
           "image": str(OUT / "bedroom_head_rgb.png")}
    res["criteria"] = {"bed_in_head_camera": vis, "not_black": float(img.mean()) > 5}
    return res


t0 = time.time()
try:
    result = globals()[f"test_{args.test}"]()
    result["pass"] = all(result["criteria"].values())
except Exception as e:  # noqa: BLE001
    import traceback

    traceback.print_exc()
    result = {"error": repr(e), "pass": False}
result.update({"test": args.test, "embodiment": getattr(args, "embodiment", None), "wall_s": time.time() - t0,
               "physics_dt": U.physics_dt, "step_dt": U.step_dt})
(OUT / f"{args.test}{args.t5_tag}.json").write_text(json.dumps(result, indent=2, default=str))
print(f"[t5] {args.test}: pass={result['pass']} {json.dumps(result.get('criteria'))}")
sys.stdout.flush()
rt.close()
