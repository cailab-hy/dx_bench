"""Kinematic reach check of the right TCP (gripper pointing down) per torso preset over a grid of base-frame
targets - used to choose oracle placement / pick postures. Fixed base, joints teleported (no stepping).

    python tools/rby1_check/reach_check.py --xs 0.45,0.55,0.62,0.70 --zs 0.35,0.45,0.55,0.65 --y -0.20
Writes reports/rby1_check/t5/reach_check.json; prints a table (position error cm per preset, x, z).
"""
from __future__ import annotations

import argparse
import json
import math
import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--xs", default="0.45,0.55,0.62,0.70")
ap.add_argument("--zs", default="0.35,0.45,0.55,0.65,0.75")
ap.add_argument("--y", type=float, default=-0.20)
ap.add_argument("--out", default="~/dx_bench/reports/rby1_check/t5/reach_check.json")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from dxbench.embodiments.rby1_cfg import TORSO, TORSO_PRESETS, make_rby1_cfg  # noqa: E402
from dxbench.utils.kin_ik import ArmIK, T  # noqa: E402


def main():
    cfg = make_rby1_cfg(preset="tuned")
    cfg.spawn.articulation_props.fix_root_link = True

    @configclass
    class S(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
        robot = cfg

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args.device, dt=1 / 240))
    scene = InteractiveScene(S(num_envs=1, env_spacing=2.0))
    sim.reset()
    r = scene["robot"]
    ik = ArmIK(r, sim, scene, "right")
    tid = [r.joint_names.index(n) for n in TORSO]
    q_home = T(r.data.default_joint_pos).clone()
    res = {}
    for name, (t1, t2, t3) in TORSO_PRESETS.items():
        q = q_home.clone()
        q[0, tid[1]], q[0, tid[2]], q[0, tid[3]] = t1, t2, t3
        ik.fk_set(q)
        for x in [float(v) for v in args.xs.split(",")]:
            for z in [float(v) for v in args.zs.split(",")]:
                _, ep, er = ik.solve((x, args.y, z), torch.eye(3), q0=q)
                res[f"{name}|{x}|{z}"] = {"pos_err": float(ep), "rot_err": float(er)}
    out = os.path.expanduser(args.out)
    json.dump(res, open(out, "w"), indent=1)
    zs = [float(v) for v in args.zs.split(",")]
    for name in TORSO_PRESETS:
        print(f"[reach] {name:8s} z:" + " ".join(f"{z:5.2f}" for z in zs))
        for x in [float(v) for v in args.xs.split(",")]:
            print(f"[reach]   x={x:.2f}  " + " ".join(
                f"{100 * res[f'{name}|{x}|{z}']['pos_err']:5.1f}" + ("*" if res[f'{name}|{x}|{z}']['rot_err'] > 0.05 else " ")
                for z in zs))
    app.close()


if __name__ == "__main__":
    main()
