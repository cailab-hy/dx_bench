"""Search a floor-pick posture: torso (torso_1, torso_2, torso_3) grid x right-arm kinematic IK so that the
fingertips are 1.2 cm above the floor, 15-50 cm ahead of the base front, gripper pointing down, while the
centre of mass stays >= 5 cm behind the front wheel axle and the torso gravity torques stay <= 80 % of effort.

    python tools/rby1_check/floor_posture_search.py [--aheads 0.15,0.20,0.25,0.30]

Kinematic only (fixed base, joints teleported; no stepping). Writes reports/rby1_check/t5/floor_posture.json
and dxbench/dxbench/embodiments/floor_posture.json (the chosen posture for oracles / teleop presets).
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--aheads", default="0.15,0.20,0.25,0.30")
ap.add_argument("--y", type=float, default=-0.20)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

from pathlib import Path  # noqa: E402

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from dxbench.embodiments.rby1_cfg import EFFORT, TORSO, _sel, make_rby1_cfg  # noqa: E402
from dxbench.utils.kin_ik import ArmIK, T  # noqa: E402

BASE_FRONT_X = 0.32
FRONT_AXLE_X = 0.228


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
    eff = {}
    import re

    for n in TORSO:
        eff[n] = next(v for k, v in _sel(EFFORT, TORSO).items() if re.fullmatch(k, n))
    q_home = T(r.data.default_joint_pos).clone()
    m = T(r.data.body_mass)[0]
    results = []
    grid = itertools.product((30, 45, 60, 75, 90), (-150, -130, -110, -90, -70), (0, 30, 60, 90))
    aheads = [float(a) for a in args.aheads.split(",")]
    for t1, t2, t3 in grid:
        q = q_home.clone()
        q[0, tid[1]], q[0, tid[2]], q[0, tid[3]] = math.radians(t1), math.radians(t2), math.radians(t3)
        ik.fk_set(q)
        for a in aheads:
            target = (BASE_FRONT_X + a, args.y, 0.031 + 0.012)
            qs, ep, er = ik.solve(target, torch.eye(3), q0=q)
            c = T(r.data.body_com_pos_w)[0]
            com_x = float((m[:, None] * c).sum(0)[0] / m.sum())
            g = T(r.data.gravity_compensation_forces)[0]
            g = g[g.shape[-1] - len(r.joint_names):]
            tq = {n: float(abs(g[i]) / eff[n]) for n, i in zip(TORSO, tid)}
            rec = {"torso_deg": (t1, t2, t3), "ahead_m": a, "ik_pos_err": ep, "ik_rot_err": er, "com_x": com_x,
                   "com_margin": FRONT_AXLE_X - com_x, "torso_torque_ratio_max": max(tq.values()),
                   "arm_q": qs[0, ik.arm].tolist()}
            rec["ok"] = ep < 0.01 and er < 0.05 and rec["com_margin"] >= 0.05 and rec["torso_torque_ratio_max"] <= 0.8
            results.append(rec)
    ok = [x for x in results if x["ok"]]
    best = max(ok, key=lambda x: (x["com_margin"], -x["torso_torque_ratio_max"])) if ok else None
    out = {"criteria": "IK<1cm/0.05rad, COM >= 5 cm behind front axle (x 0.228), torso gravity torque <= 80 %",
           "n": len(results), "n_ok": len(ok), "best": best,
           "ok_by_ahead": {str(a): sum(1 for x in ok if x["ahead_m"] == a) for a in aheads}, "results": results}
    od = Path(os.path.expanduser("~/dx_bench/reports/rby1_check/t5"))
    od.mkdir(parents=True, exist_ok=True)
    (od / "floor_posture.json").write_text(json.dumps(out, indent=1))
    if best:
        (Path(os.path.expanduser("~/dx_bench/dxbench/dxbench/embodiments")) / "floor_posture.json").write_text(
            json.dumps(best, indent=1))
    print(f"[floor_posture] ok {len(ok)}/{len(results)} best={json.dumps({k: best[k] for k in ('torso_deg', 'ahead_m', 'com_margin', 'torso_torque_ratio_max')}) if best else None}")


if __name__ == "__main__":
    main()
    app.close()
