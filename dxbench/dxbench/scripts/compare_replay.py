"""Compare a replayed raw episode with the original recording (numpy only).

    python dxbench/dxbench/scripts/compare_replay.py --orig outputs/pipeline_test/raw2/episode-xxx \
        --replay outputs/pipeline_test/replay_raw/episode-yyy [--out reports/stage4/replay_check.json]

Errors per step: base pose (xy m, yaw rad), arm / torso joint positions (rad), TCP position (m, base frame).
Pass (DECISIONS D-26): final base xy error <= 5 cm, >= 97 % of frames with every arm joint within 0.05 rad, and 99th
percentile TCP error <= 2 cm. The cloth solver is not bit-deterministic, so while the hand carries or presses the cloth
contact forces differ between runs and give short joint transients (2-5 frames, up to ~0.25 rad) that recover. The
strict all-frames check (max arm joint <= 0.05 rad, max TCP <= 5 cm) is still reported as `pass_strict`. The original is
recorded with rby1_ik (joint targets from DiffIK each step) and replayed as joint targets on rby1_joint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(ep):
    ep = Path(ep)
    return dict(np.load(ep / "trajectory.npz")), json.loads((ep / "episode.json").read_text())


def compare(orig, replay) -> dict:
    a, ma = load(orig)
    b, mb = load(replay)
    n = min(len(a["action"]), len(b["action"]))
    na, nb = ma["joint_names"], mb["joint_names"]
    arm = [f"{s}_arm_{i}" for s in ("right", "left") for i in range(7)]
    tor = [f"torso_{i}" for i in range(6)]
    ia = [na.index(j) for j in arm + tor]
    ib = [nb.index(j) for j in arm + tor]
    qa, qb = a["observation.joint_pos"][:n][:, ia], b["observation.joint_pos"][:n][:, ib]
    dq = np.abs(qa - qb)
    pa, pb = a["observation.base_pose"][:n], b["observation.base_pose"][:n]
    dxy = np.linalg.norm(pa[:, :2] - pb[:, :2], axis=1)
    dyaw = np.abs((pa[:, 2] - pb[:, 2] + np.pi) % (2 * np.pi) - np.pi)
    dtcp = np.linalg.norm(a["observation.tcp_right"][:n, :3] - b["observation.tcp_right"][:n, :3], axis=1)
    res = {"orig": str(orig), "replay": str(replay), "frames": int(n),
           "base_xy_err_final_m": float(dxy[-1]), "base_xy_err_max_m": float(dxy.max()),
           "base_yaw_err_max_rad": float(dyaw.max()),
           "arm_joint_err_max_rad": float(dq[:, :14].max()), "arm_joint_err_mean_rad": float(dq[:, :14].mean()),
           "torso_joint_err_max_rad": float(dq[:, 14:].max()),
           "tcp_right_err_max_m": float(dtcp.max()), "tcp_right_err_mean_m": float(dtcp.mean()),
           "tcp_right_err_p99_m": float(np.percentile(dtcp, 99)),
           "arm_frames_within_0.05rad": float((dq[:, :14].max(axis=1) <= 0.05).mean())}
    res["pass_strict"] = bool(res["base_xy_err_final_m"] <= 0.05 and res["arm_joint_err_max_rad"] <= 0.05
                              and res["tcp_right_err_max_m"] <= 0.05)
    res["pass"] = bool(res["base_xy_err_final_m"] <= 0.05 and res["arm_frames_within_0.05rad"] >= 0.97
                       and res["tcp_right_err_p99_m"] <= 0.02)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True)
    ap.add_argument("--replay", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    r = compare(a.orig, a.replay)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(r, indent=1))
    print(f"[replay_check] {json.dumps(r)}")


if __name__ == "__main__":
    main()
