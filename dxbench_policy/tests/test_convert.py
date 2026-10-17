"""Synthetic raw episode -> LeRobot (rby1 + dx24) -> validate. Runs in dx_policy without the simulator.

    ~/miniforge3/envs/dx_policy/bin/python -m pytest dxbench_policy/tests -q
"""
import json

import numpy as np
import pytest
from PIL import Image

JOINTS = ([f"base_{i}" for i in range(2)] + [f"torso_{i}" for i in range(6)] + [f"right_arm_{i}" for i in range(7)]
          + [f"left_arm_{i}" for i in range(7)] + ["gripper_finger_l1", "gripper_finger_l2", "gripper_finger_r1",
                                                    "gripper_finger_r2", "head_0", "head_1"])


def make_raw(root, T=12, success=True, seed=0):
    rng = np.random.default_rng(seed)
    ep = root / f"episode-{seed:012d}"
    for c in ("head", "left_wrist", "right_wrist"):
        (ep / c).mkdir(parents=True)
        for t in range(T):
            img = np.zeros((480, 640, 3), np.uint8)
            img[:, :, 0] = (t * 20) % 255
            img[100:200, 100 + 10 * t:200 + 10 * t] = (30, 200, 90)
            Image.fromarray(img).save(ep / c / f"{t:06d}.png")
    q = rng.normal(size=(T, len(JOINTS))).astype(np.float32) * 0.3
    q[:, JOINTS.index("gripper_finger_r1")] = -0.04
    q[:, JOINTS.index("gripper_finger_r2")] = 0.04
    a = {"observation.joint_pos": q, "observation.joint_vel": q * 0,
         "observation.torso": q[:, 2:8], "observation.base_vel": rng.normal(size=(T, 2)).astype(np.float32),
         "observation.base_pose": rng.normal(size=(T, 3)).astype(np.float32),
         "observation.tcp_right": rng.normal(size=(T, 7)).astype(np.float32),
         "observation.tcp_left": rng.normal(size=(T, 7)).astype(np.float32),
         "observation.state16": rng.normal(size=(T, 16)).astype(np.float32),
         "action": rng.normal(size=(T, 17)).astype(np.float32),
         "action.joint_target": q, "action.rby1_16": np.c_[q[:, 8:22], rng.uniform(size=(T, 2))].astype(np.float32),
         "action.torso": q[:, 2:8], "action.base": rng.normal(size=(T, 2)).astype(np.float32),
         "stage_id": np.arange(T) % 3, "intervention": np.zeros(T, int), "sim_time": np.arange(T) / 15.0,
         "subtask": np.array(["go"] * T)}
    np.savez_compressed(ep / "trajectory.npz", **a)
    (ep / "initial_state.json").write_text("{}")
    (ep / "episode.json").write_text(json.dumps({"task": "BED-03", "success": success, "joint_names": JOINTS,
                                                 "language": {"high": "put the clothes in the basket"}, "fps": 15}))
    return ep


@pytest.mark.parametrize("fmt", ["dx24", "rby1"])
def test_convert_validate(tmp_path, fmt):
    from dxbench_policy.convert.raw_to_lerobot import convert
    from dxbench_policy.convert.validate import validate

    raw = tmp_path / "raw"
    make_raw(raw, seed=1)
    make_raw(raw, seed=2)
    make_raw(raw, seed=3, success=False)
    out = tmp_path / f"ds_{fmt}"
    res = convert(raw, out, "local/test", fmt=fmt)
    assert res["episodes"] == 2
    again = convert(raw, out, "local/test", fmt=fmt)  # idempotent
    assert again["new"] == 0 and again["episodes"] == 2
    s = validate(out, [raw])
    assert s["ok"], s
    assert s["frames"] == 24
