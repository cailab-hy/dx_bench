"""Teleop device logic without the simulator: keyboard keymap, rotation deltas, VR clutch through the real WebSocket."""
import math
import time

import numpy as np

from dxbench.teleop.devices import KeyboardTeleop, VRTeleopDevice, quat_xyzw_to_mat, rot_delta
from dxbench.teleop.vr.vr_teleop import mat_to_quat


def test_keyboard_keymap():
    k = KeyboardTeleop(fps=15)
    k.on_key("W", True)
    k.on_key("UP", True)
    a, ev = k.step()
    assert a[0] > 0 and a[6] == 0 and a[15] > 0 and a[12] == 1.0 and ev == []
    k.on_key("TAB", True); k.on_key("TAB", False)
    k.on_key("K", True); k.on_key("K", False)
    a, _ = k.step()
    assert a[6] > 0 and a[0] == 0 and a[13] == 0.0 and a[12] == 1.0  # left arm active, left gripper closed
    k.on_key("J", True)
    a, _ = k.step()
    assert a[14] == 1.0
    a, _ = k.step()  # held key: edge only
    assert a[14] == 0.0
    k.on_key("B", True)
    _, ev = k.step()
    assert ev == ["start"]


def test_rot_delta():
    rng = np.random.default_rng(0)
    for _ in range(20):
        w = rng.normal(size=3) * 0.5
        th = np.linalg.norm(w)
        K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]]) / th
        Rw = np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K
        q = rng.normal(size=4)
        R0 = quat_xyzw_to_mat(q / np.linalg.norm(q))
        assert np.allclose(rot_delta(Rw @ R0, R0), w, atol=1e-6)


def test_vr_clutch_closed_loop():
    """Scripted operator -> WebSocket -> VRTeleop -> deltas; a fake TCP integrates the deltas (perfect IK)."""
    from dxbench.teleop.scripted_vr import ScriptedVROperator

    port = 18231
    dev = VRTeleopDevice(port)
    op = ScriptedVROperator(port)
    t0 = time.monotonic()
    while dev.server.status()["clients"] < 1 and time.monotonic() - t0 < 3:
        time.sleep(0.01)
    tcp = {"right": np.r_[0.45, -0.25, 0.95, 0, 0, 0, 1.0], "left": np.r_[0.45, 0.25, 0.95, 0, 0, 0, 1.0]}
    log, grips, events = [], [], []
    try:
        while not op.done:
            op.step()
            time.sleep(0.01)
            a, ev = dev.step(tcp)
            events += ev
            for s, off in (("right", 0), ("left", 6)):
                R = quat_xyzw_to_mat(tcp[s][3:])
                w = a[off + 3:off + 6]
                th = np.linalg.norm(w)
                if th > 1e-9:
                    K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]]) / th
                    R = (np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K) @ R
                q = mat_to_quat(R)  # wxyz
                tcp[s] = np.r_[tcp[s][:3] + a[off:off + 3], q[1], q[2], q[3], q[0]]
            log.append(tcp["right"][:3].copy())
            grips.append(a[12])
            time.sleep(0.07)  # wall time for the long Y press
    finally:
        op.close()
        dev.server.close()
    log = np.asarray(log)
    moved = log[op.marks["end"]] - log[op.marks["grip1"]]
    assert np.linalg.norm(moved - op.expected_offset) < 0.01, moved
    assert "start" in events and "recenter" in events
    assert min(grips) == 0.0 and grips[-1] == 1.0


def test_base_stick_shaping():
    """VR driving (BaseStick): no jump at the dead zone, expo, axis snap, acceleration limits, slow / fast, clutch lock."""
    from dxbench.teleop.devices import BaseStick

    b = BaseStick()
    assert b.shape(0.19, 0.0) == (0.0, 0.0)
    f, t = b.shape(0.21, 0.0)
    assert 0.0 < f < 0.01 and t == 0.0                      # starts from 0 at the dead-zone edge
    assert abs(b.shape(1.0, 0.0)[0] - 1.0) < 1e-9           # full deflection = full speed
    assert b.shape(0.5, 0.0)[0] < 0.5 * 0.7                 # expo: half deflection well under half speed
    f, t = b.shape(1.0, 0.2)                                # 11 deg off straight: snapped, no turn
    assert t == 0.0 and abs(f - 1.0) < 1e-9
    f, t = b.shape(0.2, -1.0)                               # 11 deg off pure turn: no speed
    assert f == 0.0 and t < -0.9
    f, t = b.shape(0.7, 0.7)                                # diagonal: both
    assert f > 0.1 and t > 0.1
    # acceleration: 0.5 m/s2 at 15 Hz -> 1/30 m/s per step, 9 steps to 0.3 m/s
    vs = [b.step(1.0, 0.0)[0] for _ in range(12)]
    assert abs(vs[0] - 0.5 / 15) < 1e-9 and abs(vs[8] - 0.3) < 1e-9 and abs(vs[-1] - 0.3) < 1e-9
    v, w = b.step(1.0, 0.0, locked=True)                    # grip held: decelerates to 0
    assert v < 0.3
    for _ in range(20):
        v, w = b.step(1.0, 0.0, locked=True)
    assert v == 0.0
    b.step(0.0, 0.0, click=True)                            # click: slow mode, one toggle per press
    b.step(0.0, 0.0, click=True)
    assert b.mode == "slow"
    for _ in range(30):
        v, _ = b.step(1.0, 0.0)
    assert abs(v - 0.10) < 1e-9
    b.step(0.0, 0.0, click=False)
    b.step(0.0, 0.0, click=True)
    assert b.mode == "fast"
    b.reset()
    assert (b.v, b.w) == (0.0, 0.0)


def test_vr_sticks_through_websocket():
    """Right stick -> base (v, omega), left stick up -> one torso step, through the real WebSocket (scripted 'drive')."""
    from dxbench.teleop.scripted_vr import ScriptedVROperator

    port = 18233
    dev = VRTeleopDevice(port)
    op = ScriptedVROperator(port, "drive")
    t0 = time.monotonic()
    while dev.server.status()["clients"] < 1 and time.monotonic() - t0 < 3:
        time.sleep(0.01)
    tcp = {"right": np.r_[0.45, -0.25, 0.95, 0, 0, 0, 1.0], "left": np.r_[0.45, 0.25, 0.95, 0, 0, 0, 1.0]}
    acts = []
    try:
        while not op.done:
            op.step()
            time.sleep(0.01)
            a, _ = dev.step(tcp)
            acts.append(a.copy())
            time.sleep(0.05)
    finally:
        op.close()
        dev.server.close()
    A = np.asarray(acts)
    m = op.marks
    assert A[m["drive0"]:m["turn0"], 15].max() > 0.29 and abs(A[m["drive0"]:m["turn0"], 16]).max() < 1e-9
    assert A[m["turn0"]:m["slow0"], 16].min() < -0.59 and abs(A[m["turn0"] + 20:m["slow0"] - 12, 15]).max() < 1e-9
    assert 0.09 < A[m["slow0"]:m["lock0"], 15].max() < 0.11
    assert A[m["lock0"] + 15:m["torso0"], 15].max() < 1e-9          # grip held: base stopped
    assert A[m["torso0"]:m["end"], 14].sum() == 1.0                  # one torso step per push
