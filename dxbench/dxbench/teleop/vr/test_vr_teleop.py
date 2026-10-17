# Ported from cailab-hy/lightwheel_rby1_transfer scripts/vr/test_vr_teleop.py @1178454 (same lab, internal use; DECISIONS D-2).
"""Unit tests for vr_teleop (no Isaac Sim):  conda run -n lerobot-arena python -m pytest -q scripts/vr"""

import math

import numpy as np
import pytest

from dxbench.teleop.vr.vr_teleop import (
    LOWER, SIDES, SQUEEZE, TRIGGER, UPPER, Button, Hysteresis, VRTeleop,
    mat_to_quat, quat_to_mat, rot_z, xr_to_robot,
)

I4 = [1.0, 0.0, 0.0, 0.0]
E_LEFT = (np.array([0.45, 0.25, 0.95]), np.eye(3))
E_RIGHT = (np.array([0.45, -0.25, 0.95]), rot_z(0.3))
CURRENT = {"left": E_LEFT, "right": E_RIGHT}


def q_axis(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    return [math.cos(angle / 2), *(math.sin(angle / 2) * axis)]


def hand(pos, ori=I4, squeeze=0.0, trigger=0.0, lower=0.0, upper=0.0, tracked=True):
    return {"tracked": tracked, "hand_tracking": False, "position": list(pos), "orientation": list(ori),
            "buttons": [trigger, squeeze, 0.0, 0.0, lower, upper], "axes": [0.0, 0.0, 0.0, 0.0]}


def report(t, left=None, right=None, head_ori=I4):
    hands = {}
    if left is not None:
        hands["left"] = left
    if right is not None:
        hands["right"] = right
    return {"recv_t": t, "hands": hands, "head": {"position": [0, 1.2, 0], "orientation": list(head_ori)}}


def run(tel, t, **kw):
    return tel.update(report(t, **kw), t, CURRENT)


# ---------------------------------------------------------------- axes
@pytest.mark.parametrize("xr, robot", [
    ([0, 0, -0.1], [0.1, 0, 0]),   # forward
    ([-0.1, 0, 0], [0, 0.1, 0]),   # left
    ([0, 0.1, 0], [0, 0, 0.1]),    # up
])
def test_axes(xr, robot):
    p, _ = xr_to_robot(xr, I4)
    np.testing.assert_allclose(p, robot, atol=1e-12)


def test_rotation_axes():
    # turning the controller left (about WebXR +y) is a positive yaw about robot +z
    _, R = xr_to_robot([0, 0, 0], q_axis([0, 1, 0], 0.4))
    np.testing.assert_allclose(R, rot_z(0.4), atol=1e-12)
    # tilting the controller's front up (about WebXR +x) is a negative pitch about robot +y
    _, R = xr_to_robot([0, 0, 0], q_axis([1, 0, 0], 0.4))
    np.testing.assert_allclose(R @ [1, 0, 0], [math.cos(0.4), 0, math.sin(0.4)], atol=1e-12)


def test_quaternion_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(200):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        q = q if q[0] >= 0 else -q
        np.testing.assert_allclose(mat_to_quat(quat_to_mat(q)), q, atol=1e-9)
    for angle in (math.pi, -math.pi + 1e-9):  # near-180 deg branch
        R = quat_to_mat(q_axis([0.3, -0.5, 0.8], angle))
        np.testing.assert_allclose(quat_to_mat(mat_to_quat(R)), R, atol=1e-9)


# ---------------------------------------------------------------- clutch
def test_no_motion_without_grip():
    tel = VRTeleop()
    s = run(tel, 0.0, left=hand([0, 1, 0]), right=hand([0.2, 1, 0]))
    assert s.targets == {"right": None, "left": None}
    s = run(tel, 0.1, left=hand([0, 1, -0.3]), right=hand([0.2, 1, -0.3]))
    assert s.targets == {"right": None, "left": None}


def test_engage_has_no_jump_and_follows():
    tel = VRTeleop()
    p, R = run(tel, 0.0, left=hand([0.3, 0.9, -0.2], q_axis([0, 1, 0], 1.0), squeeze=1)).targets["left"]
    np.testing.assert_allclose(p, E_LEFT[0], atol=1e-12)  # controller far away, still no jump
    np.testing.assert_allclose(R, E_LEFT[1], atol=1e-12)
    # forward 0.1, left 0.05, up 0.02 (WebXR) -> robot +x, +y, +z
    p, R = run(tel, 0.05, left=hand([0.25, 0.92, -0.3], q_axis([0, 1, 0], 1.0), squeeze=1)).targets["left"]
    np.testing.assert_allclose(p, E_LEFT[0] + [0.1, 0.05, 0.02], atol=1e-12)
    np.testing.assert_allclose(R, E_LEFT[1], atol=1e-12)


def test_rotation_delta_is_applied_in_robot_frame():
    tel = VRTeleop()
    c0 = q_axis([1, 1, 0], 0.7)  # arbitrary controller orientation at engage
    run(tel, 0.0, right=hand([0, 1, 0], c0, squeeze=1))
    # rotate the controller 0.5 rad about world up (WebXR +y) -> target turns 0.5 rad about robot +z
    c1 = (quat_to_mat(q_axis([0, 1, 0], 0.5)) @ quat_to_mat(c0))
    _, R = run(tel, 0.05, right=hand([0, 1, 0], mat_to_quat(c1), squeeze=1)).targets["right"]
    np.testing.assert_allclose(R, rot_z(0.5) @ E_RIGHT[1], atol=1e-9)


def test_regrip_is_continuous():
    tel = VRTeleop()
    run(tel, 0.0, left=hand([0, 1, 0], squeeze=1))
    p1, _ = run(tel, 0.05, left=hand([0, 1, -0.1], squeeze=1)).targets["left"]
    assert run(tel, 0.10, left=hand([0, 1, -0.1], squeeze=0)).targets["left"] is None  # release
    # move the hand back without grip: arm holds
    assert run(tel, 0.15, left=hand([0, 1, 0.2], squeeze=0)).targets["left"] is None
    # collector's current target is now p1; grabbing again continues from there
    current = {"left": (p1, np.eye(3)), "right": E_RIGHT}
    p, _ = tel.update(report(0.20, left=hand([0, 1, 0.2], squeeze=1)), 0.20, current).targets["left"]
    np.testing.assert_allclose(p, p1, atol=1e-12)
    p, _ = tel.update(report(0.25, left=hand([0, 1, 0.1], squeeze=1)), 0.25, current).targets["left"]
    np.testing.assert_allclose(p, p1 + [0.1, 0, 0], atol=1e-12)


def test_arms_are_independent():
    tel = VRTeleop()
    run(tel, 0.0, left=hand([0, 1, 0], squeeze=1), right=hand([0.2, 1, 0], squeeze=0))
    s = run(tel, 0.05, left=hand([0, 1.1, 0], squeeze=1), right=hand([0.2, 1.1, 0], squeeze=0))
    np.testing.assert_allclose(s.targets["left"][0], E_LEFT[0] + [0, 0, 0.1], atol=1e-12)
    assert s.targets["right"] is None


def test_grip_hysteresis():
    tel = VRTeleop()
    assert run(tel, 0.0, left=hand([0, 1, 0], squeeze=0.55)).targets["left"] is None
    assert run(tel, 0.05, left=hand([0, 1, 0], squeeze=0.7)).targets["left"] is not None
    assert run(tel, 0.10, left=hand([0, 1, 0], squeeze=0.45)).targets["left"] is not None
    assert run(tel, 0.15, left=hand([0, 1, 0], squeeze=0.3)).targets["left"] is None


def test_motion_scale_and_position_only():
    tel = VRTeleop(scale=0.5, rotation=False)
    run(tel, 0.0, left=hand([0, 1, 0], squeeze=1))
    p, R = run(tel, 0.05, left=hand([0, 1, -0.2], q_axis([0, 1, 0], 1.0), squeeze=1)).targets["left"]
    np.testing.assert_allclose(p, E_LEFT[0] + [0.1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(R, E_LEFT[1], atol=1e-12)


def test_workspace_clamp():
    tel = VRTeleop(workspace={"left": ([0.2, 0.0, 0.8], [0.6, 0.5, 1.2])})
    run(tel, 0.0, left=hand([0, 1, 0], squeeze=1))
    p, _ = run(tel, 0.05, left=hand([0, 1, -1.0], squeeze=1)).targets["left"]
    np.testing.assert_allclose(p, [0.6, 0.25, 0.95], atol=1e-12)


def test_filter_reaches_target():
    tel = VRTeleop(alpha=0.5)
    run(tel, 0.0, left=hand([0, 1, 0], squeeze=1))
    p, _ = run(tel, 0.05, left=hand([0, 1, -0.1], squeeze=1)).targets["left"]
    np.testing.assert_allclose(p, E_LEFT[0] + [0.05, 0, 0], atol=1e-12)
    for i in range(40):
        p, _ = run(tel, 0.1 + 0.01 * i, left=hand([0, 1, -0.1], squeeze=1)).targets["left"]
    np.testing.assert_allclose(p, E_LEFT[0] + [0.1, 0, 0], atol=1e-9)


# ---------------------------------------------------------------- recenter
def test_recenter_maps_operator_forward_to_robot_x():
    yaw = math.radians(35)  # operator sits turned 35 deg to the left of the WebXR origin
    head = q_axis([0, 1, 0], yaw)
    tel = VRTeleop()
    run(tel, 0.0, head_ori=head, left=hand([0, 1, 0], upper=1))
    s = run(tel, 1.01, head_ori=head, left=hand([0, 1, 0], upper=1))
    assert s.events == ["recenter"]
    assert tel.yaw == pytest.approx(yaw)
    fwd_xr = np.array([-math.sin(yaw), 0, -math.cos(yaw)])  # the operator's forward in WebXR
    run(tel, 1.05, head_ori=head, left=hand([0, 1, 0], squeeze=1))
    p, _ = run(tel, 1.10, head_ori=head, left=hand(np.array([0, 1, 0]) + 0.1 * fwd_xr, squeeze=1)).targets["left"]
    np.testing.assert_allclose(p, E_LEFT[0] + [0.1, 0, 0], atol=1e-12)


def test_recenter_ignores_head_pitch():
    tel = VRTeleop()
    yaw, pitch = 0.6, -0.5  # looking down at the desk
    head = mat_to_quat(quat_to_mat(q_axis([0, 1, 0], yaw)) @ quat_to_mat(q_axis([1, 0, 0], pitch)))
    tel.recenter({"position": [0, 1.2, 0], "orientation": list(head)})
    assert tel.yaw == pytest.approx(yaw)


def test_recenter_while_engaged_reanchors_without_jump():
    tel = VRTeleop()
    run(tel, 0.0, left=hand([0, 1, 0], squeeze=1))
    p1, _ = run(tel, 0.05, left=hand([0, 1, -0.1], squeeze=1)).targets["left"]
    tel.recenter({"position": [0, 1.2, 0], "orientation": q_axis([0, 1, 0], 0.8)})
    current = {"left": (p1, np.eye(3)), "right": E_RIGHT}
    p, _ = tel.update(report(0.10, left=hand([0, 1, -0.1], squeeze=1)), 0.10, current).targets["left"]
    np.testing.assert_allclose(p, p1, atol=1e-12)


# ---------------------------------------------------------------- safety
def test_stale_input_releases_and_keeps_gripper():
    tel = VRTeleop(stale_s=0.2)
    s = run(tel, 0.0, left=hand([0, 1, 0], squeeze=1, trigger=1))
    assert s.gripper_closed["left"] and s.status["left"]["engaged"]
    s = tel.update(report(0.0, left=hand([0, 1, -0.3], squeeze=1, trigger=0)), 0.25, CURRENT)  # 250 ms old
    assert s.targets["left"] is None and not s.status["left"]["engaged"]
    assert s.gripper_closed["left"]  # held object is not dropped
    s = tel.update(None, 0.3, CURRENT)
    assert s.targets["left"] is None and s.gripper_closed["left"]


@pytest.mark.parametrize("bad", [
    dict(tracked=False),
    dict(pos=[float("nan"), 1, 0]),
    dict(hand_tracking=True),
])
def test_bad_pose_releases_and_resumes_without_jump(bad):
    tel = VRTeleop()
    run(tel, 0.0, left=hand([0, 1, 0], squeeze=1))
    p1, _ = run(tel, 0.05, left=hand([0, 1, -0.1], squeeze=1)).targets["left"]
    h = hand(bad.get("pos", [0, 1, -0.5]), squeeze=1, tracked=bad.get("tracked", True))
    h["hand_tracking"] = bad.get("hand_tracking", False)
    assert run(tel, 0.10, left=h).targets["left"] is None
    current = {"left": (p1, np.eye(3)), "right": E_RIGHT}
    # tracking returns somewhere else while grip is still held: re-anchor, no jump
    p, _ = tel.update(report(0.15, left=hand([0.3, 1.2, 0.4], squeeze=1)), 0.15, current).targets["left"]
    np.testing.assert_allclose(p, p1, atol=1e-12)


def test_trigger_hysteresis_gripper():
    tel = VRTeleop()
    states = [run(tel, 0.05 * i, right=hand([0, 1, 0], trigger=v)).gripper_closed["right"]
              for i, v in enumerate([0.0, 0.5, 0.7, 0.5, 0.45, 0.3, 0.5])]
    assert states == [False, False, True, True, True, False, False]


# ---------------------------------------------------------------- buttons
def test_button_edges_and_long_press():
    b = Button(long_s=1.0)
    assert b.update(True, 0.0) == ["press"]
    assert b.update(True, 0.5) == []
    assert b.update(True, 1.0) == ["long"]
    assert b.update(True, 2.0) == []  # once per hold
    assert b.update(False, 2.1) == ["release"]
    assert b.update(True, 2.2) == ["press"]
    assert b.update(False, 2.3) == ["release"]  # short tap: no long


def test_bindings():
    tel = VRTeleop()
    assert run(tel, 0.0, right=hand([0, 1, 0], lower=1)).events == ["start"]  # A
    assert run(tel, 0.1, right=hand([0, 1, 0], lower=1)).events == []
    assert run(tel, 0.2, left=hand([0, 1, 0], lower=1)).events == ["pause"]  # X
    run(tel, 0.3, right=hand([0, 1, 0], upper=1))  # B short: nothing
    assert run(tel, 0.5, right=hand([0, 1, 0], upper=0)).events == []
    run(tel, 0.6, right=hand([0, 1, 0], upper=1))  # B held 1 s -> discard
    assert run(tel, 1.6, right=hand([0, 1, 0], upper=1)).events == ["discard"]


def test_reset_opens_grippers_and_releases():
    tel = VRTeleop()
    run(tel, 0.0, left=hand([0, 1, 0], squeeze=1, trigger=1))
    tel.reset()
    s = run(tel, 0.05, left=hand([0, 1, -0.1], squeeze=0, trigger=0))
    assert not s.gripper_closed["left"] and s.targets["left"] is None


def test_command_vector():
    tel = VRTeleop()
    run(tel, 0.0, left=hand([0, 1, 0], squeeze=1))
    s = run(tel, 0.05, left=hand([0, 1, -0.1], squeeze=1, trigger=0.8))
    v = s.vector(CURRENT)
    assert v.shape == (20,) and v.dtype == np.float32
    right, left = v[:10], v[10:]
    np.testing.assert_allclose(right[:3], [0, 0, 1])  # not engaged, trigger 0, gripper open
    np.testing.assert_allclose(right[3:6], E_RIGHT[0], atol=1e-6)
    np.testing.assert_allclose(right[6:], mat_to_quat(E_RIGHT[1]), atol=1e-6)
    np.testing.assert_allclose(left[:3], [1, 0.8, 0], atol=1e-6)  # engaged, trigger 0.8, closed
    np.testing.assert_allclose(left[3:6], E_LEFT[0] + [0.1, 0, 0], atol=1e-6)
    assert SIDES == ("right", "left")
    assert (TRIGGER, SQUEEZE, LOWER, UPPER) == (0, 1, 4, 5)


def test_recenter_without_head_orientation_is_ignored():
    # a stale browser tab (older page) sent the head as a bare position list
    tel = VRTeleop()
    for t in (0.0, 1.01):
        r = report(t, left=hand([0, 1, 0], upper=1))
        r["head"] = [0, 1.2, 0]
        s = tel.update(r, t, CURRENT)
    assert s.events == ["recenter"] and tel.yaw == 0.0


def test_not_ok_reasons():
    tel = VRTeleop()
    h = hand([0, 1, 0]); h["hand_tracking"] = True
    s = run(tel, 0.0, left=h, right=hand([0, 1, 0], tracked=False))
    assert (s.status["left"]["reason"], s.status["right"]["reason"]) == ("hand tracking", "untracked")
    s = tel.update(report(0.0, left=hand([0, 1, 0])), 1.0, CURRENT)
    assert (s.status["left"]["reason"], s.status["right"]["reason"]) == ("stale", "no data")


def test_vr_input_vector():
    tel = VRTeleop()
    s = run(tel, 0.0, left=hand([0, 1, -0.2], squeeze=0.7, trigger=0.3))
    v = s.vr_input()
    assert v.shape == (20,)
    np.testing.assert_allclose(v[:10], 0)  # right controller absent
    np.testing.assert_allclose(v[10:14], [1, 0.2, 0, 1], atol=1e-6)
    np.testing.assert_allclose(v[18:], [0.3, 0.7], atol=1e-6)
