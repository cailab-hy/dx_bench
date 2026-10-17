"""Scripted VR operator: plays the person holding the Quest controllers, through the real WebSocket (WSClient ->
VRServer -> VRTeleop clutch -> rby1_ik deltas), so the VR path is tested end to end without a headset.

Script (control steps at 15 Hz): idle -> Y held (recenter) -> A (start recording) -> right grip held:
hand +10 cm forward, +10 cm up -> trigger (close gripper) -> release trigger (open) -> grip released, hand moved
back -> grip again and -5 cm sideways (clutch re-anchor, no jump) -> done. `expected_offset` is the TCP displacement
the robot should show at the end relative to where the first grip started.
"""
from __future__ import annotations

import numpy as np

from dxbench.teleop.vr.vr_server import PAGE_VERSION, WSClient
from dxbench.teleop.vr.vr_teleop import XR_TO_ROBOT, mat_to_quat

LOWER, UPPER = 4, 5


def _ctrl(p, R=np.eye(3), squeeze=0.0, trigger=0.0, lower=0.0, upper=0.0):
    pxr, Rxr = XR_TO_ROBOT.T @ p, XR_TO_ROBOT.T @ R @ XR_TO_ROBOT
    return {"tracked": True, "hand_tracking": False, "position": pxr.tolist(), "orientation": mat_to_quat(Rxr).tolist(),
            "buttons": [trigger, squeeze, 0.0, 0.0, lower, upper, 0.0], "axes": [0.0] * 4}


class ScriptedVROperator:
    def __init__(self, port: int):
        self.client = WSClient(port)
        self.t = 0
        self.right = np.array([-0.1, -0.25, 0.95])
        self.left = np.array([-0.1, 0.25, 0.95])
        self.plan = self._plan()
        self.expected_offset = np.array([0.10, -0.05, 0.10])
        self.marks: dict[str, int] = {}

    def _plan(self):
        P = []
        P += [dict()] * 10                                            # idle, tracked
        P += [dict(upper_l=1.0)] * 25                                 # Y held >= 1 s wall -> recenter
        P += [dict()] * 3
        P += [dict(lower_r=1.0)] * 2 + [dict()] * 3                   # A -> start
        P += [dict(squeeze=1.0, mark="grip1")] + [dict(squeeze=1.0)] * 4
        P += [dict(squeeze=1.0, dp=(0.10 / 30, 0, 0))] * 30           # +10 cm x
        P += [dict(squeeze=1.0, dp=(0, 0, 0.10 / 30))] * 30           # +10 cm z
        P += [dict(squeeze=1.0, trigger=1.0)] * 15                    # close
        P += [dict(squeeze=1.0, trigger=0.0)] * 15                    # open
        P += [dict(dp=(-0.10 / 10, 0, 0))] * 10                       # released: hand moves back, robot holds
        P += [dict(squeeze=1.0, mark="grip2")] + [dict(squeeze=1.0)] * 4
        P += [dict(squeeze=1.0, dp=(0, -0.05 / 20, 0))] * 20          # -5 cm y
        P += [dict(squeeze=1.0)] * 20 + [dict(mark="end")] + [dict()] * 15
        return P

    @property
    def done(self) -> bool:
        return self.t >= len(self.plan)

    def step(self):
        """Send one WebXR report for this control step."""
        s = self.plan[min(self.t, len(self.plan) - 1)]
        if "mark" in s and s["mark"] not in self.marks:
            self.marks[s["mark"]] = self.t
        self.right = self.right + np.asarray(s.get("dp", (0, 0, 0)))
        hands = {"right": _ctrl(self.right, squeeze=s.get("squeeze", 0.0), trigger=s.get("trigger", 0.0),
                                lower=s.get("lower_r", 0.0)),
                 "left": _ctrl(self.left, upper=s.get("upper_l", 0.0))}
        self.client.send({"kind": "frame", "v": PAGE_VERSION, "hands": hands,
                          "head": {"position": [0.0, 1.6, 0.0], "orientation": [1.0, 0.0, 0.0, 0.0]}, "vseq": 0})
        self.t += 1

    def close(self):
        self.client.close()
