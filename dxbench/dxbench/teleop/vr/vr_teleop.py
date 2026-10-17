# Ported from cailab-hy/lightwheel_rby1_transfer scripts/vr/vr_teleop.py @1178454 (same lab, internal use; DECISIONS D-2).
"""Quest 2 (WebXR) -> RB-Y1 two-arm teleoperation logic, independent of Isaac Sim.

Input is one WebXR report (what the headset page sends each frame):

    {"recv_t": <PC time.monotonic() when received>,
     "hands": {"left"|"right": {"tracked": bool, "hand_tracking": bool,
                                "position": [x, y, z], "orientation": [w, x, y, z],
                                "buttons": [values...], "axes": [values...]}},
     "head": {"position": [x, y, z], "orientation": [w, x, y, z]}}

Output per control step: an end-effector target (p, R) in the RB-Y1 base frame
(x forward, y left, z up; the Pinocchio frame the collector's IK uses) for every arm whose
grip is held, a gripper open/closed state per arm, and button commands.

  * Axes: WebXR is y up, -z forward -> robot p = M @ p_xr, R = M @ R_xr @ M.T.
  * Recenter: the operator's facing direction (head yaw at recenter) becomes robot +x.
  * Clutch: holding grip ties the arm to the controller *relative* to where both were when
    grip was pressed:  p* = p_E0 + s (p_C - p_C0),  R* = (R_C R_C0^T) R_E0.  Releasing holds
    the arm, so the hand can be repositioned and grip pressed again without a jump.
  * Safety: a stale (> stale_s), untracked, hand-tracked or NaN pose releases the clutch;
    grippers keep their last state, so a held object is not dropped on a tracking glitch.
"""

import math
from dataclasses import dataclass, field

import numpy as np

SIDES = ("right", "left")  # RB-Y1 dataset order (right first)
XR_TO_ROBOT = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
# xr-standard gamepad mapping of the Touch controllers (checked on Quest 2, 2026-10-01):
TRIGGER, SQUEEZE, THUMBSTICK, LOWER, UPPER = 0, 1, 3, 4, 5  # LOWER = A/X, UPPER = B/Y
STICK_X, STICK_Y = 2, 3  # axes; stick pushed up -> negative y
# (side, button, event) -> command name for the collector
BINDINGS = {
    ("right", LOWER, "press"): "start",  # A
    ("right", UPPER, "long"): "discard",  # B held
    ("left", LOWER, "press"): "pause",  # X
    ("left", UPPER, "long"): "recenter",  # Y held
}


# ---------------------------------------------------------------- rotations
def quat_to_mat(q):
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def mat_to_quat(R):
    """Rotation matrix -> unit quaternion [w, x, y, z] with w >= 0."""
    R = np.asarray(R, dtype=float)
    t = np.trace(R)
    if t > 0:
        s = 2 * math.sqrt(1 + t)
        q = [s / 4, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * math.sqrt(1 + R[i, i] - R[j, j] - R[k, k])
        q = [0.0] * 4
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = s / 4
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.array(q) / np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def heading(R):
    """Yaw of a frame's forward (+x) axis about +z; well defined unless it points straight up/down."""
    return math.atan2(R[1, 0], R[0, 0])


def slerp(q0, q1, t):
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    a = math.acos(d)
    return (math.sin((1 - t) * a) * q0 + math.sin(t * a) * q1) / math.sin(a)


def xr_to_robot(position, orientation):
    """WebXR pose (y up, -z forward) -> robot axes (x forward, y left, z up)."""
    return XR_TO_ROBOT @ np.asarray(position, float), XR_TO_ROBOT @ quat_to_mat(orientation) @ XR_TO_ROBOT.T


# ---------------------------------------------------------------- inputs
class Hysteresis:
    """Analog value -> bool: turns on above `on`, off below `off`."""

    def __init__(self, on=0.6, off=0.4):
        self.on, self.off, self.state = on, off, False

    def update(self, value):
        if self.state and value < self.off:
            self.state = False
        elif not self.state and value > self.on:
            self.state = True
        return self.state


class Button:
    """Sampled pressed state -> 'press' / 'release' edges and one 'long' event after long_s held."""

    def __init__(self, long_s=1.0):
        self.long_s, self.down, self.t0, self.long_fired = long_s, False, 0.0, False

    def update(self, pressed, now):
        events = []
        if pressed and not self.down:
            self.down, self.t0, self.long_fired = True, now, False
            events.append("press")
        elif not pressed and self.down:
            self.down = False
            events.append("release")
        if self.down and not self.long_fired and now - self.t0 >= self.long_s:
            self.long_fired = True
            events.append("long")
        return events


@dataclass
class Hand:
    ok: bool = False  # pose usable: tracked controller, finite, fresh
    reason: str = ""  # why not ok: "no data", "stale", "untracked", "hand tracking", "bad pose"
    p: np.ndarray = None  # robot axes, before recenter
    R: np.ndarray = None
    buttons: list = field(default_factory=list)
    axes: list = field(default_factory=list)

    def value(self, i):
        return float(self.buttons[i]) if i < len(self.buttons) and math.isfinite(self.buttons[i]) else 0.0


def parse_hand(d, fresh=True):
    if not d:
        return Hand(reason="no data")
    buttons, axes = list(d.get("buttons") or []), list(d.get("axes") or [])
    if not fresh:
        return Hand(reason="stale", buttons=[0.0] * len(buttons), axes=[0.0] * len(axes))
    pos, ori = d.get("position"), d.get("orientation")
    if d.get("hand_tracking"):
        return Hand(reason="hand tracking", buttons=buttons, axes=axes)
    if not d.get("tracked") or pos is None or ori is None:
        return Hand(reason="untracked", buttons=buttons, axes=axes)
    pos, ori = np.asarray(pos, float), np.asarray(ori, float)
    if not (pos.shape == (3,) and ori.shape == (4,) and np.all(np.isfinite(pos))
            and np.all(np.isfinite(ori)) and np.linalg.norm(ori) > 0.5):
        return Hand(reason="bad pose", buttons=buttons, axes=axes)
    p, R = xr_to_robot(pos, ori)
    return Hand(True, "", p, R, buttons, axes)


# ---------------------------------------------------------------- clutch
class ArmClutch:
    """Grip-held relative pose following for one arm (see module docstring)."""

    def __init__(self, scale=1.0, rotation=True, alpha=1.0, workspace=None):
        self.scale, self.rotation, self.alpha = scale, rotation, alpha
        self.workspace = None if workspace is None else (np.asarray(workspace[0], float), np.asarray(workspace[1], float))
        self.grip = Hysteresis()
        self.engaged = False
        self.target = None  # last commanded (p, R)

    def release(self):
        self.engaged = False

    def update(self, hand, squeeze, current):
        """hand: Hand in recentred robot axes; current: the arm's present target (p, R), used as E0.
        Returns the new target (p, R) while engaged, else None (hold the arm)."""
        held = self.grip.update(squeeze)
        if not (hand.ok and held):
            self.engaged = False
            return None
        if not self.engaged:  # grip pressed (or tracking back while held): anchor, no jump
            self.engaged = True
            self.fp, self.fq = hand.p.copy(), mat_to_quat(hand.R)
            self.C0 = (self.fp.copy(), quat_to_mat(self.fq))
            self.E0 = (np.asarray(current[0], float).copy(), np.asarray(current[1], float).copy())
        else:  # optional low-pass (alpha = 1: none)
            self.fp = self.fp + self.alpha * (hand.p - self.fp)
            self.fq = slerp(self.fq, mat_to_quat(hand.R), self.alpha)
        Rc = quat_to_mat(self.fq)
        p = self.E0[0] + self.scale * (self.fp - self.C0[0])
        R = Rc @ self.C0[1].T @ self.E0[1] if self.rotation else self.E0[1].copy()
        if self.workspace is not None:
            p = np.clip(p, *self.workspace)
        self.target = (p, R)
        return self.target


# ---------------------------------------------------------------- teleop
@dataclass
class Step:
    targets: dict  # side -> (p, R) or None (hold)
    gripper_closed: dict  # side -> bool
    events: list  # command names, e.g. ["start"]
    status: dict  # side -> {"tracked", "reason", "engaged", "trigger", "squeeze"}, plus "age"
    hands: dict = field(default_factory=dict)  # side -> Hand (recentred robot axes)

    def vector(self, current):
        """teleop.command row: per arm (right, left) [clutch, trigger, gripper_open, x, y, z, qw, qx, qy, qz]."""
        row = []
        for side in SIDES:
            p, R = self.targets[side] or current[side]
            st = self.status[side]
            row += [float(st["engaged"]), st["trigger"], float(not self.gripper_closed[side]), *p, *mat_to_quat(R)]
        return np.array(row, dtype=np.float32)

    def vr_input(self):
        """teleop.vr_input row: per controller (right, left) [tracked, x, y, z, qw, qx, qy, qz, trigger, squeeze],
        recentred robot axes (pose zero when untracked)."""
        row = []
        for side in SIDES:
            h = self.hands.get(side) or Hand()
            pose = [*h.p, *mat_to_quat(h.R)] if h.ok else [0.0] * 7
            row += [float(h.ok), *pose, h.value(TRIGGER), h.value(SQUEEZE)]
        return np.array(row, dtype=np.float32)


class VRTeleop:
    def __init__(self, scale=1.0, rotation=True, alpha=1.0, stale_s=0.2, long_s=1.0, workspace=None):
        workspace = workspace or {}
        self.arms = {s: ArmClutch(scale, rotation, alpha, workspace.get(s)) for s in SIDES}
        self.grippers = {s: Hysteresis(0.6, 0.4) for s in SIDES}
        self.buttons = {(s, b): Button(long_s) for s in SIDES for b in (LOWER, UPPER)}
        self.stale_s = stale_s
        self.yaw = 0.0  # operator heading in WebXR-derived robot axes; removed from every pose

    def reset(self):
        """New episode: release clutches and open both grippers."""
        for s in SIDES:
            self.arms[s].release()
            self.arms[s].grip.state = False
            self.grippers[s].state = False

    def recenter(self, head):
        """Make the operator's current facing direction robot +x (head yaw only, pitch/roll ignored)."""
        _, R = xr_to_robot(head["position"], head["orientation"])
        self.yaw = heading(R)
        for arm in self.arms.values():  # re-anchor on the next update
            arm.release()

    def update(self, report, now, current):
        """report: latest WebXR report (or None); now: time.monotonic(); current: side -> (p, R)."""
        age = math.inf if report is None else now - report.get("recv_t", -math.inf)
        fresh = age <= self.stale_s
        hands_in = (report or {}).get("hands", {})
        hands = {s: parse_hand(hands_in.get(s), fresh) for s in SIDES}
        events = []
        for (side, b), button in self.buttons.items():
            for ev in button.update(hands[side].value(b) > 0.5, now):
                name = BINDINGS.get((side, b, ev))
                if name:
                    events.append(name)
        head = (report or {}).get("head")
        if "recenter" in events and fresh and isinstance(head, dict) and head.get("orientation"):
            self.recenter(head)
        Rz = rot_z(-self.yaw)
        targets, closed, status = {}, {}, {"age": age}
        for side in SIDES:
            h = hands[side]
            if h.ok:
                h.p, h.R = Rz @ h.p, Rz @ h.R
            targets[side] = self.arms[side].update(h, h.value(SQUEEZE), current[side])
            if fresh and hands_in.get(side):  # stale input keeps the last gripper state
                self.grippers[side].update(h.value(TRIGGER))
            closed[side] = self.grippers[side].state
            status[side] = {
                "tracked": h.ok,
                "reason": h.reason,
                "engaged": self.arms[side].engaged,
                "trigger": h.value(TRIGGER),
                "squeeze": h.value(SQUEEZE),
            }
        return Step(targets, closed, events, status, hands)
