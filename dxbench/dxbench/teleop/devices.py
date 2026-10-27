"""Teleop devices -> `rby1_ik` 17-D actions (Isaac-free logic; Kit keyboard hookup in `attach_carb_keyboard`).

rby1_ik action: [right dpos 3, right drot 3, left dpos 3, left drot 3, gripper r, gripper l (1 open),
                 torso preset step (+1 deeper / -1 more upright, edge triggered), base v, base omega]
(base frame, relative DiffIK; docs/action_space.md).

Keyboard (lightwheel collect_keyboard.py keymap + base / torso keys):
  W/S x, A/D y, Q/E z, Z/X roll, T/G pitch, C/V yaw    active arm (TAB switches, starts with the right arm)
  K gripper toggle (active arm)    L stop the arm (clears held keys)
  arrows: UP/DOWN base forward/back, LEFT/RIGHT turn     U torso more upright, J torso deeper (one preset per press)
  H arms home (joint-space back to READY)
  B start recording, ENTER save (only if the task succeeded), BACKSPACE discard + reset, R reset, P pause, ESC quit
VR (Quest WebXR, lightwheel vr_server / vr_teleop): grip = clutch, trigger = gripper, A start, B (hold) discard,
  X pause, Y (hold) recenter; right thumbstick = base (v, omega; `BaseStick`: dead zone, expo, axis snap, acceleration
  limits, click = slow / fast, locked while a grip is held), left thumbstick up/down = torso preset step.
"""
from __future__ import annotations

import math
import time

import numpy as np

LIN_SPEED = 0.25    # m/s TCP (keyboard)
ROT_SPEED = 0.8     # rad/s TCP (keyboard)
BASE_V = 0.3        # m/s (keyboard; VR fast mode)
BASE_W = 0.6        # rad/s
FPS = 15
MAX_DP = 0.03       # m per step (VR tracking clip)
MAX_DR = 0.12       # rad per step

ARM_KEYS = {"W": (0, +1), "S": (0, -1), "A": (1, +1), "D": (1, -1), "Q": (2, +1), "E": (2, -1),
            "Z": (3, +1), "X": (3, -1), "T": (4, +1), "G": (4, -1), "C": (5, +1), "V": (5, -1)}
EVENT_KEYS = {"B": "start", "ENTER": "save", "BACKSPACE": "discard", "R": "reset", "P": "pause", "ESCAPE": "quit",
              "H": "home"}


class KeyboardTeleop:
    def __init__(self, fps: int = FPS):
        self.dt = 1.0 / fps
        self.held: set[str] = set()
        self.pressed: list[str] = []      # edge events since the last step
        self.arm = "right"
        self.grip = {"right": 1.0, "left": 1.0}

    def on_key(self, name: str, down: bool):
        if down:
            if name not in self.held:
                self.pressed.append(name)
            self.held.add(name)
        else:
            self.held.discard(name)

    def reset(self):
        self.held.clear()
        self.pressed.clear()
        self.grip = {"right": 1.0, "left": 1.0}

    def step(self) -> tuple[np.ndarray, list[str]]:
        a = np.zeros(17, dtype=np.float32)
        events = []
        for k in self.pressed:
            if k == "TAB":
                self.arm = "left" if self.arm == "right" else "right"
            elif k == "K":
                self.grip[self.arm] = 0.0 if self.grip[self.arm] > 0.5 else 1.0
            elif k == "L":
                self.held.clear()
            elif k == "U":
                a[14] = -1.0
            elif k == "J":
                a[14] = +1.0
            elif k in EVENT_KEYS:
                events.append(EVENT_KEYS[k])
        self.pressed.clear()
        off = 0 if self.arm == "right" else 6
        for k, (i, sgn) in ARM_KEYS.items():
            if k in self.held:
                a[off + i] += sgn * (LIN_SPEED if i < 3 else ROT_SPEED) * self.dt
        a[12], a[13] = self.grip["right"], self.grip["left"]
        a[15] = BASE_V * (("UP" in self.held) - ("DOWN" in self.held))
        a[16] = BASE_W * (("LEFT" in self.held) - ("RIGHT" in self.held))
        return a, events


def attach_carb_keyboard(teleop: KeyboardTeleop):
    """Subscribe the Kit keyboard (needs a GUI app, not headless). Returns the subscription handle."""
    import carb.input
    import omni.appwindow

    inp = carb.input.acquire_input_interface()
    kb = omni.appwindow.get_default_app_window().get_keyboard()

    def on_event(e, *args):
        name = e.input.name if hasattr(e.input, "name") else str(e.input)
        if e.type == carb.input.KeyboardEventType.KEY_PRESS:
            teleop.on_key(name, True)
        elif e.type == carb.input.KeyboardEventType.KEY_RELEASE:
            teleop.on_key(name, False)
        return True

    return inp.subscribe_to_keyboard_events(kb, on_event)


# ---------------------------------------------------------------- VR
def rot_delta(R_des: np.ndarray, R_cur: np.ndarray) -> np.ndarray:
    """Axis-angle (base frame) rotating R_cur onto R_des: R_des = Exp(w) R_cur."""
    Rd = R_des @ R_cur.T
    c = np.clip((np.trace(Rd) - 1) / 2, -1.0, 1.0)
    ang = math.acos(c)
    if ang < 1e-6:
        return np.zeros(3)
    ax = np.array([Rd[2, 1] - Rd[1, 2], Rd[0, 2] - Rd[2, 0], Rd[1, 0] - Rd[0, 1]])
    n = np.linalg.norm(ax)
    if n < 1e-9:  # ~180 deg: any perpendicular axis
        return np.array([math.pi, 0.0, 0.0])
    return ax / n * ang


def quat_xyzw_to_mat(q) -> np.ndarray:
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class BaseStick:
    """Right thumbstick -> base (v, omega) for VR driving (2026-10-27, user: make driving with linear + angular speed
    easy). Up / down = forward / back, left / right = turn left / right, as before, plus:

    * radial dead zone with rescaling: the output starts at 0 at the dead-zone edge (it used to jump to 20 %);
    * expo curve per axis, out = (1 - expo) x + expo x^3: small deflections are slow for fine positioning;
    * axis snap: within `snap_deg` of straight up / down the turn is 0, within `snap_deg` of straight left / right the
      speed is 0, so a slightly crooked thumb still drives straight or turns in place; diagonals blend both;
    * slow / fast mode toggled by clicking the stick: slow for lining up at a garment or the basket;
    * acceleration limits, so starts and stops are smooth (and the recorded base commands too);
    * locked (target 0) while an arm clutch (grip) is held: a thumb brushing the stick during fine arm work does not
      move the robot. Holding a garment (trigger) with the grip released still drives.
    """

    MODES = {"slow": (0.10, 0.30), "fast": (BASE_V, BASE_W)}

    def __init__(self, fps: int = FPS, dead: float = 0.2, expo: float = 0.6, snap_deg: float = 15.0,
                 acc_v: float = 0.5, acc_w: float = 1.5, mode: str = "fast"):
        self.dt = 1.0 / fps
        self.dead, self.expo, self.snap = dead, expo, math.radians(snap_deg)
        self.acc_v, self.acc_w = acc_v, acc_w
        self.mode = mode
        self.v = self.w = 0.0
        self._click = False

    def reset(self):
        self.v = self.w = 0.0
        self._click = False

    def shape(self, fwd: float, turn: float) -> tuple[float, float]:
        """Stick (forward, turn-left) in [-1, 1] -> normalised (speed, turn) in [-1, 1]."""
        r = math.hypot(fwd, turn)
        if r <= self.dead:
            return 0.0, 0.0
        k = min(1.0, (r - self.dead) / (1.0 - self.dead)) / r
        f, t = fwd * k, turn * k
        ang = math.atan2(abs(t), abs(f))  # 0 = straight forward / back, pi / 2 = pure turn
        if ang < self.snap:
            t = 0.0
            f = math.copysign(min(1.0, (r - self.dead) / (1.0 - self.dead)), f)
        elif ang > math.pi / 2 - self.snap:
            f = 0.0
            t = math.copysign(min(1.0, (r - self.dead) / (1.0 - self.dead)), t)
        e = self.expo
        return (1 - e) * f + e * f ** 3, (1 - e) * t + e * t ** 3

    def step(self, fwd: float, turn: float, click: bool = False, locked: bool = False) -> tuple[float, float]:
        if click and not self._click:
            self.mode = "slow" if self.mode == "fast" else "fast"
        self._click = click
        vmax, wmax = self.MODES[self.mode]
        f, t = (0.0, 0.0) if locked else self.shape(fwd, turn)
        tv, tw = vmax * f, wmax * t
        self.v += float(np.clip(tv - self.v, -self.acc_v * self.dt, self.acc_v * self.dt))
        self.w += float(np.clip(tw - self.w, -self.acc_w * self.dt, self.acc_w * self.dt))
        return self.v, self.w

    def status(self) -> str:
        return f"BASE {self.mode.upper()} v {self.v:+.2f} m/s  w {self.w:+.2f} rad/s"


class VRTeleopDevice:
    """WebXR reports (VRServer) -> clutch targets (VRTeleop) -> rby1_ik deltas towards the targets."""

    def __init__(self, port: int = 8012, server=None, scale: float = 1.0):
        from dxbench.teleop.vr.vr_server import VRServer
        from dxbench.teleop.vr.vr_teleop import VRTeleop

        self.server = server or VRServer(port, log=lambda m: None).start()
        self.teleop = VRTeleop(scale=scale)
        self.hold: dict[str, tuple[np.ndarray, np.ndarray] | None] = {"right": None, "left": None}
        self.torso_latch = 0.0
        self.last_step = None
        self.base = BaseStick()

    def reset(self):
        self.teleop.reset()
        self.hold = {"right": None, "left": None}
        self.base.reset()

    def step(self, tcp_b: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
        """tcp_b: side -> [x, y, z, qx, qy, qz, qw] measured TCP in the base frame."""
        from dxbench.teleop.vr.vr_teleop import STICK_X, STICK_Y, THUMBSTICK

        cur = {s: (np.asarray(tcp_b[s][:3], float), quat_xyzw_to_mat(tcp_b[s][3:7])) for s in ("right", "left")}
        # the clutch anchors on the commanded target (held pose) so releasing / re-grabbing does not jump
        anchor = {s: self.hold[s] or cur[s] for s in cur}
        report = self.server.take()
        st = self.teleop.update(report, time.monotonic(), anchor)
        self.last_step = st
        a = np.zeros(17, dtype=np.float32)
        for side, off in (("right", 0), ("left", 6)):
            tgt = st.targets[side]
            if tgt is not None:
                self.hold[side] = (np.asarray(tgt[0], float), np.asarray(tgt[1], float))
            if self.hold[side] is None:
                continue
            p, R = self.hold[side]
            a[off:off + 3] = np.clip(p - cur[side][0], -MAX_DP, MAX_DP)
            dr = rot_delta(R, cur[side][1])
            n = np.linalg.norm(dr)
            a[off + 3:off + 6] = dr * min(1.0, MAX_DR / n) if n > 0 else 0.0
        a[12] = 0.0 if st.gripper_closed["right"] else 1.0
        a[13] = 0.0 if st.gripper_closed["left"] else 1.0
        hands = (report or {}).get("hands", {})
        ax_r = (hands.get("right") or {}).get("axes") or [0.0] * 4
        ax_l = (hands.get("left") or {}).get("axes") or [0.0] * 4
        dead = lambda v: 0.0 if abs(v) < 0.2 else float(v)  # noqa: E731
        bt_r = (hands.get("right") or {}).get("buttons") or []
        click = len(bt_r) > THUMBSTICK and bt_r[THUMBSTICK] > 0.5
        locked = any(t is not None for t in st.targets.values())  # an arm clutch is held
        a[15], a[16] = self.base.step(-float(ax_r[STICK_Y]), -float(ax_r[STICK_X]), click, locked)
        ly = -dead(ax_l[STICK_Y])
        a[14] = float(np.sign(ly)) if abs(ly) > 0.7 and abs(self.torso_latch) < 0.7 else 0.0
        self.torso_latch = ly
        return a, list(st.events)
