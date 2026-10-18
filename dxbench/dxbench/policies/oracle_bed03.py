"""Scripted oracle for BED-03 (laundry to basket) on the `rby1_ik` embodiment — privileged (reads object states).

    python submodules/IsaacLab-Arena/isaaclab_arena/evaluation/policy_runner.py \
        --policy_type dxbench.policies.oracle_bed03.Bed03OraclePolicy --num_episodes 1 \
        --external_environment_class_path dxbench.tasks.bed_tasks:DxBed03Environment dx_bed03 --scene <json>

Per garment (nearest first): go to the shirt (base pose with the garment 0.30 m ahead of the base front, under the
right hand) -> floor posture (torso preset) -> pick up the shirt (TCP above, down to the cloth, close: assisted
grasp) -> straighten -> go to the basket -> put it in the basket (release above the rim). The current subtask label is
exposed as `policy.subtask` for recording.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg

BASE_FRONT_X = 0.32
TCP_OFF_Y = -0.20  # right hand lateral offset in the base frame during picks
ROBOT_R = 0.45     # base footprint radius (m) for approach planning
AXLE_X = 0.228     # drive-wheel axle ahead of the base origin (the base turns about it)


@dataclass
class Bed03OraclePolicyCfg(PolicyCfg):
    pick_ahead_m: float = 0.18       # garment ahead of the base front when picking (deep posture: fingertip 1.3 cm
                                     # above the floor at 0.15-0.20 m, 4 cm at 0.30 m - out of the 5 cm assist radius)
    place_ahead_m: float = 0.35      # basket centre ahead of the base front when placing (0.30: a rotated basket's
                                     # near corner can touch the base front; crouch still reaches 0.67 m)
    place_tcp_z: float = 0.50        # release height (TCP); crouch reaches x 0.55-0.70 m at z 0.45-0.75 m. The
                                     # garment (pinched near its centre) hangs ~0.4 m below the hand, into the basket
    place_above_z: float = 0.75      # approach height above the basket centre
    press_tcp_z: float = 0.40        # after release, push the garment down below the rim (32 cm) once
    max_v: float = 0.35
    max_w: float = 0.8
    floor_preset_steps: int = 3      # torso preset index for floor picking (0 upright .. 3 deep)
    place_preset_steps: int = 2      # torso preset for placing into the basket (crouch); upright cannot reach it
    device: str = "cuda:0"


def _T(x):
    import warp as wp

    if hasattr(x, "torch"):
        return x.torch
    return wp.to_torch(x) if isinstance(x, wp.array) else x


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


@register_policy
class Bed03OraclePolicy(PolicyBase[Bed03OraclePolicyCfg]):
    name = "dx_bed03_oracle"

    def __init__(self, config: Bed03OraclePolicyCfg):
        super().__init__(config)
        self.cfg = config
        self.reset()

    # ---------------------------------------------------------------- bookkeeping
    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        self.phase = "init"
        self.queue: list[str] = []
        self.target: str | None = None
        self.t_phase = 0
        self.torso_idx = 0
        self.subtask = "look for clothes"
        self.goal = None
        self.done_clothes: list[str] = []
        self.layout = None
        self.plan_log: list[dict] = []
        self.last_err = None
        self.retries = 0
        self.regrasps = 0
        self.replans = 0
        self.homed_for_place = False

    def _set(self, phase, subtask=None):
        self.phase, self.t_phase = phase, 0
        if subtask:
            self.subtask = subtask

    # ---------------------------------------------------------------- state access (env 0)
    def _base(self, U):
        r = U.scene["robot"]
        p = (_T(r.data.root_link_pos_w)[0] - U.scene.env_origins[0]).cpu().numpy()
        x, y, z, w = _T(r.data.root_link_quat_w)[0].tolist()
        return p, math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def _tcp_b(self, obs):
        return obs["policy"]["tcp_right"][0, :3].cpu().numpy()

    def _cloth_xyz(self, U, name):
        n = _T(U.scene[name].data.nodal_pos_w)[0] - U.scene.env_origins[0]
        c = n.mean(dim=0)
        return c.cpu().numpy(), n.cpu().numpy()

    def _to_base(self, U, xyz):
        p, yaw = self._base(U)
        d = np.asarray(xyz[:2]) - p[:2]
        c, s = math.cos(-yaw), math.sin(-yaw)
        return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1], xyz[2]])

    # ---------------------------------------------------------------- approach planning
    def _obstacles(self, U, exclude: str | None):
        """(xy, radius) discs the base must avoid: basket and rigid distractors. Garments are not obstacles: the
        wheels / base do not collide with cloth (D-18)."""
        obs = []
        b = (_T(U.scene["laundry_basket"].data.root_link_pos_w)[0] - U.scene.env_origins[0]).cpu().numpy()
        obs.append((b[:2], 0.38))  # 60 x 44 cm basket: half diagonal 0.37 m
        for n in ("doll", "book"):
            if n in U.scene.rigid_objects.keys():
                p = (_T(U.scene[n].data.root_link_pos_w)[0] - U.scene.env_origins[0]).cpu().numpy()
                obs.append((p[:2], 0.12))
        return obs

    def _clear(self, xy, obs, layout, margin=0.0) -> float:
        """Signed clearance of the robot disc at xy (m; < 0 = collision) to discs, furniture AABBs and walls."""
        x, y = xy
        c = 9.0
        for o, r in obs:
            c = min(c, float(np.hypot(x - o[0], y - o[1])) - r - ROBOT_R)
        if layout is not None:
            x0, x1, y0, y1 = layout["room_box_xy"]
            c = min(c, x - (x0 + 0.2) - ROBOT_R, (x1 - 0.2) - x - ROBOT_R, y - (y0 + 0.2) - ROBOT_R,
                    (y1 - 0.2) - y - ROBOT_R)
            for lo, hi in layout["furniture_aabb"].values():
                dx = max(lo[0] - x, 0.0, x - hi[0])
                dy = max(lo[1] - y, 0.0, y - hi[1])
                c = min(c, float(np.hypot(dx, dy)) - ROBOT_R)
        return c - margin

    def _grid(self, obs, layout, res=0.05):
        """Occupancy grid of base-centre positions (True = blocked): obstacles inflated by ROBOT_R."""
        x0, x1, y0, y1 = layout["room_box_xy"]
        xs = np.arange(x0, x1, res)
        ys = np.arange(y0, y1, res)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        clear = np.minimum.reduce([X - (x0 + 0.2), (x1 - 0.2) - X, Y - (y0 + 0.2), (y1 - 0.2) - Y]) - ROBOT_R
        for lo, hi in layout["furniture_aabb"].values():
            dx = np.maximum.reduce([lo[0] - X, np.zeros_like(X), X - hi[0]])
            dy = np.maximum.reduce([lo[1] - Y, np.zeros_like(Y), Y - hi[1]])
            clear = np.minimum(clear, np.hypot(dx, dy) - ROBOT_R)
        for o, r in obs:
            clear = np.minimum(clear, np.hypot(X - o[0], Y - o[1]) - r - ROBOT_R)
        return {"x0": x0, "y0": y0, "res": res, "clear": clear, "shape": X.shape}

    def _cell(self, G, xy):
        i = int(round((xy[0] - G["x0"]) / G["res"]))
        j = int(round((xy[1] - G["y0"]) / G["res"]))
        return min(max(i, 0), G["shape"][0] - 1), min(max(j, 0), G["shape"][1] - 1)

    def _astar(self, G, start, goal, margin=0.03):
        """8-connected A* on the grid; cells with clearance < margin are blocked (the start cell is always allowed,
        and cells near the start may be left if they are less blocked). Returns xy waypoints or None."""
        import heapq

        free = G["clear"] >= margin
        s, g = self._cell(G, start), self._cell(G, goal)
        if not free[g]:
            return None
        free = free.copy()
        si, sj = s
        r = int(0.3 / G["res"])  # allow leaving a tight spot near the start
        sl = (slice(max(si - r, 0), si + r + 1), slice(max(sj - r, 0), sj + r + 1))
        free[sl] |= G["clear"][sl] > -0.15
        free[s] = True
        H = lambda c: G["res"] * math.hypot(c[0] - g[0], c[1] - g[1])  # noqa: E731
        openq = [(H(s), 0.0, s)]
        came, cost = {s: None}, {s: 0.0}
        steps = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        while openq:
            _, c, cur = heapq.heappop(openq)
            if cur == g:
                break
            if c > cost[cur] + 1e-9:
                continue
            for di, dj in steps:
                n = (cur[0] + di, cur[1] + dj)
                if not (0 <= n[0] < G["shape"][0] and 0 <= n[1] < G["shape"][1]) or not free[n]:
                    continue
                nc = c + G["res"] * math.hypot(di, dj) + 0.02 * max(0.0, 0.3 - G["clear"][n])  # prefer clearance
                if nc < cost.get(n, 1e9):
                    cost[n], came[n] = nc, cur
                    heapq.heappush(openq, (nc + H(n), nc, n))
        if g not in came:
            return None
        path, cur = [], g
        while cur is not None:
            path.append((G["x0"] + cur[0] * G["res"], G["y0"] + cur[1] * G["res"]))
            cur = came[cur]
        path.reverse()
        path[-1] = (float(goal[0]), float(goal[1]))
        return [np.asarray(p) for p in path[::3]] + [np.asarray(goal, dtype=float)]

    def _plan_approach(self, U, target_xy, off_xy, exclude, obs=None):
        """Base goal (xy, yaw) putting `target_xy` at `off_xy` in the base frame + an A* path to it. 16 approach
        directions; the shortest path (+ turning) wins. Returns (goal, yaw, ok, path_len, goal_clear, path)."""
        if self.layout is None:
            from dxbench.tasks.scene_spec import load_layout

            self.layout = load_layout()
        p, yaw0 = self._base(U)
        obs = self._obstacles(U, exclude) if obs is None else obs
        G = self._grid(obs, self.layout)
        best, best_key = None, None
        for k in range(16):
            yaw = math.atan2(target_xy[1] - p[1], target_xy[0] - p[0]) + k * 2 * math.pi / 16
            cy, sy = math.cos(yaw), math.sin(yaw)
            goal = np.asarray(target_xy) - np.array([cy * off_xy[0] - sy * off_xy[1], sy * off_xy[0] + cy * off_xy[1]])
            gc = float(G["clear"][self._cell(G, goal)])
            if gc < 0.03:
                continue
            path = self._astar(G, p[:2], goal)
            if path is None:
                continue
            length = sum(float(np.linalg.norm(path[i + 1] - path[i])) for i in range(len(path) - 1))
            key = length + 0.25 * abs(wrap(yaw - yaw0))
            if best_key is None or key < best_key:
                best, best_key = (goal, yaw, True, length, gc, path), key
        if best is None:  # fall back: direct approach from the current side
            yaw = math.atan2(target_xy[1] - p[1], target_xy[0] - p[0])
            cy, sy = math.cos(yaw), math.sin(yaw)
            goal = np.asarray(target_xy) - np.array([cy * off_xy[0] - sy * off_xy[1], sy * off_xy[0] + cy * off_xy[1]])
            best = (goal, yaw, False, float(np.linalg.norm(goal - p[:2])), -1.0, [goal])
        return best

    def _follow(self, U, path, goal_yaw, lookahead=0.35):
        """(v, w) along waypoints: aim at the first waypoint >= lookahead away; final yaw at the end."""
        p, _ = self._base(U)
        while len(path) > 1 and float(np.linalg.norm(path[0] - p[:2])) < lookahead:
            path.pop(0)
        if len(path) > 1:
            (v, w), _ = self._nav(U, path[0], goal_yaw, final=False)
            return (v, w), False
        return self._nav(U, path[0], goal_yaw)

    def _plan_approach_basket(self, U, b_xy):
        """Approach the basket facing one of its sides (long sides first), so its near wall is a known distance from
        the base front: basket centre at BASE_FRONT_X + half extent + 6 cm ahead, under the right hand."""
        bq = _T(U.scene["laundry_basket"].data.root_link_quat_w)[0].tolist()
        x, y, z, w = bq
        byaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        obs = [(np.asarray(b_xy), 0.2)] + self._obstacles(U, None)[1:]
        if self.layout is None:
            from dxbench.tasks.scene_spec import load_layout

            self.layout = load_layout()
        G = self._grid(obs, self.layout)
        p, yaw0 = self._base(U)
        best, best_key = None, None
        # basket frame: x = long side (0.60), y = short side (0.44). Facing a long side = approaching along +-y.
        for k, half in ((1, 0.22), (3, 0.22), (0, 0.30), (2, 0.30)):
            yaw = byaw + k * math.pi / 2
            ahead = BASE_FRONT_X + half + 0.06
            cy, sy = math.cos(yaw), math.sin(yaw)
            goal = np.asarray(b_xy) - np.array([cy * ahead - sy * TCP_OFF_Y, sy * ahead + cy * TCP_OFF_Y])
            if float(G["clear"][self._cell(G, goal)]) < 0.03:
                continue
            path = self._astar(G, p[:2], goal)
            if path is None:
                continue
            length = sum(float(np.linalg.norm(path[i + 1] - path[i])) for i in range(len(path) - 1))
            key = length + 0.25 * abs(wrap(yaw - yaw0)) + (0.0 if half < 0.25 else 0.5)
            if best_key is None or key < best_key:
                best, best_key = (goal, yaw, True, length, float(G["clear"][self._cell(G, goal)]), path), key
        if best is None:
            return self._plan_approach(U, b_xy, (BASE_FRONT_X + self.cfg.place_ahead_m, TCP_OFF_Y), None, obs=obs)
        return best

    # ---------------------------------------------------------------- primitives
    def _nav(self, U, goal_xy, goal_yaw, final=True):
        """(v, w) towards a base pose: turn to the goal, drive, turn to the final yaw. Returns (cmd, reached).
        Controls the drive-axle midpoint (the point the base turns about, AXLE_X ahead of the base origin), turns in
        place while the heading error is large (a 0.35 m/s / 0.8 rad/s arc is wider than the last metres - the robot
        orbited the goal). final=False: an intermediate waypoint (no final turn)."""
        p, yaw = self._base(U)
        ax = p[:2] + AXLE_X * np.array([math.cos(yaw), math.sin(yaw)])
        g = np.asarray(goal_xy, dtype=float)
        if final:
            g = g + AXLE_X * np.array([math.cos(goal_yaw), math.sin(goal_yaw)])
        d = g - ax
        dist = float(np.linalg.norm(d))
        tol = 0.03 if final else 0.0
        if dist > tol:
            head = math.atan2(d[1], d[0])
            e = wrap(head - yaw)
            back = final and abs(e) > math.pi / 2 and dist < 0.4  # small corrections: drive backwards
            if back:
                e = wrap(e + math.pi)
            w = float(np.clip(2.0 * e, -self.cfg.max_w, self.cfg.max_w))
            if abs(e) > 0.35:
                v = 0.0  # turn in place first
            else:
                v = min(self.cfg.max_v, 1.0 * dist + 0.05) if final else self.cfg.max_v
                v *= math.cos(e) ** 2
            return (-v if back else v, w), False
        e = wrap(goal_yaw - yaw)
        if abs(e) > 0.03:
            return (0.0, float(np.clip(2.0 * e, -self.cfg.max_w, self.cfg.max_w))), False
        return (0.0, 0.0), True

    @staticmethod
    def _qmat(q):
        x, y, z, w = q
        return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])

    def _servo(self, obs, target_b, kp=0.6, max_dp=0.025, R_des=None):
        """Relative IK step towards a base-frame TCP position; orientation: gripper pointing down (identity in the
        base frame) or R_des."""
        p = self._tcp_b(obs)
        d = np.asarray(target_b) - p
        dp = np.clip(kp * d, -max_dp, max_dp)
        R = self._qmat(obs["policy"]["tcp_right"][0, 3:].cpu().numpy())  # xyzw, base frame
        Re = (R_des if R_des is not None else np.eye(3)) @ R.T
        ang = math.acos(np.clip((np.trace(Re) - 1) / 2, -1, 1))
        if ang > 1e-4:
            ax = np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]]) / (2 * math.sin(ang))
            dr = np.clip(ax * ang * kp, -0.08, 0.08)
        else:
            dr = np.zeros(3)
        return np.r_[dp, dr], float(np.linalg.norm(d))

    # ---------------------------------------------------------------- main
    def get_action(self, env: gym.Env, observation) -> torch.Tensor:
        U = env.unwrapped
        a = np.zeros(17, dtype=np.float32)
        a[12:14] = 1.0  # grippers open
        grip = 1.0
        self.t_phase += 1
        torso_term = U.action_manager.get_term("torso")
        self.torso_idx = int(torso_term.preset_index[0])

        def torso_to(idx):
            if self.torso_idx < idx:
                return 1.0 if self.t_phase % 2 == 1 else 0.0
            if self.torso_idx > idx:
                return -1.0 if self.t_phase % 2 == 1 else 0.0
            return 0.0

        if self.phase == "init":
            tcp = observation["policy"]["tcp_right"][0].cpu().numpy()
            self.home = (tcp[:3].copy(), self._qmat(tcp[3:]))
            spec = getattr(U.cfg, "_dx_spec", None)
            names = [n for n in U.scene.deformable_objects.keys()]
            garments = [n for n in names if n.startswith("cloth_")]
            p, _ = self._base(U)
            garments.sort(key=lambda n: float(np.linalg.norm(self._cloth_xyz(U, n)[0][:2] - p[:2])))
            self.queue = garments
            self._set("next", "look for clothes")
        if self.phase == "next":
            if not self.queue:
                self._set("finished", "look for clothes")
            else:
                self.target = self.queue.pop(0)
                self.retries = 0
                self.regrasps = 0
                self.replans = 0
                self.homed_for_place = False
                self._set("goto_cloth", "go to the shirt")
        if self.phase == "goto_cloth":
            if self.goal is None:  # plan once per approach: garment pick_ahead in front, under the right hand
                c, _ = self._cloth_xyz(U, self.target)
                g = self._plan_approach(U, c[:2], (BASE_FRONT_X + self.cfg.pick_ahead_m, TCP_OFF_Y), self.target)
                self.goal = (g[0], g[1], list(g[5]))
                self.plan_log.append({"target": self.target, "goal": g[0].tolist(), "yaw": g[1], "ok": g[2],
                                      "path_len": g[3], "goal_clear": g[4], "waypoints": len(g[5])})
            goal, yaw, path = self.goal
            (v, w), ok = self._follow(U, path, yaw)
            a[15:17] = (v, w)
            if ok or self.t_phase > 600:
                self.goal = None
                self._set("crouch", "pick up the shirt")
        elif self.phase == "crouch":
            a[14] = torso_to(self.cfg.floor_preset_steps)
            if self.torso_idx == self.cfg.floor_preset_steps and self.t_phase > 60:
                self._set("above")
        elif self.phase == "above":
            c, n = self._cloth_xyz(U, self.target)
            near = n[np.argsort(np.linalg.norm(n[:, :2] - c[:2], axis=1))[:30]]
            top = float(near[:, 2].max())
            tb = self._to_base(U, [c[0], c[1], top + 0.031 + 0.12])
            a[0:6], err = self._servo(observation, tb)
            if err < 0.02 or self.t_phase > 90:
                self._set("descend")
        elif self.phase == "descend":
            c, n = self._cloth_xyz(U, self.target)
            near = n[np.argsort(np.linalg.norm(n[:, :2] - c[:2], axis=1))[:30]]
            top = float(near[:, 2].max())
            tb = self._to_base(U, [c[0], c[1], top + 0.031 + 0.008])
            a[0:6], err = self._servo(observation, tb, max_dp=0.012)
            self.last_err = err
            if err < 0.012 or self.t_phase > 90:
                self._set("close")
        elif self.phase == "close":
            grip = 0.0
            if self.t_phase > 12:
                # privileged check: did the assisted grasp capture any node (right hand)? if not, open and re-descend
                got = int(U.action_manager.get_term("cloth_assist").grasped[0, 0])
                if got == 0 and self.regrasps < 2:
                    self.regrasps += 1
                    self.plan_log.append({"target": self.target, "regrasp": self.regrasps})
                    self._set("reopen")
                else:
                    self._set("lift")
        elif self.phase == "reopen":  # gripper open (default), short pause so the fingers clear, then descend again
            if self.t_phase > 8:
                self._set("descend")
        elif self.phase == "lift":
            grip = 0.0
            a[14] = torso_to(0)
            tb = self._tcp_b(observation) + np.array([0.0, 0.0, 0.02])
            a[0:6], _ = self._servo(observation, tb)
            if self.torso_idx == 0 and self.t_phase > 75:
                # verify the garment came up with the hand; otherwise home the arm and pick again (twice at most)
                _, n = self._cloth_xyz(U, self.target)
                if float(n[:, 2].max()) < 0.25 and self.retries < 2:
                    self.retries += 1
                    self.plan_log.append({"target": self.target, "retry_pick": self.retries})
                    U.action_manager.get_term("right_arm").request_home()
                    self._set("crouch", "pick up the shirt")
                elif float(n[:, 2].max()) < 0.25 and self.replans < 1:
                    # same spot failed twice (out of reach from here): re-plan the approach from the current pose
                    self.replans += 1
                    self.retries = 0
                    self.regrasps = 0
                    self.plan_log.append({"target": self.target, "replan_pick": self.replans})
                    U.action_manager.get_term("right_arm").request_home()
                    self._set("goto_cloth", "go to the shirt")
                elif float(n[:, 2].max()) < 0.25:
                    self.plan_log.append({"target": self.target, "skip": True})  # don't carry an empty hand
                    U.action_manager.get_term("right_arm").request_home()
                    self._set("next", "look for clothes")
                else:
                    self._set("goto_basket", "go to the basket")
        elif self.phase == "goto_basket":
            grip = 0.0
            if self.goal is None:
                b = (_T(U.scene["laundry_basket"].data.root_link_pos_w)[0] - U.scene.env_origins[0]).cpu().numpy()
                g = self._plan_approach_basket(U, b[:2])
                self.goal = (g[0], g[1], list(g[5]))
                self.plan_log.append({"target": "basket", "goal": g[0].tolist(), "yaw": g[1], "ok": g[2],
                                      "path_len": g[3], "goal_clear": g[4], "waypoints": len(g[5])})
            goal, yaw, path = self.goal
            (v, w), ok = self._follow(U, path, yaw)
            a[15:17] = (v, w)
            if ok or self.t_phase > 600:
                self.goal = None
                self._set("over_basket", "put it in the basket")
        elif self.phase == "over_basket":
            grip = 0.0
            a[14] = torso_to(self.cfg.place_preset_steps)
            b = (_T(U.scene["laundry_basket"].data.root_link_pos_w)[0] - U.scene.env_origins[0]).cpu().numpy()
            tb = self._to_base(U, [b[0], b[1], self.cfg.place_above_z])
            a[0:6], err = self._servo(observation, tb)
            self.last_err = err
            if self.t_phase == 90 and err > 0.25 and not self.homed_for_place:
                # the relative IK stalled far away (scene 006: 0.66 m): back to READY in joint space, then servo again
                self.homed_for_place = True
                U.action_manager.get_term("right_arm").request_home()
                self.t_phase = 0
            # the crouched arm often settles ~3 cm from the high approach point (orientation held): accept 3.5 cm
            if (err < 0.035 and self.torso_idx == self.cfg.place_preset_steps and self.t_phase > 30) or self.t_phase > 180:
                self._set("lower_basket")
        elif self.phase == "lower_basket":
            grip = 0.0
            b = (_T(U.scene["laundry_basket"].data.root_link_pos_w)[0] - U.scene.env_origins[0]).cpu().numpy()
            tb = self._to_base(U, [b[0], b[1], self.cfg.place_tcp_z])
            a[0:6], err = self._servo(observation, tb, max_dp=0.012)
            self.last_err = err
            if err < 0.015 or self.t_phase > 60:
                self._set("release")
        elif self.phase == "release":
            grip = 1.0
            if self.t_phase > 20:
                self.done_clothes.append(self.target)
                self._set("press", "put it in the basket")
        elif self.phase == "press":
            # a dropped wad can bridge the rim: push it down once with the open hand, then lift out
            b = (_T(U.scene["laundry_basket"].data.root_link_pos_w)[0] - U.scene.env_origins[0]).cpu().numpy()
            down = self.t_phase <= 30
            tb = self._to_base(U, [b[0], b[1], self.cfg.press_tcp_z if down else self.cfg.place_above_z])
            a[0:6], err = self._servo(observation, tb, max_dp=0.015)
            if self.t_phase > 60:
                self._set("retract", "look for clothes")
        elif self.phase == "retract":
            # bring the hand back to its start (READY) pose before driving: an arm left over the basket caught on it
            # while the base turned (scene 001), and a hand parked high with the gripper down sat near the wrist
            # limit, from where the relative IK could not reach the next garment (errors 0.6-0.8 m)
            a[14] = torso_to(0)
            arm = U.action_manager.get_term("right_arm")
            if self.t_phase == 1 and hasattr(arm, "request_home"):
                arm.request_home()  # joint-space return to READY (HomingDiffIKAction)
            homing = bool(arm.homing[0]) if hasattr(arm, "homing") else False
            if (not homing and self.torso_idx == 0 and self.t_phase > 45) or self.t_phase > 200:
                self._set("backoff", "look for clothes")
        elif self.phase == "backoff":
            # back away from the basket before turning: the base turns about the drive axle, its rear swings ~0.5 m
            # and knocked the basket over when the robot turned in place next to it (scene 001)
            a[15:17] = (-0.15, 0.0)
            if self.t_phase > 30:
                self._set("next", "look for clothes")
        elif self.phase == "finished":
            pass
        a[12] = grip
        return torch.as_tensor(a, device=U.device).unsqueeze(0).repeat(U.num_envs, 1)
