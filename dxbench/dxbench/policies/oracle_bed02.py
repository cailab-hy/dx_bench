"""Scripted oracle for BED-02 Easy (fold a towel on the desk) on `rby1_ik`, both arms - privileged (reads nodes).

Robot starts in front of the desk (facing -x); the towel lies 25 cm in from the desk front, long side along the
robot's lateral axis. With the gripper pointing down each arm only reaches its own half of the desk at desk height
(reports/rby1_check/t5/reach_desk_*.json), so:
1. drive to the desk front (base front 5 cm from the edge, towel centred), torso lean,
2. "fold the right side": right hand pinches the right short edge, carries it just past the centre line, releases,
3. "fold the left side": left hand does the same with the left short edge,
4. "fold in half": right hand pinches the far long edge (right half) and lays it onto the near edge,
5. "place on the stack": left hand carries the folded towel to the stack spot (desk front, left of centre),
6. arms home, torso upright.
The towel frame comes from a PCA of its nodes. `policy.subtask` follows the BED-02 subtask labels.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg

from dxbench.policies.oracle_bed03 import BASE_FRONT_X, Bed03OraclePolicy, _T

GRIP_DZ = 0.031 + 0.008   # TCP height above the cloth surface when pinching (fingertips 8 mm above)


@dataclass
class Bed02OraclePolicyCfg(PolicyCfg):
    edge_margin: float = 0.05     # base front to desk edge
    lift_dz: float = 0.10
    max_v: float = 0.25
    max_w: float = 0.6
    work_preset_steps: int = 1    # lean: crouch cannot reach z 0.8-0.9 near the centre line at x 0.6 (reach_desk_0.0.json)
    device: str = "cuda:0"


@register_policy
class Bed02OraclePolicy(PolicyBase[Bed02OraclePolicyCfg]):
    name = "dx_bed02_oracle"

    def __init__(self, config: Bed02OraclePolicyCfg):
        super().__init__(config)
        self.cfg = config
        self._h = Bed03OraclePolicy.__new__(Bed03OraclePolicy)  # reuse the geometry / nav helpers
        self._h.cfg = config
        self.reset()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        self.phase = "init"
        self.t_phase = 0
        self.subtask = "pick up a shirt"
        self.goal = None
        self.target = None
        self.moves: list[str] = []
        self.mv = None
        self.plan_log: list[dict] = []
        self.last_err = None
        self.arc_s = 0.0

    def _set(self, phase, subtask=None):
        self.phase, self.t_phase = phase, 0
        if subtask:
            self.subtask = subtask

    # ------------------------------------------------------------------ helpers
    def _servo(self, obs, target_b, side, kp=0.6, max_dp=0.025):
        """Relative IK step of one arm towards a base-frame TCP position, gripper pointing down."""
        tcp = obs["policy"][f"tcp_{side}"][0].cpu().numpy()
        d = np.asarray(target_b) - tcp[:3]
        dp = np.clip(kp * d, -max_dp, max_dp)
        Re = np.eye(3) @ self._h._qmat(tcp[3:]).T
        ang = math.acos(np.clip((np.trace(Re) - 1) / 2, -1, 1))
        dr = np.zeros(3)
        if ang > 1e-4:
            ax = np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]]) / (2 * math.sin(ang))
            dr = np.clip(ax * ang * kp, -0.08, 0.08)
        self.last_err = float(np.linalg.norm(d))
        return np.r_[dp, dr], self.last_err

    def _towel(self, U):
        n = (_T(U.scene[self.target].data.nodal_pos_w)[0] - U.scene.env_origins[0]).cpu().numpy()
        c = n.mean(axis=0)
        xy = n[:, :2] - c[:2]
        _, V = np.linalg.eigh(xy.T @ xy)
        u, v = V[:, 1], V[:, 0]  # long, short axis
        p, yaw = self._h._base(U)
        right = np.array([math.sin(yaw), -math.cos(yaw)])  # base -y in the world
        fwd = np.array([math.cos(yaw), math.sin(yaw)])
        if np.dot(u, right) < 0:
            u = -u
        if np.dot(v, fwd) < 0:
            v = -v
        pu, pv = xy @ u, xy @ v
        return n, c, u, v, (float(pu.min()), float(pu.max())), (float(pv.min()), float(pv.max()))

    def _plan(self, U, move, task):
        n, c, u, v, (u0, u1), (v0, v1) = self._towel(U)
        top = float(n[:, 2].max())
        z_pick, z_put = top + GRIP_DZ, self.table_z + 0.03 + GRIP_DZ
        if move == "right":   # right short edge -> just past the centre line
            pick, put, side, label = c[:2] + u * (u1 - 0.04), c[:2] - u * 0.03, "right", "fold the right side"
        elif move == "left":  # left short edge -> just past the centre line
            pick, put, side, label = c[:2] + u * (u0 + 0.04), c[:2] + u * 0.03, "left", "fold the left side"
        elif move == "half":  # far long edge (right half) -> near edge
            pick = c[:2] + v * (v1 - 0.03) + u * 0.25 * u1
            put = c[:2] + v * (v0 + 0.03) + u * 0.25 * u1
            side, label = "right", "fold in half"
        else:                 # folded towel -> stack spot, left hand
            pick, put, side, label = c[:2], np.asarray(task.stack_xy), "left", "place on the stack"
            z_put = self.table_z + 0.05 + GRIP_DZ
        return {"pick": np.r_[pick, z_pick], "put": np.r_[put, z_put], "side": side, "label": label}

    # ------------------------------------------------------------------ main
    def get_action(self, env: gym.Env, observation) -> torch.Tensor:
        from dxbench.tasks import bed_tasks

        U = env.unwrapped
        h = self._h
        a = np.zeros(17, dtype=np.float32)
        a[12:14] = 1.0
        self.t_phase += 1
        tidx = int(U.action_manager.get_term("torso").preset_index[0])

        def torso_to(idx):
            if tidx < idx:
                return 1.0 if self.t_phase % 2 == 1 else 0.0
            if tidx > idx:
                return -1.0 if self.t_phase % 2 == 1 else 0.0
            return 0.0

        task = bed_tasks.CURRENT_TASK
        if self.phase == "init":
            self.target = task.cloth_names[0]
            self._set("approach", "pick up a shirt")
        if self.phase == "approach":
            if self.goal is None:
                desk = task.layout["parts_aabb"].get("CareDesk") or task.layout["furniture_aabb"]["CareDesk"]
                _, c, *_ = self._towel(U)
                bx = float(desk[1][0]) + BASE_FRONT_X + self.cfg.edge_margin  # robot on the +x side, facing -x
                self.goal = (np.array([bx, float(c[1])]), math.pi)
                self.table_z = float(desk[1][2])
            (v, w), ok = h._nav(U, self.goal[0], self.goal[1])
            a[15:17] = (v, w)
            if ok or self.t_phase > 300:
                self.goal = None
                self._set("crouch", "flatten the shirt")
        elif self.phase == "crouch":
            a[14] = torso_to(self.cfg.work_preset_steps)
            if tidx == self.cfg.work_preset_steps and self.t_phase > 60:
                self.moves = ["right", "left", "half", "stack"]
                self._set("plan")
        elif self.phase == "plan":
            if not self.moves:
                self._set("retract")
            else:
                self.mv = self._plan(U, self.moves.pop(0), task)
                self.plan_log.append({k: (x.tolist() if hasattr(x, "tolist") else x) for k, x in self.mv.items()})
                self._set("above", self.mv["label"])
        elif self.phase in ("above", "descend", "close", "lift", "carry", "lower", "release"):
            mv, dz = self.mv, self.cfg.lift_dz
            side = mv["side"]
            off, gi = (0, 12) if side == "right" else (6, 13)
            pick, put = mv["pick"], mv["put"]
            grip, cmd = 1.0, None
            if self.phase == "above":
                cmd, err = self._servo(observation, h._to_base(U, pick + [0, 0, dz]), side)
                if err < 0.015 or self.t_phase > 90:
                    self._set("descend")
            elif self.phase == "descend":
                cmd, err = self._servo(observation, h._to_base(U, pick), side, max_dp=0.012)
                if err < 0.01 or self.t_phase > 60:
                    self._set("close")
            elif self.phase == "close":
                grip = 0.0
                if self.t_phase > 10:
                    self._set("lift")
            elif self.phase == "lift":  # short vertical lift so the pinched edge leaves the desk
                grip = 0.0
                cmd, err = self._servo(observation, h._to_base(U, pick + [0, 0, 0.03]), side)
                if err < 0.015 or self.t_phase > 30:
                    self.arc_s = 0.0
                    self._set("carry")
            elif self.phase == "carry":  # fold arc: the edge rises to ~half the fold distance and comes down past the line
                grip = 0.0
                D = float(np.linalg.norm(put[:2] - pick[:2]))
                hgt = max(dz, 0.5 * D + 0.03)
                # ~1.2 cm of path per step; half speed while the hand lags > 5 cm behind the waypoint (the pinned cloth
                # drags on the desk, which leaves a steady ~3 cm servo error)
                rate = 0.012 if (self.last_err is None or self.last_err < 0.05) else 0.006
                self.arc_s = min(1.0, self.arc_s + rate / max(D + 2 * hgt, 0.05))
                s_ = self.arc_s
                wp = pick + (put - pick) * s_
                wp[2] = pick[2] + 0.03 + (put[2] - pick[2] - 0.03) * s_ + hgt * math.sin(math.pi * s_)
                cmd, err = self._servo(observation, h._to_base(U, wp), side, max_dp=0.02)
                if (s_ >= 1.0 and err < 0.04) or self.t_phase > 240:
                    self._set("lower")
            elif self.phase == "lower":
                grip = 0.0
                cmd, err = self._servo(observation, h._to_base(U, put), side, max_dp=0.01)
                if err < 0.01 or self.t_phase > 30:
                    self._set("release")
            elif self.phase == "release":  # open in place first (pins drop when the fingers open), then rise
                up = 0.0 if self.t_phase <= 8 else 0.08
                cmd, _ = self._servo(observation, h._to_base(U, put + [0, 0, up]), side, max_dp=0.01)
                if self.t_phase > 30:
                    U.action_manager.get_term(f"{side}_arm").request_home()  # clear the workspace for the other arm
                    self._set("rehome")
            if cmd is not None:
                a[off:off + 6] = cmd
                st = self.plan_log[-1].setdefault("min_err", {})
                st[self.phase] = round(min(st.get(self.phase, 9.0), self.last_err), 4)
            a[gi] = grip
        elif self.phase == "rehome":
            arm = U.action_manager.get_term(f"{self.mv['side']}_arm")
            if not bool(arm.homing[0]) or self.t_phase > 90:
                self._set("plan")
        elif self.phase == "retract":
            a[14] = torso_to(0)
            if tidx == 0 and self.t_phase > 60:
                self._set("finished")
        return torch.as_tensor(a, device=U.device).unsqueeze(0).repeat(U.num_envs, 1)
