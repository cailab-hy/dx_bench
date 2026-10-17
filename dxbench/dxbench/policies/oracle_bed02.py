"""Scripted oracle for BED-02 Easy (fold a towel on the desk) on `rby1_ik` - privileged (reads the cloth nodes).

Robot starts in front of the desk (facing -x). Steps:
1. drive to the desk front (base front 5 cm from the edge), torso crouch (reaches 0.5-0.8 m ahead at desk height),
2. fold 1 (long axis in half): pinch the right short edge (assisted grasp), lift, carry over the left end, lower,
   release,
3. fold 2 (short axis in half): pinch the far long edge of the folded towel, carry it onto the near edge, release,
4. move the folded towel to the stack spot (task.stack_xy): pinch its centre, carry, release,
5. retract, torso upright.
The towel frame comes from a PCA of its nodes; "right" / "far" are in the base frame. `policy.subtask` follows
the BED-02 subtask labels for recording.
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
    work_preset_steps: int = 2    # crouch
    device: str = "cuda:0"


@register_policy
class Bed02OraclePolicy(PolicyBase[Bed02OraclePolicyCfg]):
    name = "dx_bed02_oracle"

    def __init__(self, config: Bed02OraclePolicyCfg):
        super().__init__(config)
        self.cfg = config
        self._h = Bed03OraclePolicy.__new__(Bed03OraclePolicy)  # reuse the geometry / servo / nav helpers
        self._h.cfg = config
        self.reset()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        self.phase = "init"
        self.t_phase = 0
        self.subtask = "pick up a shirt"
        self.plan: list[dict] = []
        self.step_i = 0
        self.goal = None
        self.target = None
        self.plan_log: list[dict] = []

    def _set(self, phase, subtask=None):
        self.phase, self.t_phase = phase, 0
        if subtask:
            self.subtask = subtask

    # ------------------------------------------------------------------ geometry
    def _towel(self, U):
        n = (_T(U.scene[self.target].data.nodal_pos_w)[0] - U.scene.env_origins[0]).cpu().numpy()
        c = n.mean(axis=0)
        xy = n[:, :2] - c[:2]
        w, V = np.linalg.eigh(xy.T @ xy)
        u = V[:, 1]  # long axis
        v = V[:, 0]
        pu, pv = xy @ u, xy @ v
        return n, c, u, v, (float(pu.min()), float(pu.max())), (float(pv.min()), float(pv.max()))

    def _plan_folds(self, U, table_z):
        """Waypoints (world xyz) of one pinch-carry-release move per fold, from the current towel nodes."""
        n, c, u, v, (u0, u1), (v0, v1) = self._towel(U)
        _, yaw = self._h._base(U)
        right = np.array([math.sin(yaw), -math.cos(yaw)])  # base -y in the world
        fwd = np.array([math.cos(yaw), math.sin(yaw)])
        if np.dot(u, right) < 0:
            u = -u
            u0, u1 = -u1, -u0
        if np.dot(v, fwd) < 0:
            v = -v
            v0, v1 = -v1, -v0
        top = float(n[:, 2].max())
        z_pick = top + GRIP_DZ
        z_put = table_z + 0.03 + GRIP_DZ
        p1 = c[:2] + u * (u1 - 0.04)                      # right short edge (middle)
        q1 = c[:2] + u * (u0 + 0.04)                      # onto the left end
        return {"pick": np.r_[p1, z_pick], "put": np.r_[q1, z_put], "label": "fold in half"}

    def _plan_fold2(self, U, table_z):
        n, c, u, v, (u0, u1), (v0, v1) = self._towel(U)
        _, yaw = self._h._base(U)
        fwd = np.array([math.cos(yaw), math.sin(yaw)])
        # the far edge: the axis most aligned with "forward"
        ax = u if abs(np.dot(u, fwd)) > abs(np.dot(v, fwd)) else v
        lo, hi = (u0, u1) if ax is u else (v0, v1)
        if np.dot(ax, fwd) < 0:
            ax, lo, hi = -ax, -hi, -lo
        top = float(n[:, 2].max())
        p = c[:2] + ax * (hi - 0.04)
        q = c[:2] + ax * (lo + 0.04)
        return {"pick": np.r_[p, top + GRIP_DZ], "put": np.r_[q, table_z + 0.03 + GRIP_DZ], "label": "fold the right side"}

    def _plan_stack(self, U, task, table_z):
        n, c, *_ = self._towel(U)
        top = float(n[:, 2].max())
        return {"pick": np.r_[c[:2], top + GRIP_DZ], "put": np.r_[task.stack_xy, table_z + 0.05 + GRIP_DZ],
                "label": "place on the stack"}

    # ------------------------------------------------------------------ main
    def get_action(self, env: gym.Env, observation) -> torch.Tensor:
        from dxbench.tasks import bed_tasks

        U = env.unwrapped
        h = self._h
        a = np.zeros(17, dtype=np.float32)
        a[12:14] = 1.0
        grip = 1.0
        self.t_phase += 1
        torso = U.action_manager.get_term("torso")
        tidx = int(torso.preset_index[0])

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
                p, yaw = h._base(U)
                _, c, *_ = self._towel(U)
                edge_x = float(desk[1][0])  # desk front edge (robot on the +x side, facing -x)
                bx = edge_x + BASE_FRONT_X + self.cfg.edge_margin
                self.goal = (np.array([bx, float(c[1])]), math.pi)
                self.table_z = float(desk[1][2])
            (v, w), ok = h._nav(U, self.goal[0], self.goal[1])
            a[15:17] = (v, w)
            if ok or self.t_phase > 300:
                self.goal = None
                self._set("crouch")
        elif self.phase == "crouch":
            a[14] = torso_to(self.cfg.work_preset_steps)
            if tidx == self.cfg.work_preset_steps and self.t_phase > 60:
                self.moves = ["fold1", "fold2", "stack"]
                self._set("plan")
        elif self.phase == "plan":
            if not self.moves:
                self._set("retract")
            else:
                m = self.moves.pop(0)
                mv = {"fold1": lambda: self._plan_folds(U, self.table_z),
                      "fold2": lambda: self._plan_fold2(U, self.table_z),
                      "stack": lambda: self._plan_stack(U, task, self.table_z)}[m]()
                self.mv = mv
                self.plan_log.append({k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in mv.items()})
                self.subtask = {"fold1": "fold in half", "fold2": "fold the right side", "stack": "place on the stack"}[m]
                self._set("above")
        elif self.phase in ("above", "descend", "close", "lift", "carry", "lower", "release"):
            mv = self.mv
            pick, put = mv["pick"], mv["put"]
            dz = self.cfg.lift_dz
            if self.phase == "above":
                a[0:6], err = h._servo(observation, h._to_base(U, pick + [0, 0, dz]))
                if err < 0.015 or self.t_phase > 90:
                    self._set("descend")
            elif self.phase == "descend":
                a[0:6], err = h._servo(observation, h._to_base(U, pick), max_dp=0.012)
                if err < 0.01 or self.t_phase > 60:
                    self._set("close")
            elif self.phase == "close":
                grip = 0.0
                if self.t_phase > 10:
                    self._set("lift")
            elif self.phase == "lift":
                grip = 0.0
                a[0:6], err = h._servo(observation, h._to_base(U, pick + [0, 0, dz]))
                if err < 0.02 or self.t_phase > 45:
                    self._set("carry")
            elif self.phase == "carry":
                grip = 0.0
                a[0:6], err = h._servo(observation, h._to_base(U, put + [0, 0, dz]), max_dp=0.015)
                if err < 0.015 or self.t_phase > 120:
                    self._set("lower")
            elif self.phase == "lower":
                grip = 0.0
                a[0:6], err = h._servo(observation, h._to_base(U, put), max_dp=0.01)
                if err < 0.01 or self.t_phase > 45:
                    self._set("release")
            elif self.phase == "release":
                grip = 1.0
                a[0:6], _ = h._servo(observation, h._to_base(U, put + [0, 0, 0.06]), max_dp=0.01)
                if self.t_phase > 25:
                    self._set("plan")
        elif self.phase == "retract":
            a[14] = torso_to(0)
            a[0:6], err = h._servo(observation, np.array([0.35, -0.25, 0.85]))
            if tidx == 0 and self.t_phase > 60:
                self._set("finished")
        a[12] = grip
        return torch.as_tensor(a, device=U.device).unsqueeze(0).repeat(U.num_envs, 1)
