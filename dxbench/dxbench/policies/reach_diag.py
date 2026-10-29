"""Diagnostic (2026-10-28): how far ahead does the right TCP really get in the running env (relative IK servo, PD,
contacts), gripper down, at a given height and torso preset? Optionally drives up to the bed first. Logs per target
the smallest position error, the final TCP and the arm joints closest to their limits (`policy.plan_log`).
Environment variables: DX_DIAG_PRESET (0-3, default 3 = deep), DX_DIAG_Z (base-frame z, default 0.60),
DX_DIAG_XS (comma list, default 0.60,0.70,0.80,0.90), DX_DIAG_Y (default -0.20), DX_DIAG_PITCH (gripper tilt,
deg), DX_DIAG_BED (1: drive to 6 cm from the bed frame first)."""
from __future__ import annotations

import os
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg

from dxbench.policies.oracle_bed03 import BASE_FRONT_X, Bed03OraclePolicy


@dataclass
class ReachDiagCfg(PolicyCfg):
    max_v: float = 0.2
    max_w: float = 0.6
    device: str = "cuda:0"


@register_policy
class ReachDiag(Bed03OraclePolicy, PolicyBase[ReachDiagCfg]):
    name = "dx_reach_diag"

    def __init__(self, config: ReachDiagCfg):
        PolicyBase.__init__(self, config)
        self.cfg = config
        self.preset = int(os.environ.get("DX_DIAG_PRESET", 3))
        self.z = float(os.environ.get("DX_DIAG_Z", 0.60))
        self.xs = [float(v) for v in os.environ.get("DX_DIAG_XS", "0.60,0.70,0.80,0.90").split(",")]
        # DX_DIAG_SEQ="x,z;x,z;..": visit these (x, z) targets in order instead of DX_DIAG_XS at DX_DIAG_Z
        seq = os.environ.get("DX_DIAG_SEQ")
        self.seq = [tuple(float(v) for v in t.split(",")) for t in seq.split(";")] if seq else None
        if self.seq:
            self.xs = [t[0] for t in self.seq]
        self.bed = os.environ.get("DX_DIAG_BED", "0") == "1"
        self.y = float(os.environ.get("DX_DIAG_Y", -0.20))
        th = np.radians(float(os.environ.get("DX_DIAG_PITCH", 0.0)))  # gripper tilt about the base y axis
        self.R = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
        self.reset()

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self.phase, self.t_phase, self.i = ("approach" if self.bed else "torso"), 0, 0
        self.plan_log, self.best = [], 9.0
        self.goal = None

    def get_action(self, env: gym.Env, observation) -> torch.Tensor:
        U = env.unwrapped
        a = np.zeros(17, dtype=np.float32)
        a[12:14] = 1.0
        self.t_phase += 1
        idx = int(U.action_manager.get_term("torso").preset_index[0])
        if self.phase == "approach":
            if self.goal is None:
                from dxbench.tasks.scene_spec import load_layout

                lo, _ = load_layout()["parts_aabb"]["Bed/Mattress"]
                p, _ = self._base(U)
                self.goal = np.array([lo[0] - 0.025 - 0.06 - BASE_FRONT_X, p[1]])
            (v, w), ok = self._nav(U, self.goal, 0.0)
            a[15:17] = (v, w)
            if ok or self.t_phase > 200:
                p, _ = self._base(U)
                self.plan_log.append({"base_x": round(float(p[0]), 3), "goal_x": round(float(self.goal[0]), 3)})
                self._set("torso")
        elif self.phase == "torso":
            if idx != self.preset:
                a[14] = (1.0 if idx < self.preset else -1.0) if self.t_phase % 2 == 1 else 0.0
            elif self.t_phase > 40:
                self._set("reach")
        elif self.phase == "reach":
            if self.i >= len(self.xs):
                self._set("finished")
            else:
                tgt = np.array([self.xs[self.i], self.y, self.seq[self.i][1] if self.seq else self.z])
                a[0:6], err = self._servo(observation, tgt, max_dp=0.015, R_des=self.R)
                self.best = min(self.best, err)
                if self.t_phase > 120:
                    r = U.scene["robot"]
                    ids = U.action_manager.get_term("right_arm")._joint_ids
                    q = r.data.joint_pos.torch[0, ids].cpu().numpy()
                    lim = r.data.joint_pos_limits.torch[0, ids].cpu().numpy()
                    margin = np.minimum(q - lim[:, 0], lim[:, 1] - q)
                    self.plan_log.append({"x": self.xs[self.i], "z": float(tgt[2]), "min_err": round(self.best, 4), "final_err": round(err, 4),
                                          "tcp": np.round(self._tcp_b(observation), 3).tolist(),
                                          "joint_margin_rad": np.round(margin, 3).tolist()})
                    self.best, self.i = 9.0, self.i + 1
                    self.t_phase = 0
        return torch.as_tensor(a, device=U.device).unsqueeze(0).repeat(U.num_envs, 1)
