"""Diagnostic (2026-10-29): how close does the RB-Y1 get to the BED-02 desk in a torso preset? The desk is a table
(top 0.70-0.725 m, apron rails 0.645-0.695 m, legs at the corners), so the base can roll under its front edge until
some link (torso, arms in READY, base) touches it. Sets the preset (DX_DI_PRESET 0-3), then creeps forward at
5 cm/s until the base stops advancing; logs the base-front position relative to the desk's front edge
(negative = under the desk) and the tilt (`policy.plan_log`)."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg

from dxbench.policies.oracle_bed03 import BASE_FRONT_X, Bed03OraclePolicy


@dataclass
class DriveInDiagCfg(PolicyCfg):
    max_v: float = 0.2
    max_w: float = 0.6
    device: str = "cuda:0"


@register_policy
class DriveInDiag(Bed03OraclePolicy, PolicyBase[DriveInDiagCfg]):
    name = "dx_drive_in_diag"

    def __init__(self, config: DriveInDiagCfg):
        PolicyBase.__init__(self, config)
        self.cfg = config
        self.preset = int(os.environ.get("DX_DI_PRESET", 1))
        # DX_DI_TORSO="t1,t2,t3": replace that preset's torso_1-3 angles (rad), e.g. the rby1-sdk demo pose
        # 0.5236,-1.0472,0.5236 (examples/python/24_demo_motion.py, 30 / -60 / 30 deg)
        self.torso_override = os.environ.get("DX_DI_TORSO")
        self.reset()

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self.phase, self.t_phase = "torso", 0
        self.plan_log, self.hist = [], []

    def get_action(self, env: gym.Env, observation) -> torch.Tensor:
        from dxbench.tasks import task_env

        U = env.unwrapped
        a = np.zeros(17, dtype=np.float32)
        a[12:14] = 1.0
        self.t_phase += 1
        tterm = U.action_manager.get_term("torso")
        if self.torso_override and self.t_phase == 1:
            t = [float(v) for v in self.torso_override.split(",")]
            tterm._presets[self.preset, 1:4] = torch.tensor(t, device=U.device)
        idx = int(tterm.preset_index[0])
        desk = task_env.CURRENT_TASK.layout["parts_aabb"].get("CareDesk")
        edge = float(desk[1][0])
        p, yaw = self._base(U)
        front = float(p[0]) + BASE_FRONT_X * math.cos(yaw)  # facing -x: the front is at x - 0.32
        if self.phase == "torso":
            if idx != self.preset:
                a[14] = (1.0 if idx < self.preset else -1.0) if self.t_phase % 2 == 1 else 0.0
            elif self.t_phase > 45:
                self.plan_log.append({"preset": self.preset, "start_front_to_edge": round(front - edge, 3)})
                self._set("drive")
        elif self.phase == "drive":
            a[15] = 0.05
            self.hist.append(front)
            r = U.scene["robot"]
            q = r.data.root_link_quat_w.torch[0].tolist()
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (q[0] ** 2 + q[1] ** 2)))))
            stalled = len(self.hist) > 20 and abs(self.hist[-20] - self.hist[-1]) < 0.003
            if stalled or tilt > 3.0 or self.t_phase > 450:
                bp = (r.data.body_link_pos_w.torch[0] - U.scene.env_origins[0]).cpu().numpy()
                near = [(r.body_names[i], round(float(bp[i, 0] - edge), 3), round(float(bp[i, 2]), 3))
                        for i in np.argsort(bp[:, 0]) if 0.3 < bp[i, 2] < 0.85]  # links at desk-edge height, front first
                self.plan_log.append({"front_to_edge": round(front - edge, 3), "tilt_deg": round(tilt, 2),
                                      "stalled": stalled, "steps": self.t_phase, "base_x": round(float(p[0]), 3),
                                      "links_ahead (name, x - edge, z)": near})
                self._set("finished")
        return torch.as_tensor(a, device=U.device).unsqueeze(0).repeat(U.num_envs, 1)
