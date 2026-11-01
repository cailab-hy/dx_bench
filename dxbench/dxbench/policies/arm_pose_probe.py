"""Diagnostic (2026-11-01): servo both arms to fixed base-frame TCP targets with the gripper straight down (the BED-01
oracle's servo), the base still, then print the arm joint positions - used to make arm start presets
(embodiments/rby1_cfg.py ARM_PRESETS). Environment variables: DX_PROBE_X (default 0.62), DX_PROBE_Y (0.20: the right
hand at -y, the left at +y), DX_PROBE_Z (base-frame z, default 0.66), DX_PROBE_STEPS (default 200)."""
from __future__ import annotations

import os
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase

from dxbench.policies.oracle_bed01 import Bed01EasyOracle, Bed01EasyOracleCfg


@dataclass
class ArmPoseProbeCfg(Bed01EasyOracleCfg):
    pass


@register_policy
class ArmPoseProbe(Bed01EasyOracle, PolicyBase[ArmPoseProbeCfg]):
    name = "dx_arm_pose_probe"

    def __init__(self, config: ArmPoseProbeCfg):
        super().__init__(config)
        x, y, z = (float(os.environ.get(k, d)) for k, d in (("DX_PROBE_X", 0.62), ("DX_PROBE_Y", 0.20), ("DX_PROBE_Z", 0.66)))
        self.targets = {"right": np.array([x, -y, z]), "left": np.array([x, y, z])}
        self.steps = int(os.environ.get("DX_PROBE_STEPS", 200))
        self.t = 0

    def get_action(self, env: gym.Env, observation) -> torch.Tensor:
        U = env.unwrapped
        a = np.zeros(17, dtype=np.float32)
        a[12:14] = 1.0
        self.t += 1
        err = self._arms(a, observation, self.targets)
        if self.t == self.steps:
            r = U.scene["robot"]
            q = r.data.joint_pos.torch[0].cpu().numpy()
            arms = {n: round(float(v), 4) for n, v in zip(r.joint_names, q) if "_arm_" in n}
            tcp = {s: observation["policy"][f"tcp_{s}"][0].cpu().numpy().round(4).tolist() for s in ("right", "left")}
            print(f"[probe] err {err:.4f} arms {arms} tcp {tcp}", flush=True)
            self.plan_log.append({"probe_err": err, "arms": arms, "tcp": tcp})
        return torch.as_tensor(a, device=U.device).unsqueeze(0).repeat(U.num_envs, 1)
