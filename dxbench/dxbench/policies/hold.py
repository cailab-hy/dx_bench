"""Hold policy for `rby1_ik`: arms keep their pose (zero deltas), grippers open, torso preset unchanged, base still.
Used for scene snapshots (e.g. BED-01, which has no oracle)."""
from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import torch

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg


@dataclass
class HoldPolicyCfg(PolicyCfg):
    device: str = "cuda:0"


@register_policy
class HoldPolicy(PolicyBase[HoldPolicyCfg]):
    name = "dx_hold"

    def __init__(self, config: HoldPolicyCfg):
        super().__init__(config)
        self.cfg = config

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        pass

    def get_action(self, env: gym.Env, observation) -> torch.Tensor:
        U = env.unwrapped
        a = torch.zeros(U.num_envs, U.action_manager.total_action_dim, device=U.device)
        if a.shape[1] == 17:  # rby1_ik: [r arm 6, l arm 6, gripper r, l (1 = open), torso step, base v, w]
            a[:, 12:14] = 1.0
        return a
