"""Arena remote policy client for dxbench_policy.serve.server (dx_arena side; no lerobot import).

    # server (dx_policy):   python -m dxbench_policy.serve.server --backend lerobot --checkpoint <dir> --port 8765
    python dxbench/dxbench/scripts/run_episode.py --policy dxbench.policies.rby1_lerobot_remote.Rby1LerobotRemotePolicy \
        --policy_kw remote_port=8765 --out outputs/play/x --video \
        --external_environment_class_path dxbench.tasks.bed_tasks:DxBed03Environment dx_bed03 --scene <json> \
        --embodiment rby1_joint --enable_cameras

Wire format (see server docstring): observation.state 24 (dx24: rby1 state16 + torso 6 + base velocity 2) or 16 (rby1),
observation.images.front/right/left = head / right wrist / left wrist RGB uint8 (480 x 640), task = language.
Actions come back as (H, 24) for the `rby1_joint` embodiment; a 16-D (rby1) server is padded with the current torso
targets and a zero base command. The gripper state is re-normalized like rby1_format (half gap / 0.045, clipped).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg
from isaaclab_arena.policy.remote_policy_base import EmbodimentAdapter, RemoteChunkReplayPolicy

FINGER_OPEN_M = 0.045
CAM = {"observation.images.front": "head_rgb", "observation.images.right": "right_wrist_rgb",
       "observation.images.left": "left_wrist_rgb"}


@dataclass
class Rby1LerobotRemotePolicyCfg(PolicyCfg):
    policy_device: str = "cuda:0"
    remote_host: str = "127.0.0.1"
    remote_port: int = 8765
    open_loop_horizon: int = 8
    ping_interval: float | None = 20.0
    ping_timeout: float | None = 300.0
    fmt: str = "dx24"


class Rby1Adapter(EmbodimentAdapter):
    def __init__(self, fmt: str = "dx24"):
        self.fmt = fmt
        self.action_dim = 24 if fmt == "dx24" else 16

    def extract(self, observation: dict[str, Any], env_id: int) -> dict[str, np.ndarray]:
        pol = observation["policy"]
        s16 = pol["state16"][env_id].detach().cpu().numpy().astype(np.float32).copy()
        # obs gripper = half gap / 0.05 -> rby1_format half gap / 0.045
        s16[14:16] = np.clip(s16[14:16] * 0.05 / FINGER_OPEN_M, 0.0, 1.0)
        if self.fmt == "dx24":
            st = np.concatenate([s16, pol["torso"][env_id].detach().cpu().numpy(),
                                 pol["base_vel"][env_id].detach().cpu().numpy()]).astype(np.float32)
        else:
            st = s16
        out = {"observation.state": st}
        cam = observation.get("camera_obs", {})
        for k, src in CAM.items():
            if src in cam:
                out[k] = np.ascontiguousarray(cam[src][env_id].detach().cpu().numpy()[..., :3].astype(np.uint8))
        return out

    def pack_request(self, extracted, language_instruction: str) -> dict[str, Any]:
        return {**extracted, "task": language_instruction or ""}


@register_policy
class Rby1LerobotRemotePolicy(RemoteChunkReplayPolicy, PolicyBase[Rby1LerobotRemotePolicyCfg]):
    name = "rby1_lerobot_remote"
    server_response_actions_key = "actions"

    def __init__(self, config: Rby1LerobotRemotePolicyCfg):
        self.cfg = config
        super().__init__(config, Rby1Adapter(config.fmt), config.open_loop_horizon)
        meta = self._websocket_client.get_server_metadata()
        self.server_meta = {(k.decode() if isinstance(k, bytes) else k): v for k, v in meta.items()}
        sd = int(self.server_meta.get("action_dim", self._adapter.action_dim))
        assert sd == self._adapter.action_dim, f"server action_dim {sd} != client fmt {config.fmt}"
        self.infer_ms = []

    def set_task_description(self, text: str | None) -> None:
        self.task_description = text or "do the task"

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        super().reset(env_ids)
        if getattr(self, "_websocket_client", None) is not None:
            self._call_server_with_retry({"reset": True})

    def get_action(self, env, observation):
        a = super().get_action(env, observation)
        if self._adapter.fmt == "rby1":  # pad 16 -> 24 for rby1_joint: hold torso, base still
            U = env.unwrapped
            torso = observation["policy"]["torso"].to(a.device)
            a = torch.cat([a, torso, torch.zeros(U.num_envs, 2, device=a.device)], dim=-1)
        return a
