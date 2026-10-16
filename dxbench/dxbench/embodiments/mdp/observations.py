"""Observation terms for the RB-Y1 embodiments (torch, Isaac Lab 3.0; quaternions (x, y, z, w))."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from dxbench.embodiments.rby1_cfg import ARM_L, ARM_R, TORSO

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

TCP_OFFSET = (0.0, 0.0, -0.1045)  # in the ee_{side} frame (0-stage report)
FINGER_MAX = 0.05

_cache: dict[tuple, list[int]] = {}


def _jids(env: ManagerBasedEnv, names: tuple[str, ...]) -> list[int]:
    key = (id(env), names)
    if key not in _cache:
        r = env.scene["robot"]
        ids, _ = r.find_joints(list(names), preserve_order=True)
        _cache[key] = list(ids.tolist() if hasattr(ids, "tolist") else ids)
    return _cache[key]


def joint_pos_named(env: ManagerBasedEnv, names: tuple[str, ...]) -> torch.Tensor:
    return env.scene["robot"].data.joint_pos.torch[:, _jids(env, names)]


def joint_vel_named(env: ManagerBasedEnv, names: tuple[str, ...]) -> torch.Tensor:
    return env.scene["robot"].data.joint_vel.torch[:, _jids(env, names)]


def gripper_opening(env: ManagerBasedEnv) -> torch.Tensor:
    """[right, left] in [0, 1]: half the finger gap / 0.05 m (1 = fully open)."""
    q = env.scene["robot"].data.joint_pos.torch
    r1, r2, l1, l2 = _jids(env, ("gripper_finger_r1", "gripper_finger_r2", "gripper_finger_l1", "gripper_finger_l2"))
    return torch.stack([(q[:, r2] - q[:, r1]) / 2, (q[:, l2] - q[:, l1]) / 2], dim=1).clamp(0, FINGER_MAX) / FINGER_MAX


def state16(env: ManagerBasedEnv) -> torch.Tensor:
    """Real RB-Y1 16-D layout (rby1_format): right arm 7, left arm 7, right gripper, left gripper (1 = open)."""
    return torch.cat([joint_pos_named(env, tuple(ARM_R + ARM_L)), gripper_opening(env)], dim=1)


def torso_pos(env: ManagerBasedEnv) -> torch.Tensor:
    return joint_pos_named(env, tuple(TORSO))


def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:  # q (N,4) xyzw
    xyz, w = q[:, :3], q[:, 3:4]
    t = 2 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def _quat_conj(q):
    return torch.cat([-q[:, :3], q[:, 3:4]], dim=1)


def _quat_mul(a, b):  # xyzw
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz], dim=-1)


def base_pose_w(env: ManagerBasedEnv) -> tuple[torch.Tensor, torch.Tensor]:
    r = env.scene["robot"]
    return r.data.root_link_pos_w.torch, r.data.root_link_quat_w.torch


def tcp_pose_b(env: ManagerBasedEnv, side: str = "right") -> torch.Tensor:
    """TCP pose in the base frame: (x, y, z, qx, qy, qz, qw)."""
    r = env.scene["robot"]
    bi = r.body_names.index(f"ee_{side}")
    p = r.data.body_link_pos_w.torch[:, bi]
    q = r.data.body_link_quat_w.torch[:, bi]
    off = torch.tensor([TCP_OFFSET], device=p.device).expand(p.shape[0], 3)
    p_tcp = p + _quat_apply(q, off)
    bp, bq = base_pose_w(env)
    bqi = _quat_conj(bq)
    return torch.cat([_quat_apply(bqi, p_tcp - bp), _quat_mul(bqi, q)], dim=1)


def base_velocity(env: ManagerBasedEnv) -> torch.Tensor:
    """[forward speed (m/s), yaw rate (rad/s)] of the base in its own frame."""
    r = env.scene["robot"]
    return torch.stack([r.data.root_link_lin_vel_b.torch[:, 0], r.data.root_link_ang_vel_b.torch[:, 2]], dim=1)


def base_pose_2d(env: ManagerBasedEnv) -> torch.Tensor:
    """[x, y, yaw] in the env frame (for recording / navigation predicates)."""
    bp, bq = base_pose_w(env)
    x, y, z, w = bq.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return torch.stack([bp[:, 0] - env.scene.env_origins[:, 0], bp[:, 1] - env.scene.env_origins[:, 1], yaw], dim=1)
