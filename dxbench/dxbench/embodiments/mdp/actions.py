"""Custom action terms for the RB-Y1 embodiments.

All terms are torch based (Isaac Lab 3.0 action manager). `apply_actions` runs every physics step.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from dxbench.embodiments.rby1_cfg import BODY, TORSO_PRESETS

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

WHEEL_RADIUS = 0.1
WHEEL_TRACK = 0.53
FINGER_MAX = 0.05


class _RobotTerm(ActionTerm):
    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._raw = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed = torch.zeros_like(self._raw)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed

    def _ids(self, names: list[str]) -> list[int]:
        ids, _ = self._asset.find_joints(names, preserve_order=True)
        return list(ids.tolist() if hasattr(ids, "tolist") else ids)


# ------------------------------------------------------------------------------------------ gravity
class GravityCompensationAction(_RobotTerm):
    """No action dimensions: every physics step adds the gravity torques of the body joints as effort targets
    (the rby1-sdk feed-forward term; preset "tuned")."""

    cfg: GravityCompensationActionCfg

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._body = self._ids(cfg.joint_names)
        self._ff = torch.zeros(self.num_envs, self._asset.num_joints, device=self.device)

    @property
    def action_dim(self) -> int:
        return 0

    def process_actions(self, actions: torch.Tensor):
        pass

    def apply_actions(self):
        g = self._asset.data.gravity_compensation_forces.torch
        g = g[:, g.shape[-1] - self._asset.num_joints:]
        self._ff.zero_()
        self._ff[:, self._body] = g[:, self._body] * self.cfg.scale
        self._asset.set_joint_effort_target_index(target=self._ff)


@configclass
class GravityCompensationActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = GravityCompensationAction
    joint_names: list[str] = BODY
    scale: float = 1.0


# ------------------------------------------------------------------------------------------ gripper
class ParallelGripperAction(_RobotTerm):
    """[right, left] opening in [0, 1] (1 = open, rby1_format convention). Each finger pair is driven with
    mirrored targets (l1 = -0.05*g, l2 = +0.05*g): the software mimic of DECISIONS D-3."""

    cfg: ParallelGripperActionCfg

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._f = self._ids(["gripper_finger_r1", "gripper_finger_r2", "gripper_finger_l1", "gripper_finger_l2"])

    @property
    def action_dim(self) -> int:
        return 2

    def process_actions(self, actions: torch.Tensor):
        self._raw[:] = actions
        g = actions.clamp(0.0, 1.0)
        if self.cfg.binary_threshold is not None:
            g = (g > self.cfg.binary_threshold).float()
        o = g * FINGER_MAX
        self._processed = torch.stack([-o[:, 0], o[:, 0], -o[:, 1], o[:, 1]], dim=1)

    def apply_actions(self):
        self._asset.set_joint_position_target_index(target=self._processed, joint_ids=self._f)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw[env_ids] = 1.0


@configclass
class ParallelGripperActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = ParallelGripperAction
    binary_threshold: float | None = None


# ------------------------------------------------------------------------------------------ base
class DiffDriveBaseAction(_RobotTerm):
    """Base (v [m/s], omega [rad/s]) -> wheel velocity targets with an outer wheel-angle loop.

    Wheel joint frames are flipped: +wheel velocity drives backward (0-stage T3e), so the commanded wheel rates are
    -(v -/+ omega*W/2)/R for right/left. The casters are frictionless (D-U3), so a damping-only velocity drive lets
    the base creep under any load; like the original controller (integrated position target + kp/kd) the commanded
    wheel angle is integrated and the velocity target gets `k_pos * (angle_cmd - angle_meas)`, with the measured
    angle unwrapped every physics step (PhysX reports unlimited revolute joints wrapped)."""

    cfg: DiffDriveBaseActionCfg

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._w = self._ids(["right_wheel", "left_wheel"])
        self._rate = torch.zeros(self.num_envs, 2, device=self.device)
        self._cmd_angle = torch.zeros(self.num_envs, 2, device=self.device)
        self._meas_angle = torch.zeros(self.num_envs, 2, device=self.device)
        self._prev_raw = None
        self._dt = env.physics_dt

    @property
    def action_dim(self) -> int:
        return 2

    def process_actions(self, actions: torch.Tensor):
        self._raw[:] = actions
        v = actions[:, 0].clamp(-self.cfg.max_v, self.cfg.max_v)
        w = actions[:, 1].clamp(-self.cfg.max_w, self.cfg.max_w)
        vr = v + w * WHEEL_TRACK / 2  # right wheel sits at -y: CCW yaw speeds it up
        vl = v - w * WHEEL_TRACK / 2
        self._rate = torch.stack([-vr / WHEEL_RADIUS, -vl / WHEEL_RADIUS], dim=1)

    def _update_meas(self):
        q = self._asset.data.joint_pos.torch[:, self._w]
        if self._prev_raw is None:
            self._prev_raw = q.clone()
        self._meas_angle += wrap_to_pi(q - self._prev_raw)
        self._prev_raw = q.clone()

    def apply_actions(self):
        self._update_meas()
        self._cmd_angle += self._rate * self._dt
        err = (self._cmd_angle - self._meas_angle).clamp(-self.cfg.max_err, self.cfg.max_err)
        self._cmd_angle = self._meas_angle + err  # anti wind-up when the wheels are blocked
        self._processed = self._rate + self.cfg.k_pos * err
        self._asset.set_joint_velocity_target_index(target=self._processed, joint_ids=self._w)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw[env_ids] = 0.0
        self._processed[env_ids] = 0.0
        self._rate[env_ids] = 0.0
        self._cmd_angle[env_ids] = 0.0
        self._meas_angle[env_ids] = 0.0
        self._prev_raw = None


@configclass
class DiffDriveBaseActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = DiffDriveBaseAction
    max_v: float = 0.5
    max_w: float = 1.0
    k_pos: float = 20.0  # 1/s, outer wheel-angle loop gain
    max_err: float = 0.5  # rad, clamp of the wheel-angle error


# ------------------------------------------------------------------------------------------ torso presets
class TorsoPresetAction(_RobotTerm):
    """Teleop torso control: one scalar; crossing +0.5 / -0.5 (rising edge) steps to the next / previous preset
    of PRESET_ORDER. The torso_1/2/3 targets move there **together** on a straight joint-space line with a cosine
    time profile (peak speed of the largest joint = `ramp_rad_s`); torso_0/4/5 stay at 0.
    Per-joint constant-rate ramps reach intermediate postures such as (90, -90, 90) deg on the way to "deep" whose
    centre of mass is over the front axle and that need >100 % torso torque - the robot tipped (stage 2, t5 torso_ramp)."""

    cfg: TorsoPresetActionCfg
    PRESET_ORDER = ("upright", "lean", "crouch", "deep")

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._t = self._ids([f"torso_{i}" for i in range(6)])
        self._presets = torch.tensor([[0.0, *TORSO_PRESETS[p], 0.0, 0.0] for p in self.PRESET_ORDER],
                                     device=self.device)
        self._idx = torch.full((self.num_envs,), self.PRESET_ORDER.index(cfg.initial), device=self.device,
                               dtype=torch.long)
        self._cmd = self._presets[self._idx].clone()
        self._start = self._cmd.clone()
        self._goal_idx = self._idx.clone()
        self._dur = torch.ones(self.num_envs, device=self.device)
        self._tt = torch.ones(self.num_envs, device=self.device)
        self._prev = torch.zeros(self.num_envs, device=self.device)
        self._dt = env.physics_dt

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def preset_index(self) -> torch.Tensor:
        return self._idx

    def process_actions(self, actions: torch.Tensor):
        self._raw[:] = actions
        a = actions[:, 0]
        up = (a > 0.5) & (self._prev <= 0.5)
        dn = (a < -0.5) & (self._prev >= -0.5)
        self._idx = (self._idx + up.long() - dn.long()).clamp(0, len(self.PRESET_ORDER) - 1)
        self._prev = a.clone()
        # the goal only changes here, so the re-plan runs once per control step (in apply_actions its .any() was a
        # host sync every physics step); _cmd is the last physics step's command either way
        goal = self._presets[self._idx]
        new = self._idx != self._goal_idx
        if new.any():  # re-plan from the current command: duration so the largest joint peaks at ramp_rad_s
            self._start[new] = self._cmd[new]
            span = (goal[new] - self._start[new]).abs().max(dim=-1).values
            self._dur[new] = (span * (math.pi / 2) / self.cfg.ramp_rad_s).clamp(min=1e-3)
            self._tt[new] = 0.0
            self._goal_idx = self._idx.clone()

    def apply_actions(self):
        goal = self._presets[self._idx]
        self._tt = self._tt + self._dt
        s = (self._tt / self._dur).clamp(0.0, 1.0)
        a = (0.5 * (1.0 - torch.cos(math.pi * s))).unsqueeze(-1)
        self._cmd = self._start + (goal - self._start) * a
        self._processed = self._cmd
        self._asset.set_joint_position_target_index(target=self._cmd, joint_ids=self._t)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._idx[env_ids] = self.PRESET_ORDER.index(self.cfg.initial)
        self._cmd[env_ids] = self._presets[self._idx[env_ids]]
        self._start[env_ids] = self._cmd[env_ids]
        self._goal_idx[env_ids] = self._idx[env_ids]
        self._tt[env_ids] = 1.0
        self._dur[env_ids] = 1.0
        self._prev[env_ids] = 0.0


@configclass
class TorsoPresetActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = TorsoPresetAction
    initial: str = "upright"
    ramp_rad_s: float = 0.5   # peak speed of the largest torso joint during a preset change


def wrap_to_pi(a: torch.Tensor) -> torch.Tensor:
    return (a + math.pi) % (2 * math.pi) - math.pi



# ------------------------------------------------------------------------------------------ IK with arm homing
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg  # noqa: E402
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction  # noqa: E402


class HomingDiffIKAction(DifferentialInverseKinematicsAction):
    """Isaac Lab relative DiffIK arm action + a joint-space "home" side channel (not part of the action vector).

    `request_home(env_ids)` makes the arm joint targets move to the default (READY) joint positions at `home_rad_s`
    (largest joint), overriding the IK output until reached; teleop (home key) and the scripted oracles use it.
    The relative IK alone cannot always bring the arm back from far configurations (crouched over the basket):
    it got stuck near joint limits and the next pick failed (stage 3). Recorded joint targets include the homing.
    """

    cfg: "HomingDiffIKActionCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._homing = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._dt = env.physics_dt

    def request_home(self, env_ids=None):
        if env_ids is None:
            self._homing[:] = True
        else:
            self._homing[env_ids] = True

    @property
    def homing(self) -> torch.Tensor:
        return self._homing

    def apply_actions(self):
        super().apply_actions()
        if not self._homing.any():
            return
        q = self._asset.data.joint_pos.torch[:, self._joint_ids]
        q_home = self._asset.data.default_joint_pos.torch[:, self._joint_ids]
        d = q_home - q
        span = d.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-6)
        step = self.cfg.home_rad_s * self._dt * 4.0  # targets lead the joints a little (PD lag)
        tgt = q + d * (step / span).clamp(max=1.0)
        done = span.squeeze(-1) < 0.02
        self._homing &= ~done
        ids = torch.nonzero(self._homing | done, as_tuple=False).squeeze(-1)
        if len(ids):
            # override the IK output for homing envs only
            cur = self._asset.data.joint_pos_target.torch[:, self._joint_ids].clone()
            cur[ids] = tgt[ids]
            self._asset.set_joint_position_target_index(target=cur, joint_ids=self._joint_ids)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if env_ids is None:
            self._homing[:] = False
        else:
            self._homing[env_ids] = False


@configclass
class HomingDiffIKActionCfg(DifferentialInverseKinematicsActionCfg):
    class_type: type[ActionTerm] = HomingDiffIKAction
    home_rad_s: float = 0.8


def hold_joints_at_default(env, env_ids, joint_names, asset_name: str = "robot"):
    """Reset event: set the PD position target of joints no action term drives (the head) to their default
    (initial) position, so they hold the start pose instead of returning to 0."""
    asset = env.scene[asset_name]
    ids = asset.find_joints(list(joint_names), preserve_order=True)[0]
    q = asset.data.default_joint_pos
    q = q.torch if hasattr(q, "torch") else q
    env_ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    asset.set_joint_position_target_index(target=q[env_ids][:, ids], joint_ids=ids, env_ids=env_ids)

