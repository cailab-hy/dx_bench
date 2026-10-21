"""BED-01 grasp demo (not an oracle, 2026-10-21): shows the pad-box grasp with the support ring on the real-size
duvet. From its start pose at the bed side the robot leans (crouch), pinches the top layer of the duvet with the
right hand near its own right, lifts 8 cm, pulls 25 cm towards the headboard (base +y side = the robot's left),
holds, releases and straightens. No navigation, no pillow, no task completion - BED-01 demos come from VR teleop.
"""
from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg

from dxbench.policies.oracle_bed03 import Bed03OraclePolicy, pinch_dz


@dataclass
class Bed01GraspDemoCfg(PolicyCfg):
    reach_x: tuple = (0.50, 0.66)    # base-frame window where the right arm reaches the mattress top in crouch
    reach_y: tuple = (-0.32, -0.08)
    pull: tuple = (-0.05, 0.25, 0.08)  # hand travel after the pinch (base frame): back, towards the head, up
    pull_steps: int = 75             # duration of the pull (15 Hz steps)
    preset_steps: int = 2            # crouch
    max_v: float = 0.2               # used by the inherited _nav
    max_w: float = 0.6
    device: str = "cuda:0"


@register_policy
class Bed01GraspDemo(Bed03OraclePolicy, PolicyBase[Bed01GraspDemoCfg]):  # Arena: PolicyBase[Cfg] as a direct base
    name = "dx_bed01_grasp_demo"

    def __init__(self, config: Bed01GraspDemoCfg):
        PolicyBase.__init__(self, config)
        self.cfg = config
        self.reset()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        super().reset(env_ids)
        self.phase, self.t_phase, self.subtask = "approach", 0, "grasp the top corners"
        self.goal = None
        self.pick = None
        self.layout = None
        self.plan_log = []

    def get_action(self, env: gym.Env, observation) -> torch.Tensor:
        U = env.unwrapped
        a = np.zeros(17, dtype=np.float32)
        a[12:14] = 1.0
        self.t_phase += 1
        idx = int(U.action_manager.get_term("torso").preset_index[0])
        if idx < self.cfg.preset_steps and self.phase in ("lean",):
            a[14] = 1.0 if self.t_phase % 2 == 1 else 0.0
        elif idx > 0 and self.phase == "rise":
            a[14] = -1.0 if self.t_phase % 2 == 1 else 0.0
        tcp = self._tcp_b(observation)
        if self.phase == "approach":
            # the scenes start 0.9 m from the mattress: drive straight in until the base front is 6 cm from the frame
            if self.goal is None:
                from dxbench.policies.oracle_bed03 import BASE_FRONT_X
                from dxbench.tasks.scene_spec import load_layout

                lo, _hi = load_layout()["parts_aabb"]["Bed/Mattress"]
                p, yaw = self._base(U)
                self.goal = (np.array([lo[0] - 0.025 - 0.06 - BASE_FRONT_X, p[1]]), yaw)
                self.plan_log.append({"approach_goal": self.goal[0].tolist()})
            (v, w), reached = self._nav(U, self.goal[0], self.goal[1])
            a[15:17] = (v, w)
            if reached or self.t_phase > 150:
                self._set("lean")
        elif self.phase == "lean":
            if idx == self.cfg.preset_steps and self.t_phase > 60:
                _, n = self._cloth_xyz(U, "duvet")
                nb = np.array([self._to_base(U, p) for p in n])
                m = ((nb[:, 0] > self.cfg.reach_x[0]) & (nb[:, 0] < self.cfg.reach_x[1])
                     & (nb[:, 1] > self.cfg.reach_y[0]) & (nb[:, 1] < self.cfg.reach_y[1]))
                k = np.nonzero(m)[0]
                # the top layer: the highest node in the window (the folded half lies on the other)
                j = k[np.argmax(nb[k, 2])] if len(k) else int(np.argmin(np.linalg.norm(nb[:, :2] - [0.58, -0.2], axis=1)))
                self.pick = nb[j].copy()
                self.plan_log.append({"pick_base": self.pick.tolist()})
                self._set("above")
        elif self.phase == "above":
            a[0:6], err = self._servo(observation, self.pick + [0, 0, 0.08])
            if err < 0.015 or self.t_phase > 90:
                self._set("descend")
        elif self.phase == "descend":
            a[0:6], err = self._servo(observation, self.pick + [0, 0, pinch_dz(U) - self.pick[2] * 0 - 0.003],
                                      max_dp=0.012)
            if err < 0.006 or self.t_phase > 90:
                self._set("close")
        elif self.phase in ("close", "pull", "hold"):
            a[12] = 0.0
            if self.phase == "close":
                a[0:6], _ = self._servo(observation, tcp)
                if self.t_phase > 15:
                    got = int(U.action_manager.get_term("cloth_assist").grasped[0, 0])
                    self.plan_log.append({"grasped": got, "diag": U.action_manager.get_term("cloth_assist").last_capture.get((0, 0))})
                    self.start = tcp.copy()
                    self._set("pull", "pull toward the headboard")
            elif self.phase == "pull":
                s = min(1.0, self.t_phase / float(self.cfg.pull_steps))
                a[0:6], _ = self._servo(observation, self.start + s * np.asarray(self.cfg.pull), max_dp=0.012)
                if self.t_phase > self.cfg.pull_steps + 15:
                    self._set("hold")
            else:
                a[0:6], _ = self._servo(observation, self.start + np.asarray(self.cfg.pull))
                if self.t_phase > 22:
                    self._set("release")
        elif self.phase == "release":
            a[0:6], _ = self._servo(observation, tcp + [0, 0, 0.01])
            if self.t_phase > 30:
                U.action_manager.get_term("right_arm").request_home()
                self._set("rise")
        elif self.phase == "rise":
            if idx == 0 and self.t_phase > 60:
                self._set("finished")
        return torch.as_tensor(a, device=U.device).unsqueeze(0).repeat(U.num_envs, 1)
