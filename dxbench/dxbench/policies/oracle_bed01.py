"""Scripted oracle for BED-01 Easy (D-55: the duvet slid 60-66 cm towards the far side) on `rby1_ik` - privileged.

PROTOTYPE, NOT WORKING (2026-10-29): 0/2 easy scenes. The pick lands in the reachable window (0.81 m ahead) but the
relative IK from the home pose ends 13 cm away (joint limits on the way; the reach diagnostic got there only after
visiting nearer targets), and the base misses the side spots by 0.2-0.35 m (yaw-0 navigation to a lateral goal).

The near edge of the duvet lies 0.34-0.40 m in from the open side of the mattress. In the running env (reach_diag,
2026-10-28, base front 6 cm from the bed frame, gripper down, z 0.56) deep reaches 0.79-0.84 m ahead of the base:
nearer, the wrist (arm_5) and elbow (arm_3) sit at their limits; farther, the arm / torso touches the bed. Crouch
covers up to ~0.77 m. So the edge (0.77-0.83 m ahead) is just reachable, and the arm cannot pull back in deep:
the base reverses instead while the arm holds. Each pull:
1. drive to the bed side so the right hand (base y -0.20) is in front of the pull spot, torso deep,
2. pinch the duvet's near edge 4 cm in (pad grasp: fingertips 2 mm into the cloth),
3. lift 3 cm and reverse the base so the edge ends beyond the near side of the mattress (the made-up overhang),
4. release, arm home, torso upright.
Spots along the bed: the middle of the target area, then 45 cm towards the head, then towards the foot, until the task
reports success. Single arm (the relative IK servo of the other oracles drives the right arm).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg

from dxbench.policies.oracle_bed03 import BASE_FRONT_X, Bed03OraclePolicy, pinch_dz

PRESET_DEEP = 3


@dataclass
class Bed01EasyOracleCfg(PolicyCfg):
    hand_y: float = -0.20            # right hand lateral offset in the base frame
    frame_gap: float = 0.06          # base front to the bed frame
    edge_in: float = 0.01            # pinch this far in from the duvet's near edge
    crouch_max_x: float = 0.77       # picks nearer than this (base frame) in crouch, farther in deep
    overhang: float = 0.25           # the near edge should end this far beyond the mattress side
    reverse_v: float = 0.12          # base speed while pulling back (m/s)
    spots: tuple = (0.0, 0.45, -0.45)  # pull spots along the bed (m from the target-area middle, + = head)
    max_v: float = 0.2
    max_w: float = 0.6
    device: str = "cuda:0"


@register_policy
class Bed01EasyOracle(Bed03OraclePolicy, PolicyBase[Bed01EasyOracleCfg]):  # Arena: PolicyBase[Cfg] as a direct base
    name = "dx_bed01_easy_oracle"

    def __init__(self, config: Bed01EasyOracleCfg):
        PolicyBase.__init__(self, config)
        self.cfg = config
        self.reset()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        super().reset(env_ids)
        self.phase, self.t_phase, self.subtask = "plan", 0, "grasp the duvet edge"
        self.spot_i = 0
        self.goal = None
        self.pick = None
        self.plan_log = []

    def _bed(self):
        from dxbench.tasks import bed_tasks

        b = bed_tasks.CURRENT_TASK.bed
        return b, bed_tasks.CURRENT_TASK

    def get_action(self, env: gym.Env, observation) -> torch.Tensor:
        U = env.unwrapped
        c = self.cfg
        a = np.zeros(17, dtype=np.float32)
        a[12:14] = 1.0
        self.t_phase += 1
        idx = int(U.action_manager.get_term("torso").preset_index[0])
        work = getattr(self, "work", PRESET_DEEP)
        want = work if self.phase in ("deep", "above", "descend", "close", "pull", "release") else 0
        if idx != want and self.phase not in ("plan", "approach", "finished"):
            a[14] = (1.0 if idx < want else -1.0) if self.t_phase % 2 == 1 else 0.0
        tcp = self._tcp_b(observation)
        bed, task = self._bed()
        lo, hi = bed.mattress_min, bed.mattress_max
        if self.phase == "plan":
            if self.spot_i >= len(c.spots):
                self._set("finished")
            else:
                y_mid = 0.5 * (lo[1] + hi[1] - bed.cover_head_gap)
                self.spot_y = y_mid + c.spots[self.spot_i]
                bx = lo[0] - 0.025 - c.frame_gap - BASE_FRONT_X
                self.goal = (np.array([bx, self.spot_y - c.hand_y]), 0.0)
                self.plan_log.append({"spot": self.spot_i, "spot_y": round(self.spot_y, 3), "goal": self.goal[0].tolist()})
                self._set("approach", "grasp the duvet edge")
        elif self.phase == "approach":
            (v, w), ok = self._nav(U, self.goal[0], self.goal[1])
            a[15:17] = (v, w)
            if ok or self.t_phase > 200:
                self._set("deep")
        elif self.phase == "deep":
            if idx == getattr(self, "work", PRESET_DEEP) and self.t_phase > 30 and not getattr(self, "picked", False):
                _, n = self._cloth_xyz(U, "duvet")
                top = bed.mattress_top_z
                m = (np.abs(n[:, 1] - self.spot_y) < 0.05) & (n[:, 2] > top - 0.005) & (n[:, 0] > lo[0] + 0.02)
                if not m.any():
                    self.plan_log.append({"spot": self.spot_i, "no_edge": True})
                    self.spot_i += 1
                    self._set("rise")
                else:
                    j = np.nonzero(m)[0][np.argmin(n[m, 0])]
                    p = n[j].copy()
                    p[0] += c.edge_in
                    near = np.linalg.norm(n[:, :2] - p[:2], axis=1) < 0.03
                    p[2] = float(n[near, 2].max()) if near.any() else p[2]
                    self.edge_x = float(n[j, 0])
                    self.pick = self._to_base(U, p)
                    self.work = 2 if self.pick[0] < c.crouch_max_x else PRESET_DEEP
                    self.picked = True
                    self.plan_log.append({"spot": self.spot_i, "edge_x": round(self.edge_x, 3), "preset": self.work,
                                          "pick_base": np.round(self.pick, 3).tolist()})
            elif getattr(self, "picked", False) and idx == self.work and self.t_phase > 30:
                self.picked = False
                self._set("above")
        elif self.phase == "above":
            a[0:6], err = self._servo(observation, self.pick + [0, 0, 0.08])
            if err < 0.015 or self.t_phase > 90:
                self._set("descend")
        elif self.phase == "descend":
            _, n = self._cloth_xyz(U, "duvet")
            nb = np.array([self._to_base(U, q) for q in n])
            near = np.linalg.norm(nb[:, :2] - self.pick[:2], axis=1) < 0.03
            if near.any():
                self.pick[2] = float(nb[near, 2].max())
            a[0:6], err = self._servo(observation, self.pick + [0, 0, pinch_dz(U) - 0.003], max_dp=0.012)
            if err < 0.006 or self.t_phase > 90:
                self._set("close")
        elif self.phase in ("close", "pull"):
            a[12] = 0.0
            if self.phase == "close":
                a[0:6], _ = self._servo(observation, tcp)
                if self.t_phase > 15:
                    got = int(U.action_manager.get_term("cloth_assist").grasped[0, 0])
                    self.plan_log.append({"spot": self.spot_i, "grasped": got,
                                          "tip_err": round(float(np.linalg.norm(tcp - self.pick)), 4)})
                    if got == 0:
                        self.spot_i += 1
                        self._set("release")
                    else:
                        # total edge travel: from its x to `overhang` beyond the mattress side; the arm gives part
                        p, _ = self._base(U)
                        travel = self.edge_x - (lo[0] - c.overhang)
                        self.reverse_s = max(0.0, travel) / c.reverse_v
                        self.start = tcp.copy()
                        self._set("pull", "pull the duvet")
            else:
                n_arm = 45
                s = min(1.0, self.t_phase / n_arm)
                tgt = self.start + s * np.array([0.0, 0.0, 0.03])
                a[0:6], _ = self._servo(observation, tgt, max_dp=0.015)
                if self.t_phase * (1 / 15) < self.reverse_s:
                    a[15] = -c.reverse_v
                if self.t_phase > n_arm and self.t_phase * (1 / 15) >= self.reverse_s + 0.5:
                    self.plan_log.append({"spot": self.spot_i, "pulled_s": round(self.reverse_s, 2)})
                    self.spot_i += 1
                    self._set("release")
        elif self.phase == "release":
            a[0:6], _ = self._servo(observation, tcp + [0, 0, 0.01])
            if self.t_phase > 20:
                U.action_manager.get_term("right_arm").request_home()
                self._set("rise")
        elif self.phase == "rise":
            self.work = PRESET_DEEP
            if idx == 0 and self.t_phase > 45:
                done = task.evaluate_cached(U)[0]
                self.plan_log.append({"after_spot": self.spot_i, "coverage": round(float(done["coverage"]), 3)})
                self._set("finished" if done["success"] and done["stages"].get("duvet_neat") else "plan")
        return torch.as_tensor(a, device=U.device).unsqueeze(0).repeat(U.num_envs, 1)
