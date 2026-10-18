"""Scripted oracle for BED-02 Easy (fold a towel on the desk) on `rby1_ik`, both arms - privileged (reads nodes).

Robot starts in front of the desk (facing -x); the towel lies 25 cm in from the desk front, long side along the
robot's lateral axis. With the gripper pointing down each arm only reaches its own half of the desk at desk height
(reports/rby1_check/t5/reach_desk_*.json), and a single 2.5 cm pinch in the middle of an edge crumples the flap into a
cone that springs back, while pinching both corners of an edge lays a flat fold (tools/cloth_spike/towel_fold.py,
reports/towel_fold). The fold order is therefore:
1. drive to the desk front (base front 3 cm from the edge, towel centred), torso lean,
2. "fold in half": both hands pinch the far long edge at its two corners and lay it on the near long edge (one arc;
   folding away from the robot left the strip ~0.7 m ahead, out of reach for the side folds),
3. "fold the right side" / "fold the left side": each hand pinches its end of the folded strip (both layers) and lays
   it 3 cm past the centre line on a 10 cm arc (higher arcs open the half fold again),
4. "place on the stack": left hand carries the folded towel to the stack spot (desk front, left of centre),
5. arms home, torso upright.
In the robot-free test this order gave 4/4 side folds and 2/4 all three folds (towel E 1e5, D-28). The towel frame
comes from a PCA of its nodes. `policy.subtask` follows the BED-02 subtask labels.
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
ARM = {"right": (0, 12), "left": (6, 13)}  # action offset of the 6-D arm delta, gripper index


@dataclass
class Bed02OraclePolicyCfg(PolicyCfg):
    edge_margin: float = 0.03     # base front to desk edge (the half fold pinches the far edge ~0.74 m ahead)
    above_dz: float = 0.06
    arc_h: float = 0.10           # fold arc height
    max_v: float = 0.25
    max_w: float = 0.6
    work_preset_steps: int = 1    # lean: crouch cannot reach z 0.8-0.9 near the centre line at x 0.6 (reach_desk_0.0.json)
    debug: bool = False           # print per-arm base-frame target / TCP / error every 15 steps
    device: str = "cuda:0"


@register_policy
class Bed02OraclePolicy(PolicyBase[Bed02OraclePolicyCfg]):
    name = "dx_bed02_oracle"

    def __init__(self, config: Bed02OraclePolicyCfg):
        super().__init__(config)
        self.cfg = config
        self.debug = bool(config.debug)
        self._h = Bed03OraclePolicy.__new__(Bed03OraclePolicy)  # reuse the geometry / nav helpers
        self._h.cfg = config
        self.reset()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        self.phase = "init"
        self.t_phase = 0
        self.subtask = "pick up a shirt"
        self.goal = None
        self.target = None
        self.moves: list[str] = []
        self.mv = None
        self.plan_log: list[dict] = []
        self.last_err = None
        self.arc_s = 0.0
        self.axes = None

    def _set(self, phase, subtask=None):
        self.phase, self.t_phase = phase, 0
        if subtask:
            self.subtask = subtask

    # ------------------------------------------------------------------ helpers
    def _servo(self, obs, target_b, side, kp=0.6, max_dp=0.025):
        """Relative IK step of one arm towards a base-frame TCP position, gripper pointing down."""
        tcp = obs["policy"][f"tcp_{side}"][0].cpu().numpy()
        d = np.asarray(target_b) - tcp[:3]
        dp = np.clip(kp * d, -max_dp, max_dp)
        Re = np.eye(3) @ self._h._qmat(tcp[3:]).T
        ang = math.acos(np.clip((np.trace(Re) - 1) / 2, -1, 1))
        dr = np.zeros(3)
        if ang > 1e-4:
            ax = np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]]) / (2 * math.sin(ang))
            dr = np.clip(ax * ang * kp, -0.08, 0.08)
        return np.r_[dp, dr], float(np.linalg.norm(d))

    def _towel(self, U):
        n = (_T(U.scene[self.target].data.nodal_pos_w)[0] - U.scene.env_origins[0]).cpu().numpy()
        c = n.mean(axis=0)
        xy = n[:, :2] - c[:2]
        if self.axes is None:  # towel frame from the flat towel; kept for later moves (a folded strip's PCA can flip)
            _, V = np.linalg.eigh(xy.T @ xy)
            u, v = V[:, 1], V[:, 0]  # long, short axis
            p, yaw = self._h._base(U)
            right = np.array([math.sin(yaw), -math.cos(yaw)])  # base -y in the world
            fwd = np.array([math.cos(yaw), math.sin(yaw)])
            if np.dot(u, right) < 0:
                u = -u
            if np.dot(v, fwd) < 0:
                v = -v
            if self.phase != "approach":  # the robot has arrived: freeze the frame
                self.axes = (u, v)
        else:
            u, v = self.axes
        return n, c, u, v, xy @ u, xy @ v

    def _plan(self, U, move, task):
        """Per-arm pick / put points (world xyz) of one fold move. All arms of a move ride one shared arc."""
        n, c, u, v, pu, pv = self._towel(U)
        u0, u1, v0, v1 = float(pu.min()), float(pu.max()), float(pv.min()), float(pv.max())

        def local_top(xy, r=0.04):  # cloth surface height around xy (the global max may be a crumple elsewhere)
            d = np.linalg.norm(n[:, :2] - np.asarray(xy)[:2], axis=1)
            return float(n[d < r, 2].max()) if (d < r).any() else self.table_z

        def end_v(sign):  # centre (along v) of the strip end on the right (+1) / left (-1)
            m = (pu > u1 - 0.08) if sign > 0 else (pu < u0 + 0.08)
            return float(pv[m].mean()) if m.any() else 0.0

        if move == "half2":   # far long edge (v1) by both corners -> near long edge (v0): the strip ends up close
            label = "fold in half"  # to the robot, where both hands reach its ends for the side folds
            arms = [(s, c[:2] + u * uu + v * (v1 - 0.04), c[:2] + u * uu + v * (v0 + 0.04))
                    for s, uu in (("right", u1 - 0.04), ("left", u0 + 0.04))]
        elif move == "right":  # right end of the strip -> 3 cm past the centre line
            label, ve = "fold the right side", end_v(+1)
            arms = [("right", c[:2] + u * (u1 - 0.04) + v * ve, c[:2] - u * 0.03 + v * ve)]
        elif move == "left":
            label, ve = "fold the left side", end_v(-1)
            arms = [("left", c[:2] + u * (u0 + 0.04) + v * ve, c[:2] + u * 0.03 + v * ve)]
        else:                  # folded towel -> stack spot, left hand
            label = "place on the stack"
            arms = [("left", c[:2], np.asarray(task.stack_xy, dtype=float))]
        out = []
        for side, pick, put in arms:
            z_put = (self.table_z + 0.05 if move == "stack" else local_top(put) + 0.012) + GRIP_DZ
            out.append({"side": side, "pick": np.r_[pick, local_top(pick) + GRIP_DZ], "put": np.r_[put, z_put]})
        return {"move": move, "label": label, "arms": out}

    # ------------------------------------------------------------------ main
    def get_action(self, env: gym.Env, observation) -> torch.Tensor:
        from dxbench.tasks import bed_tasks

        U = env.unwrapped
        h = self._h
        a = np.zeros(17, dtype=np.float32)
        a[12:14] = 1.0
        self.t_phase += 1
        tidx = int(U.action_manager.get_term("torso").preset_index[0])

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
                _, c, *_ = self._towel(U)
                bx = float(desk[1][0]) + BASE_FRONT_X + self.cfg.edge_margin  # robot on the +x side, facing -x
                self.goal = (np.array([bx, float(c[1])]), math.pi)
                self.table_z = float(desk[1][2])
            (v, w), ok = h._nav(U, self.goal[0], self.goal[1])
            a[15:17] = (v, w)
            if ok or self.t_phase > 300:
                self.goal = None
                self._set("crouch", "flatten the shirt")
        elif self.phase == "crouch":
            a[14] = torso_to(self.cfg.work_preset_steps)
            if tidx == self.cfg.work_preset_steps and self.t_phase > 60:
                self.moves = ["half2", "right", "left", "stack"]
                self._set("plan")
        elif self.phase == "plan":
            if not self.moves:
                self._set("retract")
            else:
                self.mv = self._plan(U, self.moves.pop(0), task)
                self.plan_log.append({"move": self.mv["move"], "arms": [
                    {k: (x.tolist() if hasattr(x, "tolist") else x) for k, x in arm.items()} for arm in self.mv["arms"]]})
                self._set("above", self.mv["label"])
        elif self.phase in ("above", "descend", "close", "lift", "carry", "release"):
            arms = self.mv["arms"]
            grip = 0.0 if self.phase in ("close", "lift", "carry") else 1.0
            targets = {}
            if self.phase == "carry":  # shared fold arc: rises to arc_h and comes down past the line
                D = max(float(np.linalg.norm(arm["put"][:2] - arm["pick"][:2])) for arm in arms)
                # ~0.8 cm of path per step; half speed while a hand lags > 2.5 cm, wait while it lags > 4 cm (the arms
                # trail the arc by up to 15 cm at 1.2 cm per step with the towel pinned)
                e_ = self.last_err if self.last_err is not None else 0.0
                rate = 0.008 if e_ < 0.025 else (0.004 if e_ < 0.07 else 0.0)
                self.arc_s = min(1.0, self.arc_s + rate / max(D + 2 * self.cfg.arc_h, 0.05))
            for arm in arms:
                pick, put = arm["pick"], arm["put"]
                if self.phase == "above":
                    tgt, mdp = pick + [0, 0, self.cfg.above_dz], 0.025
                elif self.phase in ("descend", "close"):
                    tgt, mdp = pick, 0.012
                elif self.phase == "lift":  # short vertical lift so the pinched edge leaves the desk
                    tgt, mdp = pick + [0, 0, 0.03], 0.02
                elif self.phase == "carry":
                    s_ = self.arc_s
                    tgt = pick + (put - pick) * s_
                    tgt[2] = pick[2] + 0.03 + (put[2] - pick[2] - 0.03) * s_ + self.cfg.arc_h * math.sin(math.pi * s_)
                    mdp = 0.02
                else:  # release: open in place first (pins drop when the fingers open), then rise 12 cm and stay
                    tgt, mdp = put + [0, 0, 0.0 if self.t_phase <= 8 else 0.12], 0.015
                targets[arm["side"]] = (tgt, mdp)
            errs = []
            for side, (tgt, mdp) in targets.items():
                off, gi = ARM[side]
                tb = h._to_base(U, tgt)
                a[off:off + 6], e = self._servo(observation, tb, side, max_dp=mdp)
                a[gi] = grip
                errs.append(e)
                if self.debug and self.t_phase % 15 == 1:
                    tcp = observation["policy"][f"tcp_{side}"][0, :3].cpu().numpy()
                    print(f"[bed02dbg] {self.mv['move']} {self.phase} t={self.t_phase} {side} tgt_b={np.round(tb, 3)} "
                          f"tcp_b={np.round(tcp, 3)} err={e:.3f} s={self.arc_s:.2f}", flush=True)
            err = max(errs)
            self.last_err = err
            st = self.plan_log[-1].setdefault("min_err", {})
            st[self.phase] = round(min(st.get(self.phase, 9.0), err), 4)
            if self.phase == "above" and (err < 0.015 or self.t_phase > 90):
                self._set("descend")
            elif self.phase == "descend" and (err < 0.01 or self.t_phase > 60):
                self._set("close")
            elif self.phase == "close" and self.t_phase > 10:
                term = U.action_manager.get_term("cloth_assist")
                self.plan_log[-1]["grasped"] = {s: int(term.grasped[0, 0 if s == "right" else 1]) for s in targets}
                self._set("lift")
            elif self.phase == "lift" and (err < 0.015 or self.t_phase > 30):
                self.arc_s = 0.0
                self._set("carry")
            elif self.phase == "carry" and ((self.arc_s >= 1.0 and err < 0.03) or self.t_phase > 450):
                self._set("release")
            elif self.phase == "release" and self.t_phase > 30:
                # no joint homing between moves: from the READY pose with the lean torso the relative IK drifted away
                # from desk targets (v5 / v9); the next move starts from this raised pose instead
                self._set("plan")
        elif self.phase == "rehome":
            homing = [bool(U.action_manager.get_term(f"{arm['side']}_arm").homing[0]) for arm in self.mv["arms"]]
            if (not any(homing) and self.t_phase > 5) or self.t_phase > 200:
                self._set("plan")
        elif self.phase == "retract":
            a[14] = torso_to(0)
            if tidx == 0 and self.t_phase > 60:
                self._set("finished")
        return torch.as_tensor(a, device=U.device).unsqueeze(0).repeat(U.num_envs, 1)
