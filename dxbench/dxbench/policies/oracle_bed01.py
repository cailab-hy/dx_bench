"""Scripted oracle for BED-01 Easy (D-55: the duvet slid 60-66 cm towards the far side) on `rby1_ik` - privileged.

Status (2026-10-29, D-57): 9/10 easy evaluation scenes (score 1.0), 27/30 training scenes collected
(data/raw/bed01_oracle_easy). Crouch only - deep's elbow (arm_3) sits at its limit and the hand ran off to 0.92-0.96 m;
side moves go through a waypoint backed off from the bed (turning in place 6 cm from the frame missed by 0.2-0.35 m).

The near edge of the duvet lies 0.29-0.31 m in from the open side of the mattress (D-57: easy slides 57-59 cm). In the
running env (reach_diag, base front 6 cm from the bed frame, gripper down, z 0.57) crouch reaches 0.78 m ahead of the
base (~0.35 m into the mattress); the base reverses to pull while the arm holds. Each pull:
1. drive to the bed side so the right hand (base y -0.20) is in front of the pull spot, torso crouch,
2. pinch the duvet's near edge 4 cm in (pad grasp: fingertips 2 mm into the cloth),
3. lift 3 cm and reverse the base so the edge ends beyond the near side of the mattress (the made-up overhang),
4. release, arm home, torso upright.
Spots along the bed: the middle of the target area, then 45 cm towards the head, then towards the foot, until the task
reports success. D-63 (2026-10-31, user review of the demo video): both hands go to the edge (base y +-hand_y), and
the arms no longer shake on the reach - at the crouch reach the wrist cannot point straight down (35-53 deg off), and
the orientation servo, at its step limit the whole reach, made the joints reverse 0.4 times per step. The reach now
uses a soft orientation servo and aligns the wrist above the edge before going down (`soft_above`, `align_steps`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg

from dxbench.policies.oracle_bed03 import AXLE_X, BASE_FRONT_X, Bed03OraclePolicy, align_step, episode_rng, pinch_dz

PRESET_DEEP = 3


@dataclass
class Bed01EasyOracleCfg(PolicyCfg):
    hand_y: float = -0.20            # right hand lateral offset in the base frame (the left hand at -hand_y)
    two_arms: bool = True            # D-63: both hands pinch the edge
    fixed_torso: bool = True         # D-66: no torso commands (the easy scenes start crouched)
    # D-66 (user): the scene starts with both hands hovering over the bed, grippers straight down (rby1_cfg ARM_PRESETS
    # "bed_reach": TCPs `hand_x` ahead of the base at y -+0.20, z 0.62). The base stops with the hands right above the
    # duvet's near edge, the arms only go down (z), pinch, and the base reverses to pull
    ready_reach: bool = True
    hand_x: float = 0.62
    spot_on_duvet: bool = True       # pull spots relative to the duvet's centre (along the bed), not the target area
    pitch_deg: float | None = 0.0    # gripper orientation target, tilted about the base y axis (0 = straight down; None =
                                     # keep the current orientation: no rotation command)
    rot_gain: float = 0.6            # orientation servo gain
    rot_max: float = 0.08            # orientation step limit (rad per control step)
    # D-63: arm joint velocity reversals per reach step (oracle `jitter` log, 4 scenes): full orientation servo all the
    # way 0.37-0.42 (the shaking in the demo video), soft servo on the long reach + 40 steps aligning above the edge
    # 0.09 (4/4); soft everywhere 0.10 but 3/4, no orientation servo or a 20-45 deg pitch target: no grasp at all
    soft_above: bool = False         # the long reach above the edge with a soft orientation servo (gain 0.2, step 0.02)
    align_steps: int = 0             # then hold above the edge with the full orientation servo this many steps
                                     # (D-63 used soft + 40 at the 0.75 m reach; D-66 brings the edge to 0.60-0.65 m)
    frame_gap: float = 0.06          # base front to the bed frame
    edge_in: float = 0.02            # pinch this far in from the duvet's near edge (D-66: 0.01)
    crouch_max_x: float = 0.78       # crouch reaches 0.78 m ahead at the bed (reach_diag); deep is not used: its elbow
                                     # (arm_3) sits at the limit and the hand ran off to 0.92-0.96 m on every path
    overhang: float = 0.28           # the near edge should end this far beyond the mattress side: as far as the far side
                                     # hangs on a made bed - less, and the heavier far part slid it back on release (D-66)
    reverse_v: float = 0.12          # base speed while pulling back (m/s)
    spots: tuple = (0.0, 0.0, 0.0)   # pull spots along the bed (m from the middle, + = head): the duvet's centre, again
                                     # while not yet covered (D-66; spots 45 cm to either side turned the duvet - 2-3 %
                                     # less cover each; 2 pulls: 8/10, the two left at 0.974 / 0.989)
    back_off: float = 0.45           # side moves go through a waypoint this far back from the bed (turning in place
                                     # 6 cm from the frame the base missed lateral goals by 0.2-0.35 m)
    max_v: float = 0.2
    max_w: float = 0.6
    # D-70 (user 2026-11-02): training starts +-10 cm / +-0.1 rad. Sideways off the pull line, the base first goes to a
    # pose `pre_d` short of the pull pose on that line (D-68's back off / turn / drive / turn, oracle_bed03.align_step)
    # and then drives straight in: driving in at an angle, the hands over the bed caught on it (D-66, 9/18 failed)
    align: bool = True
    pre_d: float = 0.25
    align_back: float = 0.15
    align_lat_tol: float = 0.006
    align_pos_tol: float = 0.004
    # D-70 (data diversity): per episode, drawn from the scene seed - with p_miss one hand pinches miss_out short of the
    # edge (on the bare mattress) and pinches again at the edge while the other hand holds on; with p_short the first
    # pull stops short_by short and the oracle pulls again (recovery data). Only a hand that caught nothing re-grasps:
    # the duvet pinch holds 2-4 nodes, and re-grasping both hands at < 3 nodes moved the duvet off the head row
    # (4/14 vs 5/5 without, 2026-11-02 gate)
    p_miss: float = 0.13
    p_short: float = 0.0             # D-70 collection: 0 (planned 0.12) - the pull after a short one finished the bed in
                                     # 1/7 runs (it starts from an almost covered duvet), so these runs were only waste
    miss_out: float = 0.045
    short_by: float = 0.10
    min_grasp: int = 1
    # D-70: a later pull goes where the target area is still bare - the left hand (head side) over the mean y of the
    # uncovered cells. The wider training starts left the head corner on the open side bare after the first pull
    # (coverage 0.95-0.99) and pulling at the centre again made it worse; the pre-D-70 oracle failed the same 6/6 scenes
    aim_uncovered: bool = True
    device: str = "cuda:0"


@register_policy
class Bed01EasyOracle(Bed03OraclePolicy, PolicyBase[Bed01EasyOracleCfg]):  # Arena: PolicyBase[Cfg] as a direct base
    name = "dx_bed01_easy_oracle"

    def __init__(self, config: Bed01EasyOracleCfg):
        PolicyBase.__init__(self, config)
        self.cfg = config
        import os

        for kv in filter(None, os.environ.get("DX_B01_CFG", "").split(",")):  # experiments: "two_arms=0,pitch_deg=0"
            k, v_ = kv.split("=")
            cur = getattr(self.cfg, k)
            setattr(self.cfg, k, None if v_ == "none" else bool(int(v_)) if isinstance(cur, bool)
                    else tuple(float(x) for x in v_.split(";")) if isinstance(cur, tuple)
                    else float(v_) if cur is None else type(cur)(v_))
        self.reset()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        super().reset(env_ids)
        self.phase, self.t_phase, self.subtask = "plan", 0, "grasp the duvet edge"
        self.spot_i = 0
        self.goal = None
        self.picks = {}
        self.plan_log = []
        self.n_step = 0
        self.meta = {}
        self.al = None
        self.regrasped = 0
        self.true_picks = {}
        self.keep_hold = {}  # D-70: a hand that holds on while the other re-grasps

    def _servo_side(self, obs, target_b, side, kp=0.6, max_dp=0.025):
        """Relative IK step of one arm towards a base-frame TCP position, gripper at `pitch_deg` (D-63)."""
        tcp = obs["policy"][f"tcp_{side}"][0].cpu().numpy()
        d = np.asarray(target_b) - tcp[:3]
        dp = np.clip(kp * d, -max_dp, max_dp)
        dr = np.zeros(3)
        if self.cfg.pitch_deg is not None:
            th = math.radians(self.cfg.pitch_deg)
            R_des = np.array([[math.cos(th), 0.0, math.sin(th)], [0.0, 1.0, 0.0], [-math.sin(th), 0.0, math.cos(th)]])
            Re = R_des @ self._qmat(tcp[3:]).T
            ang = math.acos(np.clip((np.trace(Re) - 1) / 2, -1, 1))
            if ang > 1e-4:
                ax = np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]]) / (2 * math.sin(ang))
                g, mx = ((0.2, 0.02) if self.cfg.soft_above and self.phase == "above"
                         else (self.cfg.rot_gain, self.cfg.rot_max))
                dr = np.clip(ax * ang * g, -mx, mx)
        return np.r_[dp, dr], float(np.linalg.norm(d))

    def _arms(self, a, obs, targets, max_dp=0.025):
        """Servo each arm in `targets` {side: base-frame xyz}; returns the largest position error."""
        errs = []
        for side, tgt in targets.items():
            off = 0 if side == "right" else 6
            a[off:off + 6], e = self._servo_side(obs, tgt, side, max_dp=max_dp)
            errs.append(e)
        return max(errs)

    def _bare_cells(self, U, bed):
        """xy of the 5 cm target cells (mattress top less the D-66 rim) with no duvet node within 5 cm (predicates
        evaluate_bed01 / grid_coverage)."""
        from dxbench.tasks import predicates as P

        _, n = self._cloth_xyz(U, "duvet")
        lo = [v + P.BED01_RIM for v in bed.mattress_min]
        hi = [v - P.BED01_RIM for v in bed.mattress_max]
        hi[bed.long_axis] -= bed.cover_head_gap
        top = bed.mattress_top_z
        sel = n[(n[:, 2] >= top - 0.03) & (n[:, 2] <= top + 0.25)]
        xs = np.arange(lo[0] + 0.025, hi[0], 0.05)
        ys = np.arange(lo[1] + 0.025, hi[1], 0.05)
        g = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)
        if len(sel) == 0:
            return g
        from scipy.spatial import cKDTree

        d, _ = cKDTree(sel[:, :2]).query(g)
        return g[d > 0.05]

    def _sides(self):
        return ("right", "left") if self.cfg.two_arms else ("right",)

    def _bed(self):
        from dxbench.tasks import task_env

        b = task_env.CURRENT_TASK.bed
        return b, task_env.CURRENT_TASK

    def get_action(self, env: gym.Env, observation) -> torch.Tensor:
        U = env.unwrapped
        c = self.cfg
        a = np.zeros(17, dtype=np.float32)
        a[12:14] = 1.0
        self.t_phase += 1
        self.n_step += 1  # frame index of this action in a recording
        if not self.meta:  # D-70: this episode's injected failure, from the scene seed
            rng = episode_rng(1)
            u = float(rng.random())
            rec = "miss" if u < c.p_miss else "short_pull" if u < c.p_miss + c.p_short else None
            self.miss_side = ["right", "left"][int(rng.integers(2))]
            self.meta = {"strategy": "both_hands", "recovery": f"miss_{self.miss_side}" if rec == "miss" else rec,
                         "events": []}
        if self.phase in ("above", "descend", "close"):  # diagnostics: arm joint velocity reversals while reaching
            ph = "jit_" + self.phase
            jv = observation["policy"]["joint_vel"][0, :14].cpu().numpy()
            sg = np.sign(jv) * (np.abs(jv) > 0.05)
            prev = getattr(self, "_jv_sign", None)
            if prev is not None:
                self.jitter = getattr(self, "jitter", 0) + int(np.sum(sg * prev < 0))
                setattr(self, ph, getattr(self, ph, 0) + int(np.sum(sg * prev < 0)))
                self.reach_steps = getattr(self, "reach_steps", 0) + 1
            self._jv_sign = sg
        idx = int(U.action_manager.get_term("torso").preset_index[0])
        if c.fixed_torso:  # D-66: the scene starts in the working posture (crouch) and the torso never moves
            self.work = idx
        work = getattr(self, "work", PRESET_DEEP)
        want = work if self.phase in ("deep", "above", "align", "descend", "close", "pull", "release") else 0
        if idx != want and self.phase not in ("plan", "side", "approach", "finished") and not c.fixed_torso:
            a[14] = (1.0 if idx < want else -1.0) if self.t_phase % 2 == 1 else 0.0
        bed, task = self._bed()
        lo, hi = bed.mattress_min, bed.mattress_max
        if self.phase == "plan":
            if self.spot_i >= len(c.spots):
                self._set("finished")
            else:
                y_mid = 0.5 * (lo[1] + hi[1] - bed.cover_head_gap)
                if c.spot_on_duvet:  # pulled off its centre line the duvet turned and uncovered a corner (2026-11-01)
                    y_mid = float(self._cloth_xyz(U, "duvet")[1][:, 1].mean())
                self.spot_y = y_mid + c.spots[self.spot_i]
                bare = self._bare_cells(U, bed) if c.aim_uncovered and self.spot_i > 0 else None
                if bare is not None and len(bare):
                    # the hand nearest the bare cells over them, both hands on the mattress
                    yb = float(bare[:, 1].mean())
                    hy = abs(c.hand_y)
                    side = 1.0 if yb > self.spot_y else -1.0
                    lo_y, hi_y = lo[1] + hy + 0.05, hi[1] - hy - 0.05
                    self.spot_y = float(np.clip(yb - side * hy, lo_y, hi_y))
                    self.meta.setdefault("events", []).append(["aim_bare", round(yb, 3), len(bare), self.n_step])
                bx = lo[0] - 0.025 - c.frame_gap - BASE_FRONT_X
                by = self.spot_y if c.two_arms else self.spot_y - c.hand_y  # two arms: the hands straddle the spot
                if c.ready_reach:  # stop with the hands above the near edge (not closer than the frame allows)
                    _, n = self._cloth_xyz(U, "duvet")
                    top_ = bed.mattress_top_z
                    edges, self.edge_xy = [], {}
                    for side, hy in (("right", c.hand_y), ("left", -c.hand_y)):
                        m = (np.abs(n[:, 1] - (by + hy)) < 0.05) & (n[:, 2] > top_ - 0.005) & (n[:, 0] > lo[0] + 0.02)
                        if m.any():
                            edges.append(float(n[m, 0].min()))
                            self.edge_xy[side] = np.array([edges[-1] + c.edge_in, by + hy])
                    if edges:
                        bx = min(bx, min(edges) + c.edge_in - c.hand_x)
                        self.edge_x = min(edges)
                self.goal = (np.array([bx, by]), 0.0)
                self.via = np.array([bx - c.back_off, by])
                self.plan_log.append({"spot": self.spot_i, "spot_y": round(self.spot_y, 3), "goal": self.goal[0].tolist()})
                p, _ = self._base(U)
                # the first approach turns, drives straight and turns to face the bed (D-66: the training starts are
                # +-15 cm to the side); only a later spot along the bed backs off first (turning next to the frame missed)
                lateral = abs(float(p[1]) - float(self.goal[0][1])) > 0.05 and self.spot_i > 0
                if c.align and self.al is None:  # D-70: first approach - onto the pull line short of the bed
                    pre_x = min(float(p[0]), float(self.goal[0][0]) - c.pre_d)
                    self.al = {"stage": "check", "axle": np.array([pre_x + AXLE_X, float(self.goal[0][1])]), "yaw": 0.0}
                    self._set("align_base", "grasp the duvet edge")
                else:
                    self._set("side" if lateral else "approach", "grasp the duvet edge")
        elif self.phase == "align_base":
            a[15:17] = align_step(self, U, self.al, c.align_back, c.align_lat_tol, c.align_pos_tol, c.max_w)
            if self.al["stage"] == "done" or self.t_phase > 900:
                p_, yaw_ = self._base(U)
                self.plan_log.append({"base_after_align": np.round(p_[:2], 3).tolist(), "steps": self.t_phase,
                                      "sideways": self.al.get("lat0"), "yaw_err": round(-yaw_, 4)})
                self._set("approach")
        elif self.phase == "side":  # back off and move along the bed, facing it again
            (v, w), ok = self._nav(U, self.via, 0.0)
            a[15:17] = (v, w)
            if ok or self.t_phase > 400:
                self._set("approach")
        elif self.phase == "approach":
            (v, w), ok = self._nav(U, self.goal[0], self.goal[1])
            a[15:17] = (v, w)
            if ok or self.t_phase > 300:
                p, yaw = self._base(U)
                self.plan_log.append({"spot": self.spot_i, "base": np.round(p[:2], 3).tolist(), "yaw": round(yaw, 3)})
                if c.ready_reach:  # straight down onto the edge; the arms take up what the base stop left (frame limit,
                    # nav tolerance: 2-5 cm in x - on the bare mattress the pads caught nothing, 2026-11-01)
                    self.picks = {}
                    for s_ in self._sides():
                        tcp_ = observation["policy"][f"tcp_{s_}"][0, :3].cpu().numpy()
                        xy = getattr(self, "edge_xy", {}).get(s_)
                        self.picks[s_] = (np.r_[self._to_base(U, [xy[0], xy[1], 0.0])[:2], tcp_[2]] if xy is not None
                                          else tcp_)
                    if (self.meta["recovery"] or "").startswith("miss") and not self.meta.get("missed") \
                            and self.miss_side in self.picks:  # D-70: this hand comes down short of the edge
                        self.true_picks[self.miss_side] = self.picks[self.miss_side].copy()
                        self.picks[self.miss_side] = self.picks[self.miss_side] - [c.miss_out, 0.0, 0.0]
                        self.meta["missed"] = True
                        self.meta["events"].append(["miss", self.miss_side, self.n_step])
                    self.plan_log.append({"spot": self.spot_i, "picks_base": {k: np.round(v, 3).tolist()
                                                                              for k, v in self.picks.items()}})
                    self._set("descend")
                else:
                    self._set("deep")
        elif self.phase == "deep":
            if idx == getattr(self, "work", PRESET_DEEP) and self.t_phase > 30 and not getattr(self, "picked", False):
                _, n = self._cloth_xyz(U, "duvet")
                top = bed.mattress_top_z
                p_, yaw_ = self._base(U)
                self.picks, edge = {}, []
                for side in self._sides():
                    hy = c.hand_y if side == "right" else -c.hand_y
                    y_h = float(p_[1]) + hy * math.cos(yaw_)  # the hand's lateral line (the base faces +x)
                    m = (np.abs(n[:, 1] - y_h) < 0.05) & (n[:, 2] > top - 0.005) & (n[:, 0] > lo[0] + 0.02)
                    if not m.any():
                        continue
                    j = np.nonzero(m)[0][np.argmin(n[m, 0])]
                    q = n[j].copy()
                    q[0] += c.edge_in
                    near = np.linalg.norm(n[:, :2] - q[:2], axis=1) < 0.03
                    q[2] = float(n[near, 2].max()) if near.any() else q[2]
                    edge.append(float(n[j, 0]))
                    self.picks[side] = self._to_base(U, q)
                if not self.picks:
                    self.plan_log.append({"spot": self.spot_i, "no_edge": True})
                    self.spot_i += 1
                    self._set("rise")
                else:
                    self.edge_x = min(edge)
                    self.work = 2
                    for side, pk in self.picks.items():
                        if pk[0] > c.crouch_max_x:
                            self.plan_log.append({"spot": self.spot_i, "side": side, "out_of_reach": round(float(pk[0]), 3)})
                    self.picked = True
                    self.plan_log.append({"spot": self.spot_i, "edge_x": round(self.edge_x, 3), "preset": self.work,
                                          "picks_base": {k: np.round(v, 3).tolist() for k, v in self.picks.items()}})
            elif getattr(self, "picked", False) and idx == self.work and self.t_phase > 30:
                self.picked = False
                self._set("above")
        elif self.phase in ("above", "align"):
            err = self._arms(a, observation, {s_: pk + [0, 0, 0.08] for s_, pk in self.picks.items()})
            if self.phase == "above" and (err < 0.015 or self.t_phase > 90):
                self._set("align" if c.align_steps > 0 else "descend")
            elif self.phase == "align" and self.t_phase >= c.align_steps:
                self._set("descend")
        elif self.phase == "descend":  # ready_reach: z only - the picks are the hover TCPs (xy kept)
            _, n = self._cloth_xyz(U, "duvet")
            nb = np.array([self._to_base(U, q) for q in n])
            for pk in self.picks.values():
                near = np.linalg.norm(nb[:, :2] - pk[:2], axis=1) < 0.03
                if near.any():
                    pk[2] = float(nb[near, 2].max())
            err = self._arms(a, observation, {s_: (self.keep_hold[s_] if s_ in self.keep_hold
                                                   else pk + [0, 0, pinch_dz(U) - 0.003]) for s_, pk in self.picks.items()},
                             max_dp=0.012)
            if err < 0.006 or self.t_phase > 90:
                self.hold = {s_: (self.keep_hold[s_] if s_ in self.keep_hold
                                  else observation["policy"][f"tcp_{s_}"][0, :3].cpu().numpy()) for s_ in self.picks}
                self._set("close")
        elif self.phase in ("close", "pull"):
            for s_ in self.picks:
                a[12 if s_ == "right" else 13] = 0.0
            if self.phase == "close":
                self._arms(a, observation, self.hold)
                if self.t_phase > 15:
                    term = U.action_manager.get_term("cloth_assist")
                    got = {s_: int(term.grasped[0, 0 if s_ == "right" else 1]) for s_ in self.picks}
                    self.plan_log.append({"spot": self.spot_i, "grasped": got, "jitter": getattr(self, "jitter", 0),
                                          "jit_above": getattr(self, "jit_above", 0), "jit_descend": getattr(self, "jit_descend", 0),
                                          "reach_steps": getattr(self, "reach_steps", 0)})
                    low = [s_ for s_, k_ in got.items() if k_ < c.min_grasp]
                    self.keep_hold = {}
                    if self.true_picks or (low and self.regrasped < 1 and any(got.values())):
                        # one hand caught nothing: it opens, rises and pinches again; the other holds on (D-70)
                        self.regrasped += 1
                        redo = set(self.true_picks) | set(low)
                        self.keep_hold = {s_: self.hold[s_] for s_ in self.picks if s_ not in redo}
                        for s_, pk_ in self.true_picks.items():
                            self.picks[s_] = pk_
                        self.true_picks = {}
                        self.meta["events"].append(["regrasp", got, self.n_step])
                        self._set("reopen")
                    elif not any(got.values()):
                        self.spot_i += 1
                        self._set("release")
                    else:
                        # total edge travel: from its x to `overhang` beyond the mattress side; the arms give part
                        travel = self.edge_x - (lo[0] - c.overhang)
                        if self.meta["recovery"] == "short_pull" and not self.meta.get("shortened"):
                            travel -= c.short_by  # D-70: stop short, the next spot pulls again
                            self.meta["shortened"] = True
                            self.meta["events"].append(["short_pull", self.n_step])
                        self.reverse_s = max(0.0, travel) / c.reverse_v
                        self.start = {s_: v.copy() for s_, v in self.hold.items()}
                        self._set("pull", "pull the duvet")
            else:
                n_arm = 45
                s = min(1.0, self.t_phase / n_arm)
                self._arms(a, observation, {s_: v + s * np.array([0.0, 0.0, 0.03]) for s_, v in self.start.items()},
                           max_dp=0.015)
                if self.t_phase * (1 / 15) < self.reverse_s:
                    a[15] = -c.reverse_v
                if self.t_phase > n_arm and self.t_phase * (1 / 15) >= self.reverse_s + 0.5:
                    self.plan_log.append({"spot": self.spot_i, "pulled_s": round(self.reverse_s, 2)})
                    self.spot_i += 1
                    self._set("release")
        elif self.phase == "reopen":  # the re-grasping hand opens and rises 4 cm, then comes down on its pick again
            if self.t_phase == 1:
                self.reopen_from = {s_: observation["policy"][f"tcp_{s_}"][0, :3].cpu().numpy() for s_ in self.picks}
            self._arms(a, observation, {s_: (self.keep_hold[s_] if s_ in self.keep_hold
                                             else v + [0, 0, 0.04 if self.t_phase > 6 else 0.0])
                                        for s_, v in self.reopen_from.items()}, max_dp=0.012)
            if self.t_phase > 25:
                self._set("descend")
        elif self.phase == "release":
            if self.t_phase == 1:
                self.hold = {s_: observation["policy"][f"tcp_{s_}"][0, :3].cpu().numpy() + [0, 0, 0.01]
                             for s_ in self.picks}
            self._arms(a, observation, self.hold)
            if self.t_phase > 20:
                for s_ in self.picks:
                    U.action_manager.get_term(f"{s_}_arm").request_home()
                self._set("rise")
        elif self.phase == "rise":
            self.work = PRESET_DEEP
            if (idx == 0 or c.fixed_torso) and self.t_phase > 45:
                done = task.evaluate_cached(U)[0]
                self.plan_log.append({"after_spot": self.spot_i, "coverage": round(float(done["coverage"]), 3)})
                self._set("finished" if done["success"] and done["stages"].get("duvet_neat") else "plan")
        for s_ in self.keep_hold:  # D-70: the holding hand stays closed while the other re-grasps
            a[12 if s_ == "right" else 13] = 0.0
        return torch.as_tensor(a, device=U.device).unsqueeze(0).repeat(U.num_envs, 1)
