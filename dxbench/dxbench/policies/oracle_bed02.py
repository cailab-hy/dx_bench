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
4. "place on the stack": both hands pinch the folded towel at its two ends and carry it together to the stack spot
   (40 cm in from the desk front, straight behind the towel - D-32; one central pinch let the half fold open),
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

from dxbench.policies.oracle_bed03 import AXLE_X, BASE_FRONT_X, Bed03OraclePolicy, _T, pinch_dz, wrap

# TCP height above the cloth surface when pinching: oracle_bed03.pinch_dz (fingertips 8 mm above for the cylinder
# assist, 2 mm into the cloth for the pad assist)
ARM = {"right": (0, 12), "left": (6, 13)}  # action offset of the 6-D arm delta, gripper index


@dataclass
class Bed02OraclePolicyCfg(PolicyCfg):
    # base front to the desk's front edge, negative = under the desk (D-56, 2026-10-29). The desk is a table (apron
    # rails from 0.645 m), so the base rolls under it until the robot touches it: leaning at -0.023 m (the chest meets
    # the desk edge), upright -0.225 (but upright reaches only 0.39-0.42 m ahead at desk height), the rby1-sdk
    # 30/-60/30 deg pose -0.063 (policies/drive_in_diag.py, reach_diag.py). The robot leans 25 cm back and then rolls in
    # (leaning at the desk pushed it back 6 cm)
    edge_margin: float = -0.05       # aimed past the contact: drive_in stops when the base stops advancing
    above_dz: float = 0.06
    arc_h: float = 0.06           # fold arc height (0.10 before the 2026-10-29 sweep; 0.18 / 0.25 opened the half fold)
    max_v: float = 0.25
    max_w: float = 0.6
    half_inset: float = 0.04      # half fold: pinch points this far in from the towel's ends (along its long axis)
    half_vin: float = 0.04        # ... and this far in from its far long edge
    nudge: float = 0.08           # after the half fold, slide the strip this far into the desk (away from its edge)
    # side folds: lay the end this far past its mirror image (negative = short of it). The flap's part near the fold
    # line lands further out than the pinched end, so laying the end short of its mirror puts the whole flap on it:
    # 2026-10-29 sweep (120 Hz, 5 scenes) +0.06: 0/5, 0: 0/5, -0.04: 4/5, -0.06 / -0.04: 5/5
    side_over: float = -0.06
    side_over2: float | None = -0.04  # the second side fold's (None = side_over): both ends meet at the centre
                                     # and the second, laid past its mirror, dragged the first flap (2026-10-29)
    side_arc: float | None = 0.16  # side folds' arc height (None = arc_h); a page flip of the flap pinched 0.16 m
                                   # from the fold line is a semicircle of that radius
    hold_in: float = 0.0          # side folds: the other hand holds the strip this far inside the fold line (0 = off;
                                  # 0.06 broke the half fold more often and doubled the time, 2026-10-29)
    reaxis: bool = False          # side folds: re-estimate the strip's long axis (it turns during the half fold)
    order: str = ""               # move order override, e.g. "right,left,half2,stack" (empty: the default sequence)
    side_over_r: float | None = None  # the right side fold's offset regardless of order (None: first / second rule)
    tighten: float = 0.05         # right_t / left_t moves: pull the flap near the crease this far towards the centre
    put_dz: float = 0.012         # put height above the cloth there (the flap is released from this high)
    side_dv_r: float = 0.0        # right side fold: shift the put this far into the desk (along v; the held right
                                  # flap's end twisted ~4 cm towards the robot at 480 Hz)
    hold_first_side: bool = False  # keep the first side flap pinched until the second side fold is laid (480 Hz: the
                                   # right flap sprang back flat on release while the left one stayed)
    release_wait: int = 0         # side folds: hold the laid flap this many steps before opening (lets it settle)
    side_pick_d: float | None = None  # side folds: pinch this far from the fold line (None: side_end_in from the end)
    # D-61 (2026-10-30, with the mirror-diagonal towel mesh): 4 cm in, the free tip beyond the pinch flopped back during
    # the page flip (tip 8 cm short, the part near the line 3.5 cm too deep; right flap 39 % right after the fold), 1.5 cm
    # in: 88-100 %
    side_end_in: float = 0.015    # side folds pinch this far in from the strip end
    # stack move: pinched 3 cm from the folded strip's ends (its creases), the carry slid both flaps (right 0.99 -> 0.6);
    # 15 cm in the pads hold the flap and the layers under it together (0.94 / 0.92 at the end; 8 cm 0.75, 17 cm 0.88)
    stack_in: float = 0.15
    # D-64 (user 2026-10-31): fold the 0.4 x 0.2 m strip once more to a 0.2 m square (8 layers). An arm crosses the body
    # centre by ~0.1 m at most (reach_diag), so both hands first slide the strip `quarter_shift` towards the robot's
    # right, then the right hand lays the right end on the left end; the square goes to the stack
    quarter: bool = True
    quarter_shift: float = 0.09
    quarter_over: float = -0.04   # lay the end this far short of the far end (negative = short)
    quarter_arc: float = 0.08     # (0.12: 4 / 3 scenes the same, the stack slid more)
    quarter_stack_d: float = 0.06  # > 0: the square to the stack in both hands, pinched this far either side of its
                                   # centre (one hand at the centre: the square turned and spread, area 0.07-0.08 m2)
    side_corners: bool = False    # side folds as two one-hand moves: the end's near corner, then its far corner
                                  # (0/10, the half fold opened, 2026-10-29)
    side_two_hands: bool = False  # side folds with both hands on the two corners of the strip end (crumpled the towel:
                                  # fold errors ~25 cm, 0/10, 2026-10-29)
    # D-66 (user 2026-11-01): the easy scenes start leaning at the desk in front of the towel - turn in place to its
    # axis (no driving), keep the torso (no crouch / drive-in, no final upright)
    turn_only: bool = True
    # D-68 (user 2026-11-02, sim2real): the easy start is jittered by +-5 cm / +-0.05 rad like BED-01. Turning in place
    # leaves an xy offset, and 3 cm to the robot's right (or -0.1 rad, which turns the base 2 cm sideways about the
    # axle) already failed the quarter fold, 4 cm further back the half fold (~/dx_120hz/b02reach). With `align` the
    # base first goes to the nominal pose of the easy scenes (base origin align_desk_d from the desk front, towel centre
    # straight ahead): sideways off it, it backs off align_back, turns towards the pose, drives there and turns back;
    # then it fixes the remaining distance straight along its heading
    align: bool = True
    align_desk_d: float = 0.33    # scene_gen.gen_bed02 easy: the drive-in stalled at 0.31-0.32 m (chest at the edge)
    align_back: float = 0.15
    align_lat_tol: float = 0.006  # drive-axle offset sideways of the nominal pose that needs no sideways move
    align_pos_tol: float = 0.004
    # a pinch holding fewer than min_grasp nodes opens, comes down 3 mm lower and closes again (up to max_regrasp
    # times). The quarter fold with 3-5 nodes left the square open in 15/15 runs, 8-12 nodes held (b02reach)
    min_grasp: int = 6
    max_regrasp: int = 2
    regrasp_moves: str = "quarter"
    work_preset_steps: int = 1   # lean: crouch cannot reach z 0.8-0.9 near the centre line at x 0.6 (reach_desk_0.0.json)
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
        import os

        if os.environ.get("DX_B02_HALF_INSET"):  # experiments (2026-10-28 reach study)
            self.cfg.half_inset = float(os.environ["DX_B02_HALF_INSET"])
        if os.environ.get("DX_B02_DEBUG"):
            self.debug = True
        for kv in filter(None, os.environ.get("DX_B02_CFG", "").split(",")):  # experiments: "side_over=0.06,arc_h=0.06"
            k, v_ = kv.split("=")
            cur = getattr(self.cfg, k)
            setattr(self.cfg, k, float(v_) if cur is None else type(cur)(v_))
        self.reset()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        self.__dict__.pop("u_span", None)
        self.__dict__.pop("n_side", None)
        self.quartered = False
        self.held = None
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
        self.tilt = {"right": 0.0, "left": 0.0}

    def _set(self, phase, subtask=None):
        self.phase, self.t_phase = phase, 0
        if subtask:
            self.subtask = subtask

    # ------------------------------------------------------------------ helpers
    def _servo(self, obs, target_b, side, kp=0.6, max_dp=0.025):
        """Relative IK step of one arm towards a base-frame TCP position, gripper pointing down (identity orientation
        in the base frame). A tilt-only variant (yaw free) did not stop the left-wrist flip of v13 (v14), so v12's full
        orientation is kept."""
        tcp = obs["policy"][f"tcp_{side}"][0].cpu().numpy()
        d = np.asarray(target_b) - tcp[:3]
        dp = np.clip(kp * d, -max_dp, max_dp)
        R = self._h._qmat(tcp[3:])
        Re = R.T  # identity (gripper pointing down) times R^T
        ang = math.acos(np.clip((np.trace(Re) - 1) / 2, -1, 1))
        dr = np.zeros(3)
        if ang > 1e-4:
            ax = np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]]) / (2 * math.sin(ang))
            dr = np.clip(ax * ang * kp, -0.08, 0.08)
        self.tilt[side] = math.degrees(math.acos(np.clip(R[2, 2], -1, 1)))  # TCP z axis vs base z
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

    def _fold_fracs(self, U, task):
        """Debug: per fold line, the fraction of mirror pairs within 5 cm and their median distance (fold_done)."""
        from scipy.spatial import cKDTree

        from dxbench.tasks import predicates as P

        n = (_T(U.scene[self.target].data.nodal_pos_w)[0] - U.scene.env_origins[0]).cpu().numpy()
        kind = task.spec.obj(self.target).kind
        uv = task._rest_uv(U, self.target)
        out = {}
        for k, (ax, c) in P.FOLD_LINES[kind].items():
            a = np.nonzero(uv[:, ax] - c > 0.05)[0]
            m = uv[a].copy()
            m[:, ax] = 2 * c - m[:, ax]
            dr, j = cKDTree(uv).query(m)
            ok = dr < 0.02
            d = np.linalg.norm(n[a[ok]] - n[j[ok]], axis=1)
            out[k[5:]] = (round(float((d < 0.05).mean()), 2), round(float(np.median(d)), 3))
        return out

    def _start_moves(self):
        self.moves = (["half2", "nudge", "right_n", "right_f", "left_n", "left_f", "stack"]
                      if self.cfg.side_corners else ["half2", "nudge", "right", "left", "stack"])
        if self.cfg.quarter:
            self.moves[-1:-1] = ["shift", "quarter"]
        if self.cfg.order:
            self.moves = self.cfg.order.split(";")
        self._set("plan")

    def _arc(self):
        """Carry arc height of the current move."""
        if self.mv.get("tighten"):
            return 0.02
        if self.mv["move"] == "quarter":
            return self.cfg.quarter_arc
        if self.mv["move"].split("_")[0] in ("right", "left") and self.cfg.side_arc:
            return self.cfg.side_arc
        return self.cfg.arc_h

    def _align(self, U):
        """(v, w) of the D-68 move to the nominal pose; self.al["stage"] becomes "done" there. Steers the drive axle
        (the point the base turns about): turning in place does not move it."""
        p, yaw = self._h._base(U)
        f_now = np.array([math.cos(yaw), math.sin(yaw)])
        ax = p[:2] + AXLE_X * f_now
        al, tgt, yaw_t = self.al, self.al["axle"], self.al["yaw"]
        f_t = np.array([math.cos(yaw_t), math.sin(yaw_t)])
        d = tgt - ax
        w_max = self.cfg.max_w
        if al["stage"] == "check":
            al["lat0"] = round(float(d @ np.array([-f_t[1], f_t[0]])), 4)
            al["depth0"] = round(float(d @ f_t), 4)
            al["stage"] = "back" if abs(al["lat0"]) > self.cfg.align_lat_tol else "turn_final"
            al["from"] = ax.copy()
        if al["stage"] == "back":  # straight back, off the desk
            if float((al["from"] - ax) @ f_now) >= self.cfg.align_back:
                al["stage"] = "turn_to"
            else:
                return -0.10, 0.0
        if al["stage"] == "turn_to":
            e = wrap(math.atan2(d[1], d[0]) - yaw)
            if abs(e) > 0.01:
                return 0.0, float(np.clip(2.0 * e, -w_max, w_max))
            al["stage"], al["head"] = "drive", math.atan2(d[1], d[0])
        if al["stage"] == "drive":
            dist = float(np.linalg.norm(d))
            if dist > 0.03:  # steer at the target; the last 3 cm on the heading held so far
                al["head"] = math.atan2(d[1], d[0])
            hd = np.array([math.cos(al["head"]), math.sin(al["head"])])
            rem = float(d @ hd)
            if rem > self.cfg.align_pos_tol:
                e = wrap(al["head"] - yaw)
                return min(0.12, 0.8 * rem + 0.02) * math.cos(e) ** 2, float(np.clip(2.0 * e, -w_max, w_max))
            al["stage"] = "turn_final"
        if al["stage"] == "turn_final":
            e = wrap(yaw_t - yaw)
            if abs(e) > 0.005:
                return 0.0, float(np.clip(2.0 * e, -w_max, w_max))
            al["stage"] = "straight"
        if al["stage"] == "straight":  # the distance left along the heading (forwards or backwards)
            s_ = float(d @ f_t)
            if abs(s_) > self.cfg.align_pos_tol:
                v = math.copysign(max(0.015, min(0.06, abs(s_))), s_)
                return v, float(np.clip(2.0 * wrap(yaw_t - yaw), -w_max, w_max))
            al["stage"] = "done"
        return 0.0, 0.0

    @staticmethod
    def _strip_v(pu, pv, ua, ub):
        """Near / far edge (along v) of the strip between u = ua and ub."""
        m = (pu >= ua) & (pu <= ub)
        if m.sum() < 3:
            return float(pv.min()), float(pv.max())
        return float(np.percentile(pv[m], 5)), float(np.percentile(pv[m], 95))

    def _plan(self, U, move, task):
        """Per-arm pick / put points (world xyz) of one fold move. All arms of a move ride one shared arc."""
        n, c, u, v, pu, pv = self._towel(U)
        if self.cfg.reaxis and move.split("_")[0] in ("right", "left"):
            # the strip turned during the half fold / nudge (3-7 cm skew of the side flaps at 480 Hz, 2026-10-29):
            # side folds use its current long axis (sign kept from the frozen frame)
            xy = n[:, :2] - c[:2]
            _, V = np.linalg.eigh(xy.T @ xy)
            u2 = V[:, 1] * (1.0 if float(V[:, 1] @ u) >= 0 else -1.0)
            v2 = np.array([-u2[1], u2[0]])
            v2 = v2 * (1.0 if float(v2 @ v) >= 0 else -1.0)
            u, v, pu, pv = u2, v2, xy @ u2, xy @ v2
        u0, u1, v0, v1 = float(pu.min()), float(pu.max()), float(pv.min()), float(pv.max())

        def local_top(xy, r=0.025):
            """Cloth surface height under the grasp: 75th percentile of the node heights within the capture radius.
            The max lifted the TCP onto a crumple (4.6 cm above the desk) so the folded layers below were > 5 cm under
            the TCP, outside the assist capture window (v10 / v11: 0-4 nodes grasped)."""
            d = np.linalg.norm(n[:, :2] - np.asarray(xy)[:2], axis=1)
            m = d < r
            if m.sum() < 3:
                m = d < 2 * r
            return float(np.percentile(n[m, 2], 75)) if m.any() else self.table_z

        def end_v(sign):  # centre (along v) of the strip end on the right (+1) / left (-1), kept near the robot
            m = (pu > u1 - 0.08) if sign > 0 else (pu < u0 + 0.08)
            # (clamping this towards the near edge put the left target at x 0.55 / y 0.27, next to the body, where
            # the left arm's relative IK flipped the wrist - v13 / v14)
            return float(pv[m].mean()) if m.any() else 0.0

        if move == "half2":   # far long edge (v1) by both corners -> near long edge (v0): the strip ends up close
            label = "fold in half"  # to the robot, where both hands reach its ends for the side folds
            arms = [(s, c[:2] + u * uu + v * (v1 - self.cfg.half_vin), c[:2] + u * uu + v * (v0 + 0.02 + self.cfg.half_vin))
                    for s, uu in (("right", u1 - self.cfg.half_inset), ("left", u0 + self.cfg.half_inset))]
        elif move == "nudge":  # D-56: the strip lies at the desk front after the half fold and the side folds tipped it
            label = "fold in half"  # over the edge (3/3 on the floor, 2026-10-29): slide it 8 cm into the desk first
            ends = [("right", u1 - 0.03, end_v(+1)), ("left", u0 + 0.03, end_v(-1))]
            arms = [(s_, c[:2] + u * uu + v * ve, c[:2] + u * uu + v * (ve + self.cfg.nudge)) for s_, uu, ve in ends]
        corner = None
        if move[-2:] in ("_n", "_f"):  # side_corners: right_n / right_f / left_n / left_f
            move, corner = move[:-2], move[-1]
        tighten = move.endswith("_t")
        if tighten:  # right_t / left_t: pull the laid flap's part near the crease towards the centre
            move = move[:-2]
        if move in ("right", "left") and corner in (None, "n") and not tighten:
            self.n_side = getattr(self, "n_side", 0) + 1
        over = self.cfg.side_over if (getattr(self, "n_side", 1) <= 1 or self.cfg.side_over2 is None) else self.cfg.side_over2
        if move == "right" and self.cfg.side_over_r is not None:  # per-side override (480 Hz: the right flap needed more)
            over = self.cfg.side_over_r
        if move in ("right", "left") and not hasattr(self, "u_span"):  # the strip's length before the side folds: after the right
            self.u_span = u1 - u0  # fold it is ~0.6 m and the left fold line computed from it lay 5 cm short (2026-10-29)
        span = getattr(self, "u_span", u1 - u0)
        if move in ("half2", "nudge"):  # planned above
            pass
        elif move == "right":  # right end of the strip -> its mirror image across the fold line a quarter in (D-41:
            # fold_done checks mirror pairs; laying it 3 cm past the centre overshot by ~7 cm, stretched the flap and
            # opened the half fold, 2026-10-29)
            label, ve = "fold the right side", end_v(+1)
            pick_u = u1 - self.cfg.side_end_in if self.cfg.side_pick_d is None else u1 - 0.25 * span + self.cfg.side_pick_d
            put_u = 2 * (u1 - 0.25 * span) - pick_u - over
            arms = [("right", c[:2] + u * pick_u + v * ve, c[:2] + u * put_u + v * (ve + self.cfg.side_dv_r))]
            if tighten:
                tu = u1 - 0.06  # just inside the folded right edge (the crease), on the flap
                arms = [("right", c[:2] + u * tu + v * ve, c[:2] + u * (tu - self.cfg.tighten) + v * ve)]
            elif corner:
                vn, vf = self._strip_v(pu, pv, u1 - 0.08, u1)
                vc = vn + 0.03 if corner == "n" else vf - 0.03
                arms = [("right", c[:2] + u * pick_u + v * vc, c[:2] + u * put_u + v * vc)]
            elif self.cfg.side_two_hands:  # both corners of the end (one pinch in the middle of an edge crumples it)
                vn, vf = self._strip_v(pu, pv, u1 - 0.08, u1)
                arms = [("right", c[:2] + u * pick_u + v * (vf - 0.03), c[:2] + u * put_u + v * (vf - 0.03)),
                        ("left", c[:2] + u * pick_u + v * (vn + 0.03), c[:2] + u * put_u + v * (vn + 0.03))]
            elif self.cfg.hold_in > 0:  # the other hand pins the strip just inside the fold line (the robot-free demo
                hu = u1 - 0.25 * span - self.cfg.hold_in  # held the non-flap part; without it the part near the line
                arms.append(("left", c[:2] + u * hu + v * ve, None))  # was dragged 5-9 cm)
        elif move == "left":
            label, ve = "fold the left side", end_v(-1)
            pick_u = u0 + self.cfg.side_end_in if self.cfg.side_pick_d is None else u0 + 0.25 * span - self.cfg.side_pick_d
            put_u = 2 * (u0 + 0.25 * span) - pick_u + over
            arms = [("left", c[:2] + u * pick_u + v * ve, c[:2] + u * put_u + v * ve)]
            if tighten:
                tu = u0 + 0.06
                arms = [("left", c[:2] + u * tu + v * ve, c[:2] + u * (tu + self.cfg.tighten) + v * ve)]
            elif corner:
                vn, vf = self._strip_v(pu, pv, u0, u0 + 0.08)
                vc = vn + 0.03 if corner == "n" else vf - 0.03
                arms = [("left", c[:2] + u * pick_u + v * vc, c[:2] + u * put_u + v * vc)]
            elif self.cfg.side_two_hands:
                vn, vf = self._strip_v(pu, pv, u0, u0 + 0.08)
                arms = [("left", c[:2] + u * pick_u + v * (vf - 0.03), c[:2] + u * put_u + v * (vf - 0.03)),
                        ("right", c[:2] + u * pick_u + v * (vn + 0.03), c[:2] + u * put_u + v * (vn + 0.03))]
            elif self.cfg.hold_in > 0:
                hu = u0 + 0.25 * span + self.cfg.hold_in
                arms.append(("right", c[:2] + u * hu + v * ve, None))
        elif move == "shift":  # D-64: both hands slide the folded strip towards the robot's right (u)
            label = "fold in half"
            ends = [("right", u1 - self.cfg.stack_in, end_v(+1)), ("left", u0 + self.cfg.stack_in, end_v(-1))]
            arms = [(s_, c[:2] + u * uu + v * ve, c[:2] + u * (uu + self.cfg.quarter_shift) + v * ve) for s_, uu, ve in ends]
        elif move == "quarter":  # D-64: the right end over onto the left end (all four layers pinched)
            label = "fold in half"
            pick_u = u1 - self.cfg.side_end_in
            put_u = u0 + self.cfg.side_end_in - self.cfg.quarter_over
            ve = 0.5 * (v0 + v1)
            arms = [("right", c[:2] + u * pick_u + v * ve, c[:2] + u * put_u + v * ve)]
        elif getattr(self, "quartered", False):  # the square to the stack spot in the right hand, centre pinched
            label = "place on the stack"
            shift = np.asarray(task.stack_xy, dtype=float) - c[:2]
            uc, vc = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
            dq = self.cfg.quarter_stack_d
            ends = [("right", uc + dq, vc), ("left", uc - dq, vc)] if dq > 0 else [("right", uc, vc)]
            arms = [(s_, c[:2] + u * uu + v * ve, c[:2] + u * uu + v * ve + shift) for s_, uu, ve in ends]
        else:                  # folded towel -> stack spot with both hands (each on its own end), translated together
            label = "place on the stack"
            shift = np.asarray(task.stack_xy, dtype=float) - c[:2]
            ends = [("right", u1 - self.cfg.stack_in, end_v(+1)), ("left", u0 + self.cfg.stack_in, end_v(-1))]
            arms = [(s_, c[:2] + u * uu + v * ve, c[:2] + u * uu + v * ve + shift) for s_, uu, ve in ends]
        def local_bottom(xy, r=0.025):
            """Lowest cloth layer under the grasp (10th percentile): after the half fold the pinch must hold both
            layers; at the top layer it took only the folded-over flap and the side folds opened the half fold."""
            d = np.linalg.norm(n[:, :2] - np.asarray(xy)[:2], axis=1)
            m = d < r
            return float(np.percentile(n[m, 2], 10)) if m.sum() >= 3 else self.table_z

        out = []
        for side, pick, put in arms:
            grip_dz = pinch_dz(U)
            z_pick = (local_top(pick) if (move == "half2" or tighten) else local_bottom(pick)) + grip_dz
            if put is None:  # a holding hand: pinches in place for the whole move
                out.append({"side": side, "pick": np.r_[pick, z_pick], "put": np.r_[pick, z_pick], "hold": True})
                continue
            z_put = local_top(put) + self.cfg.put_dz + grip_dz
            out.append({"side": side, "pick": np.r_[pick, z_pick], "put": np.r_[put, z_put]})
        return {"move": move, "label": label, "arms": out, "tighten": tighten}

    # ------------------------------------------------------------------ main
    def get_action(self, env: gym.Env, observation) -> torch.Tensor:
        from dxbench.tasks import task_env

        U = env.unwrapped
        h = self._h
        a = np.zeros(17, dtype=np.float32)
        a[12:14] = 1.0
        if getattr(self, "held", None) is not None:  # a held flap stays pinched between moves
            a[ARM[self.held[0]][1]] = 0.0
        self.t_phase += 1
        tidx = int(U.action_manager.get_term("torso").preset_index[0])

        def torso_to(idx):
            if tidx < idx:
                return 1.0 if self.t_phase % 2 == 1 else 0.0
            if tidx > idx:
                return -1.0 if self.t_phase % 2 == 1 else 0.0
            return 0.0

        task = task_env.CURRENT_TASK
        if self.phase == "init":
            self.target = task.cloth_names[0]
            self._set("approach", "pick up a shirt")
        if self.phase == "approach":
            if self.goal is None:
                desk = task.layout["parts_aabb"].get("CareDesk") or task.layout["furniture_aabb"]["CareDesk"]
                n0, c, *_ = self._towel(U)
                # face the towel's long edge squarely (2026-10-29): the towel lies with up to +-0.2 rad of yaw, and
                # facing -x one far corner of the half fold was 0.81 m ahead (the other 0.59 m) - past the arm's reach
                xy = n0[:, :2] - c[:2]
                _, V = np.linalg.eigh(xy.T @ xy)
                ux, uy = V[:, 1]
                sg = -1.0 if uy > 0 else 1.0  # robot left = sg * u, with the heading near pi
                yaw = math.atan2(-sg * ux, sg * uy)
                dev = max(-0.25, min(0.25, wrap(yaw - math.pi)))
                yaw = math.pi + dev
                f = np.array([math.cos(yaw), math.sin(yaw)])
                # base front edge_margin under the desk; a turned base sticks out further on one side
                front_x = float(desk[1][0]) + self.cfg.edge_margin + 0.25 * abs(math.sin(dev))
                bx = front_x - BASE_FRONT_X * f[0]
                d = (float(c[0]) - bx) / f[0]  # towel centre straight ahead of the base
                self.goal = (np.array([bx, float(c[1]) - d * f[1]]), yaw)
                # lean 25 cm back, then drive in (2026-10-29): leaning at the goal swept the upper body / arms into the
                # desk and pushed the base back 6 cm (front only ~4 cm under the desk instead of 12)
                self.final_goal = self.goal
                self.goal = (self.goal[0] - 0.25 * f, yaw)
                self.plan_log.append({"approach_yaw_dev": round(dev, 3), "goal": self.final_goal[0].round(3).tolist()})
                self.table_z = float(desk[1][2])
                # nominal easy pose (scene_gen.gen_bed02): its drive axle is where the axle of a canonical start is
                bx = float(desk[1][0]) + self.cfg.align_desk_d
                by = float(c[1]) - (float(c[0]) - bx) / f[0] * f[1]
                self.al = {"stage": "check", "axle": np.array([bx, by]) + AXLE_X * f, "yaw": yaw}
            if self.cfg.turn_only and self.cfg.align:
                a[15:17] = self._align(U)
                if self.al["stage"] == "done" or self.t_phase > 900:
                    p_, yaw_ = h._base(U)
                    ax_ = p_[:2] + AXLE_X * np.array([math.cos(yaw_), math.sin(yaw_)])
                    self.plan_log.append({"base_after_align": np.round(p_[:2], 3).tolist(), "steps": self.t_phase,
                                          "axle_err": round(float(np.linalg.norm(ax_ - self.al["axle"])), 4),
                                          "yaw_err": round(wrap(self.al["yaw"] - yaw_), 4),
                                          "sideways": self.al.get("lat0")})
                    a[15:17] = (0.0, 0.0)
                    self._start_moves()
            elif self.cfg.turn_only:
                _, yaw_now = h._base(U)
                e = wrap(self.goal[1] - yaw_now)
                ok = abs(e) <= 0.02
                a[15:17] = (0.0, 0.0 if ok else float(np.clip(2.0 * e, -self.cfg.max_w, self.cfg.max_w)))
                if ok or self.t_phase > 150:
                    p_, _ = h._base(U)
                    self.plan_log.append({"base_after_turn": np.round(p_[:2], 3).tolist(), "yaw_err": round(e, 3)})
                    self._start_moves()
            else:
                (v, w), ok = h._nav(U, self.goal[0], self.goal[1])
                a[15:17] = (v, w)
                if ok or self.t_phase > 300:
                    self.goal = None
                    self._set("crouch", "flatten the shirt")
        elif self.phase == "crouch":
            a[14] = torso_to(self.cfg.work_preset_steps)
            if tidx == self.cfg.work_preset_steps and self.t_phase > 60:
                self._set("drive_in")
        elif self.phase == "drive_in":  # leaning, roll the base in under the desk edge until it touches
            (v, w), ok = h._nav(U, self.final_goal[0], self.final_goal[1])
            a[15:17] = (v, w)
            p_, _ = h._base(U)
            self.din_hist = (getattr(self, "din_hist", []) + [float(p_[0])])[-15:]
            stalled = self.t_phase > 30 and len(self.din_hist) == 15 and abs(self.din_hist[0] - self.din_hist[-1]) < 0.002
            if ok or stalled or self.t_phase > 200:
                self.din_hist = []
                p_, _ = h._base(U)
                self.plan_log.append({"base_after_drive_in": np.round(p_[:2], 3).tolist()})
                self._start_moves()
        elif self.phase == "plan":
            if not self.moves:
                self._set("retract")
            else:
                self.mv = self._plan(U, self.moves.pop(0), task)
                self.plan_log.append({"move": self.mv["move"], "arms": [
                    {k: (x.tolist() if hasattr(x, "tolist") else x) for k, x in arm.items()} for arm in self.mv["arms"]]})
                self._set("above", self.mv["label"])
        elif self.phase in ("above", "descend", "close", "lift", "carry", "release", "reopen"):
            arms = self.mv["arms"]
            grip = 0.0 if self.phase in ("close", "lift", "carry") else 1.0
            targets = {}
            if self.phase == "carry":  # shared fold arc: rises to arc_h and comes down past the line
                D = max(float(np.linalg.norm(arm["put"][:2] - arm["pick"][:2])) for arm in arms if not arm.get("hold"))
                arc = self._arc()
                # ~0.8 cm of path per step; half speed while a hand lags > 2.5 cm, wait while it lags > 4 cm (the arms
                # trail the arc by up to 15 cm at 1.2 cm per step with the towel pinned)
                e_ = self.last_err if self.last_err is not None else 0.0
                rate = 0.008 if e_ < 0.025 else (0.004 if e_ < 0.07 else 0.0)
                self.arc_s = min(1.0, self.arc_s + rate / max(D + 2 * arc, 0.05))
            for arm in arms:
                pick, put = arm["pick"], arm["put"]
                if arm.get("hold") and self.phase in ("lift", "carry"):
                    tgt, mdp = pick, 0.012
                elif self.phase == "above":
                    tgt, mdp = pick + [0, 0, self.cfg.above_dz], 0.025
                elif self.phase in ("descend", "close"):
                    tgt, mdp = pick, 0.012
                elif self.phase == "reopen":  # too few nodes pinched: open, rise 3 cm, then come down again
                    tgt, mdp = pick + [0, 0, 0.03 if self.t_phase > 6 else 0.0], 0.012
                elif self.phase == "lift":  # short vertical lift so the pinched edge leaves the desk
                    tgt, mdp = pick + [0, 0, 0.03], 0.02
                elif self.phase == "carry":
                    s_ = self.arc_s
                    tgt = pick + (put - pick) * s_
                    arc = self._arc()
                    tgt[2] = pick[2] + 0.03 + (put[2] - pick[2] - 0.03) * s_ + arc * math.sin(math.pi * s_)
                    mdp = 0.02
                else:  # release: open in place first (pins drop when the fingers open), then rise 12 cm and stay
                    tgt, mdp = put + [0, 0, 0.0 if self.t_phase <= 8 else 0.12], 0.015
                targets[arm["side"]] = (tgt, mdp)
            held_grip = {}
            if self.held is not None and self.held[0] not in targets:  # the held first flap: stay closed in place,
                hs, hput = self.held                                   # open with the second fold's release
                if self.phase == "release":
                    targets[hs] = (hput + [0, 0, 0.0 if self.t_phase <= 8 else 0.12], 0.015)
                    held_grip[hs] = 1.0
                else:
                    targets[hs] = (hput, 0.012)
                    held_grip[hs] = 0.0
            errs = []
            for side, (tgt, mdp) in targets.items():
                off, gi = ARM[side]
                tb = h._to_base(U, tgt)
                a[off:off + 6], e = self._servo(observation, tb, side, max_dp=mdp)
                a[gi] = held_grip.get(side, grip)
                if side not in held_grip:
                    errs.append(e)
                if self.debug and (self.t_phase % 15 == 1 or self.phase == "close" and self.t_phase == 2):
                    tcp7 = observation["policy"][f"tcp_{side}"][0].cpu().numpy()
                    tilt = self.tilt[side]  # TCP z axis vs base z (0 = pointing down)
                    print(f"[bed02dbg] {self.mv['move']} {self.phase} t={self.t_phase} {side} tgt_b={np.round(tb, 3)} "
                          f"tcp_b={np.round(tcp7[:3], 3)} err={e:.3f} tilt={tilt:.1f}deg s={self.arc_s:.2f}", flush=True)
            err = max(errs)
            self.last_err = err
            st = self.plan_log[-1].setdefault("min_err", {})
            st[self.phase] = round(min(st.get(self.phase, 9.0), err), 4)
            if (self.phase == "above" and self.mv["move"] != "half2" and any(self.tilt[s_] > 60.0 for s_ in targets)
                    and self.t_phase > 15 and not self.mv.get("recovered")):
                # a wrist flip: send the flipped arm(s) back to READY once and retry this move from there
                self.mv["recovered"] = True
                self.plan_log[-1]["recover_home"] = [s_ for s_ in targets if self.tilt[s_] > 60.0]
                for s_ in self.plan_log[-1]["recover_home"]:
                    U.action_manager.get_term(f"{s_}_arm").request_home()
                self._set("recover")
            elif self.phase == "above" and (err < 0.015 or self.t_phase > 90):
                self._set("descend")
            elif self.phase == "descend" and (err < 0.005 or self.t_phase > 60):  # D-66: 0.01 closed the pads with the
                # tips ~4 mm above the towel (the capture margin) when the arms came down from the desk_reach hover
                self._set("close")
            elif self.phase == "close" and self.t_phase > 10:
                term = U.action_manager.get_term("cloth_assist")
                g_ = {s: int(term.grasped[0, 0 if s == "right" else 1]) for s in targets}
                self.plan_log[-1]["grasped"] = g_
                n_re = self.mv.get("regrasps", 0)
                if (self.mv["move"] in self.cfg.regrasp_moves.split(",") and n_re < self.cfg.max_regrasp
                        and any(k < self.cfg.min_grasp for k in g_.values())):
                    self.mv["regrasps"] = n_re + 1
                    self.plan_log[-1].setdefault("regrasp", []).append(g_)
                    for arm in self.mv["arms"]:
                        arm["pick"] = arm["pick"] + [0, 0, -0.003]
                    self._set("reopen")
                else:
                    self._set("lift")
            elif self.phase == "reopen" and self.t_phase > 20:
                self._set("descend")
            elif self.phase == "lift" and (err < 0.015 or self.t_phase > 30):
                self.arc_s = 0.0
                self._set("carry")
            elif self.phase == "carry" and ((self.arc_s >= 1.0 and err < 0.03 and self.t_phase >= 0) or self.t_phase > 450):
                side_move = self.mv["move"].split("_")[0] in ("right", "left")
                if self.cfg.release_wait > 0 and side_move and not self.mv.get("waited"):
                    self.mv["waited"] = True  # stay closed at the put pose (the carry targets hold at s = 1)
                    self._set("carry")
                    self.t_phase = -self.cfg.release_wait
                    self.arc_s = 1.0
                else:
                    if self.debug:
                        self.plan_log[-1]["folds_held"] = self._fold_fracs(U, task)
                    if (self.cfg.hold_first_side and side_move and self.held is None
                            and getattr(self, "n_side", 0) == 1 and len(self.mv["arms"]) == 1):
                        arm0 = self.mv["arms"][0]
                        self.held = (arm0["side"], arm0["put"])  # keep pinching it: no release, next move
                        self._set("plan")
                    else:
                        self._set("release")
            elif self.phase == "release" and self.t_phase > 30:
                # no joint homing between moves: from the READY pose with the lean torso the relative IK drifted away
                # from desk targets (v5 / v9); the next move starts from this raised pose instead
                if self.debug:
                    self.plan_log[-1]["folds_after"] = self._fold_fracs(U, task)
                    import os

                    if os.environ.get("DX_B02_SNAP"):  # debug: the towel after each move -> <dir>/<i>_<move>.npz
                        snap = (_T(U.scene[self.target].data.nodal_pos_w)[0] - U.scene.env_origins[0]).cpu().numpy()
                        k = sum(1 for e_ in self.plan_log if "move" in e_)
                        np.savez(f"{os.environ['DX_B02_SNAP']}/{k}_{self.mv['move']}.npz", nodes=snap)
                if self.held is not None and self.held[0] not in [arm["side"] for arm in self.mv["arms"]]:
                    self.held = None  # released together with this move
                if self.mv["move"] == "quarter":
                    self.quartered = True
                self._set("plan")
        elif self.phase == "rehome":
            homing = [bool(U.action_manager.get_term(f"{arm['side']}_arm").homing[0]) for arm in self.mv["arms"]]
            if (not any(homing) and self.t_phase > 5) or self.t_phase > 200:
                self._set("plan")
        elif self.phase == "recover":
            homing = [bool(U.action_manager.get_term(f"{arm['side']}_arm").homing[0]) for arm in self.mv["arms"]]
            if (not any(homing) and self.t_phase > 5) or self.t_phase > 200:
                self._set("above")
        elif self.phase == "retract":
            if self.cfg.turn_only:  # D-66: the torso stays leaning
                self._set("finished")
            else:
                a[14] = torso_to(0)
                if tidx == 0 and self.t_phase > 60:
                    self._set("finished")
        return torch.as_tensor(a, device=U.device).unsqueeze(0).repeat(U.num_envs, 1)
