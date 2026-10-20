"""Pad-box assisted cloth grasp (2026-10-20): a stricter, better-looking successor of `cloth_assist` (D-4/D-U11).

`cloth_assist` captures every node within 2.5 cm of the TCP axis (5 cm below .. 3 cm above the TCP) and keeps its
offset, so a hand closed 1-2 cm above a flat cloth still "grasps" it and the cloth hangs visibly below the fingers.
Here a hand only grasps what a real parallel gripper could pinch:

* capture (on the open -> closed command edge): nodes inside the box between the two finger pads, in the grip frame
  (origin = fingertip centre, axes w = pad width, c = closing direction, a = approach, pointing out of the tips):
      |w| <= pad_w / 2 + margin_w,  |c| <= half_gap + margin_c,  -pad_h <= a <= tip_margin
  so a fingertip more than `tip_margin` above the cloth catches nothing (the sim-to-real point of this module);
* pull-in: captured nodes are driven, over `snap_s`, into the pad region (w, a clamped to the pad); along c they
  are squeezed by `squeeze` (0 = keep their spread, 1 = clamp to the closing gap). Full squeeze collapses rows of
  nodes a few cm apart onto one line, the FEM triangles degenerate and the cloth was launched (Franka test); 0.7 was
  stable but let the pinned nodes drift up to 3 cm out of the pads on BED-02 (0: <= 9 mm), so the default is 0 (D-34);
* hold: velocity pins as in `cloth_assist` (surface deformables have no kinematic targets, position writes blow up):
      v = gain * (target - x) / dt + target velocity
  and opening the command releases them.

The hand <-> cloth collision must be filtered (fingers pass through the pinched layers, as in LeIsaac's PBD cloth);
use `filter_hand_cloth_collisions` as a prestartup event.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from dxbench.embodiments.mdp.cloth_assist import _T, _cloth_targets, _filter_links, _qmat


@configclass
class HandSpec:
    """One gripper. Body frame -> grip frame is a fixed transform (`tip_offset`, `axes`)."""

    body: str = MISSING
    tip_offset: tuple[float, float, float] = MISSING  # fingertip centre in the body frame (m)
    axes: tuple[tuple[float, float, float], ...] = MISSING  # grip w, c, a axes in the body frame (rows)
    gap_joints: tuple[str, ...] = MISSING  # finger joints; half gap = gap_scale * mean |q|
    gap_scale: float = 1.0
    gripper_index: int = 0  # column of the gripper term's raw action
    close_below: float = 0.0  # closed while raw action < this
    pad_w: float = 0.02  # pad width along w (m)
    pad_h: float = 0.02  # pad length from the tip along -a (m)


class PadGraspAction(ActionTerm):
    cfg: PadGraspActionCfg

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._raw = torch.zeros(self.num_envs, 0, device=self.device)
        self._dt = env.physics_dt
        self._hands = list(cfg.hands)
        self._bodies = [self._asset.body_names.index(h.body) for h in self._hands]
        self._gap_ids = [self._asset.find_joints(list(h.gap_joints), preserve_order=True)[0] for h in self._hands]
        self._tip = torch.tensor([h.tip_offset for h in self._hands], device=self.device)  # (H,3)
        self._axes = torch.tensor([h.axes for h in self._hands], device=self.device)  # (H,3,3) rows w,c,a
        H = len(self._hands)
        self._closed = torch.zeros(self.num_envs, H, dtype=torch.bool, device=self.device)
        self._pins: dict[tuple[int, int, str], dict] = {}
        self.grasped = torch.zeros(self.num_envs, H, dtype=torch.long, device=self.device)  # pinned node count
        self.last_capture: dict[tuple[int, int], dict] = {}  # diagnostics of the latest close per (env, hand)
        # one record per finished grasp: nodes, fingertip-cloth gap at capture, how far the pinned nodes left the pad
        # box (m, max / mean over the hold), hold time (s)
        self.stats: list[dict] = []

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self):
        return self._raw

    @property
    def processed_actions(self):
        return self._raw

    def process_actions(self, actions):
        pass

    def grip_frames(self):
        """(N,H,3) fingertip centre, (N,H,3,3) world rotation whose columns are w, c, a; (N,H) half gap."""
        p = _T(self._asset.data.body_link_pos_w)[:, self._bodies]
        R = _qmat(_T(self._asset.data.body_link_quat_w)[:, self._bodies])  # body -> world
        o = p + (R @ self._tip[None, :, :, None]).squeeze(-1)
        G = R @ self._axes.transpose(-1, -2)[None]  # columns = grip axes in world
        q = _T(self._asset.data.joint_pos)
        half = torch.stack([q[:, ids].abs().mean(-1) * h.gap_scale for ids, h in zip(self._gap_ids, self._hands)], -1)
        return o, G, half

    def _closed_cmd(self):
        g = self._env.action_manager.get_term(self.cfg.gripper_term).raw_actions
        return torch.stack([g[:, h.gripper_index] < h.close_below for h in self._hands], -1)

    def apply_actions(self):
        c = self.cfg
        closed = self._closed_cmd()
        o, G, half = self.grip_frames()
        for e, h in torch.nonzero(~closed & self._closed).tolist():
            for name in c.cloth_names:
                pin = self._pins.pop((e, h, name), None)
                if pin is not None:
                    self._record(e, h, name, pin)
            self.grasped[e, h] = 0
        for e, h in torch.nonzero(closed & ~self._closed).tolist():
            hs = self._hands[h]
            diag = {"half_gap": float(half[e, h]), "n": 0}
            for name in c.cloth_names:
                nodes = _T(self._env.scene[name].data.nodal_pos_w)[e]
                loc = (nodes - o[e, h]) @ G[e, h]  # (V,3) w, c, a
                m = ((loc[:, 0].abs() <= hs.pad_w / 2 + c.margin_w) & (loc[:, 1].abs() <= half[e, h] + c.margin_c)
                     & (loc[:, 2] >= -hs.pad_h) & (loc[:, 2] <= c.tip_margin))
                # top cloth surface under the pads relative to the fingertips (topmost node in the w/c footprint that is
                # not above the pads; > 0 = cloth below the tips, < 0 = tips pressed into the cloth)
                fm = ((loc[:, 0].abs() <= hs.pad_w / 2 + c.margin_w) & (loc[:, 1].abs() <= half[e, h] + c.margin_c)
                      & (loc[:, 2] >= -hs.pad_h))
                if fm.any():
                    diag[f"{name}_tip_to_cloth"] = float(loc[fm, 2].min())
                ids = torch.nonzero(m).flatten()
                if len(ids) >= c.min_nodes:
                    self._pins[(e, h, name)] = {"ids": ids, "loc0": loc[ids].clone(), "t": 0, "prev": None,
                                                "tip_to_cloth": diag.get(f"{name}_tip_to_cloth"),
                                                "out_max": 0.0, "out_sum": 0.0, "out_n": 0}
                    self.grasped[e, h] += len(ids)
                    diag["n"] += len(ids)
            self.last_capture[(e, h)] = diag
            if diag["n"] == 0:  # a close that caught nothing: keep it in the stats too (why did it miss?)
                self.stats.append({"env": e, "hand": h, "nodes": 0, "half_gap": round(diag["half_gap"], 4),
                                   **{k: round(v, 4) for k, v in diag.items() if k.endswith("tip_to_cloth")}})
        self._closed = closed
        if not self._pins:
            return
        n_snap = max(1, round(c.snap_s / self._dt))
        by_cloth: dict[str, list] = {}
        for (e, h, name), pin in self._pins.items():
            by_cloth.setdefault(name, []).append((e, h, pin))
        for name, items in by_cloth.items():
            obj = self._env.scene[name]
            pos = _T(obj.data.nodal_pos_w)
            vel = _T(obj.data.nodal_vel_w).clone()
            for e, h, pin in items:
                hs = self._hands[h]
                loc0 = pin["loc0"]
                wlim = max(hs.pad_w / 2 - c.inset, 0.0)
                clim = torch.maximum(torch.clamp(half[e, h] - c.inset, min=0.0), (1.0 - c.squeeze) * loc0[:, 1].abs())
                goal = torch.stack([loc0[:, 0].clamp(-wlim, wlim),
                                    torch.maximum(torch.minimum(loc0[:, 1], clim), -clim),
                                    loc0[:, 2].clamp(-hs.pad_h + c.inset, -c.inset)], -1)
                pin["t"] += 1
                s = min(1.0, pin["t"] / n_snap)
                s = s * s * (3 - 2 * s)  # smoothstep: no velocity jump at capture
                tl = loc0 + s * (goal - loc0)
                if s >= 1.0:  # held: how far are the pinned nodes outside the pad box (the visible gap)?
                    cur = (pos[e, pin["ids"]] - o[e, h]) @ G[e, h]
                    out = torch.stack([cur[:, 0].abs() - hs.pad_w / 2, cur[:, 1].abs() - clim,
                                       cur[:, 2], -hs.pad_h - cur[:, 2]], -1).clamp(min=0.0).amax(-1)
                    pin["out_max"] = max(pin["out_max"], float(out.max()))
                    pin["out_sum"] += float(out.mean())
                    pin["out_n"] += 1
                tgt = o[e, h] + tl @ G[e, h].T
                v_ff = (tgt - pin["prev"]) / self._dt if pin["prev"] is not None else torch.zeros_like(tgt)
                pin["prev"] = tgt.clone()
                vel[e, pin["ids"]] = c.gain * (tgt - pos[e, pin["ids"]]) / self._dt + v_ff
            obj.write_nodal_velocity_to_sim_index(vel)

    def _record(self, e, h, name, pin):
        self.stats.append({"env": e, "hand": h, "cloth": name, "nodes": int(len(pin["ids"])),
                           "tip_to_cloth": pin["tip_to_cloth"], "out_max": round(pin["out_max"], 4),
                           "out_mean": round(pin["out_sum"] / max(pin["out_n"], 1), 4),
                           "hold_s": round(pin["t"] * self._dt, 2)})

    def pinned_offsets(self, e: int, h: int, name: str):
        """Grip-frame (w, c, a) of the pinned nodes now (diagnostics: are they between the pads?)."""
        pin = self._pins.get((e, h, name))
        if pin is None:
            return None
        o, G, _ = self.grip_frames()
        nodes = _T(self._env.scene[name].data.nodal_pos_w)[e, pin["ids"]]
        return (nodes - o[e, h]) @ G[e, h]

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = range(self.num_envs) if env_ids is None else [int(i) for i in env_ids]
        for k in [k for k in self._pins if k[0] in ids]:
            self._record(*k, self._pins.pop(k))
        for e in ids:
            self._closed[e] = False
            self.grasped[e] = 0


@configclass
class PadGraspActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = PadGraspAction
    asset_name: str = "robot"
    cloth_names: list[str] = []
    gripper_term: str = "gripper"
    hands: list[HandSpec] = []
    tip_margin: float = 0.004  # cloth may sit at most this far beyond the fingertips (a > 0) to be caught
    margin_w: float = 0.002
    margin_c: float = 0.002
    inset: float = 0.002  # pulled-in nodes stay this far inside the pad edges
    min_nodes: int = 1
    snap_s: float = 0.2
    squeeze: float = 0.0  # fraction of the closing-direction spread removed (see module docstring)
    gain: float = 1.0


def filter_hand_cloth_collisions(env, env_ids, cloth_names: list[str], links: Sequence[str]):
    """Prestartup event: FilteredPairsAPI between the given robot links and every cloth (all envs)."""
    import isaaclab.sim as sim_utils

    stage = sim_utils.get_current_stage()
    n = 0
    for e in range(env.num_envs):
        root = f"/World/envs/env_{e}"
        n += _filter_links(stage, root, set(links), _cloth_targets(stage, root, cloth_names))
    print(f"[pad_grasp] filtered {n} link(s) against {list(cloth_names)}", flush=True)


# RB-Y1 (dxbench/assets/rby1/rby1_dx.usda, measured 2026-10-20): finger boxes 1.6 (x) x 3.2 (y) x 6.2 (z) cm, tips at
# ee z -0.1355, fingers close along ee x (prismatic, |q| = 0 closed .. 0.05 open = each finger's distance from the
# centre), approach = ee -z. Gripper term raw action [right, left], 1 = open.
RBY1_HANDS = [HandSpec(body=f"ee_{side}", tip_offset=(0.0, 0.0, -0.1355),
                       axes=((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
                       gap_joints=(f"gripper_finger_{s}1", f"gripper_finger_{s}2"), gripper_index=i, close_below=0.5,
                       pad_w=0.032, pad_h=0.02)
              for i, (side, s) in enumerate((("right", "r"), ("left", "l")))]


def add_pad_grasp(env_cfg, cloth_names: list[str], squeeze: float = 0.0, tip_margin: float = 0.004):
    """Call from an env_cfg_callback (RB-Y1): registers the term as `cloth_assist` (oracles read `.grasped` there)
    plus the hand/cloth prestartup filter of `cloth_assist` (D-19 links)."""
    from isaaclab.managers import EventTermCfg

    from dxbench.embodiments.mdp.cloth_assist import filter_finger_cloth_collisions

    env_cfg.actions.cloth_assist = PadGraspActionCfg(cloth_names=list(cloth_names), gripper_term="gripper",
                                                     hands=RBY1_HANDS, squeeze=squeeze, tip_margin=tip_margin,
                                                     snap_s=0.5 if squeeze > 0 else 0.2)
    env_cfg.events.filter_finger_cloth = EventTermCfg(func=filter_finger_cloth_collisions, mode="prestartup",
                                                      params={"cloth_names": list(cloth_names)})
    return env_cfg
