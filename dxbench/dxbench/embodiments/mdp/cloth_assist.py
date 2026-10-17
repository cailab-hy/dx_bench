"""Assisted cloth grasp (stage 1 decision D-4/D-5): a sim-only crutch for pinching cloth with RB-Y1 fingers.

When a gripper command crosses to "closed" (< close_threshold), the cloth nodes inside the finger region of that
hand (within `radius` of the TCP approach axis, between 5 cm below and 3 cm above the TCP) are captured with
their offsets in the TCP frame. While the command stays closed they are driven by velocity writes:
    v = gain * (target - x) / dt + target velocity
(position writes blow large moving cloth up; surface deformables have no kinematic targets). Opening releases them.
Add with `add_cloth_assist(env_cfg, cloth_names)` in an env_cfg_callback; the finger/cloth collision filter is
authored by the `filter_finger_cloth_collisions` prestartup event.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.managers import EventTermCfg
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

TCP_OFFSET = (0.0, 0.0, -0.1045)


def _T(x):
    import warp as wp

    if hasattr(x, "torch"):
        return x.torch
    return wp.to_torch(x) if isinstance(x, wp.array) else x


def _qmat(q):  # (N,4) xyzw -> (N,3,3)
    x, y, z, w = q.unbind(-1)
    return torch.stack([torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


class ClothGraspAssistAction(ActionTerm):
    cfg: ClothGraspAssistActionCfg

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._raw = torch.zeros(self.num_envs, 0, device=self.device)
        self._dt = env.physics_dt
        self._sides = ("right", "left")
        self._bodies = [self._asset.body_names.index(f"ee_{s}") for s in self._sides]
        self._pins: dict[tuple[int, int, str], tuple[torch.Tensor, torch.Tensor]] = {}
        self._prev_tgt: dict[tuple[int, int, str], torch.Tensor] = {}
        self._closed = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        self.grasped = torch.zeros(self.num_envs, 2, dtype=torch.long, device=self.device)  # pinned node count

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

    def _tcp(self):
        p = _T(self._asset.data.body_link_pos_w)[:, self._bodies]  # (N,2,3)
        q = _T(self._asset.data.body_link_quat_w)[:, self._bodies]
        R = _qmat(q)  # (N,2,3,3)
        off = torch.tensor(TCP_OFFSET, device=self.device)
        return p + R @ off, R

    def _gripper_cmd(self):
        g = self._env.action_manager.get_term(self.cfg.gripper_term).raw_actions  # [right, left] 1 = open
        return g[:, :2] < self.cfg.close_threshold

    def apply_actions(self):
        closed = self._gripper_cmd()
        tcp, R = self._tcp()
        newly = closed & ~self._closed
        opened = ~closed & self._closed
        for e, s in torch.nonzero(opened).tolist():
            for c in self.cfg.cloth_names:
                self._pins.pop((e, s, c), None)
                self._prev_tgt.pop((e, s, c), None)
            self.grasped[e, s] = 0
        for e, s in torch.nonzero(newly).tolist():
            for c in self.cfg.cloth_names:
                nodes = _T(self._env.scene[c].data.nodal_pos_w)[e]
                loc = (nodes - tcp[e, s]) @ R[e, s]
                m = (loc[:, :2].norm(dim=-1) < self.cfg.radius) & (loc[:, 2] > -0.05) & (loc[:, 2] < 0.03)
                ids = torch.nonzero(m).flatten()
                if len(ids):
                    self._pins[(e, s, c)] = (ids, loc[ids])
                    self.grasped[e, s] += len(ids)
        self._closed = closed
        if not self._pins:
            return
        by_cloth: dict[str, list] = {}
        for (e, s, c), v in self._pins.items():
            by_cloth.setdefault(c, []).append((e, s, v))
        for c, items in by_cloth.items():
            obj = self._env.scene[c]
            pos = _T(obj.data.nodal_pos_w)
            vel = _T(obj.data.nodal_vel_w).clone()
            for e, s, (ids, loc) in items:
                tgt = tcp[e, s] + loc @ R[e, s].T
                key = (e, s, c)
                v_ff = (tgt - self._prev_tgt[key]) / self._dt if key in self._prev_tgt else torch.zeros_like(tgt)
                self._prev_tgt[key] = tgt.clone()
                vel[e, ids] = self.cfg.gain * (tgt - pos[e, ids]) / self._dt + v_ff
            obj.write_nodal_velocity_to_sim_index(vel)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = range(self.num_envs) if env_ids is None else [int(i) for i in env_ids]
        for k in [k for k in self._pins if k[0] in ids]:
            self._pins.pop(k, None)
            self._prev_tgt.pop(k, None)
        for e in ids:
            self._closed[e] = False
            self.grasped[e] = 0


@configclass
class ClothGraspAssistActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = ClothGraspAssistAction
    asset_name: str = "robot"
    cloth_names: list[str] = []
    gripper_term: str = "gripper"
    close_threshold: float = 0.5
    radius: float = 0.05
    gain: float = 1.0


HAND_LINKS = tuple(f"link_{side}_arm_{i}" for side in ("right", "left") for i in (5, 6)) + (
    "right_gripper", "left_gripper", "ee_right", "ee_left", "ee_finger_r1", "ee_finger_r2", "ee_finger_l1", "ee_finger_l2")


def _cloth_targets(stage, root, cloth_names):
    from pxr import Usd

    out = []
    for c in cloth_names:
        cp = stage.GetPrimAtPath(f"{root}/{c}")
        if cp.IsValid():
            found = [q.GetPath() for q in Usd.PrimRange(cp) if "OmniPhysicsDeformableBodyAPI" in q.GetAppliedSchemas()]
            out += found or [cp.GetPath()]
    return out


def _filter_links(stage, root, link_names, targets):
    from pxr import Usd, UsdPhysics

    robot = stage.GetPrimAtPath(f"{root}/Robot")
    if not robot.IsValid() or not targets:
        return 0
    n = 0
    for q in Usd.PrimRange(robot):
        if q.GetName() in link_names and q.HasAPI(UsdPhysics.RigidBodyAPI):
            rel = UsdPhysics.FilteredPairsAPI.Apply(q).CreateFilteredPairsRel()
            for t in targets:
                rel.AddTarget(t)
            n += 1
    return n


def filter_finger_cloth_collisions(env, env_ids, cloth_names: list[str], links=HAND_LINKS):
    """Prestartup event: FilteredPairsAPI between the hands (fingers, gripper body, ee, wrist links 5-6) and each
    cloth (every env). Fingers only was not enough on the mobile robot: nodes pinned within 5 cm of the TCP overlap
    the gripper body, the velocity pin fights that contact and the energy flung the free base (stage 3, D-19)."""
    import isaaclab.sim as sim_utils

    stage = sim_utils.get_current_stage()
    for e in range(env.num_envs):
        root = f"/World/envs/env_{e}"
        _filter_links(stage, root, set(links), _cloth_targets(stage, root, cloth_names))


def filter_base_cloth_collisions(env, env_ids, cloth_names: list[str], links=("wheel_l", "wheel_r", "base")):
    """Prestartup event: FilteredPairsAPI between the drive wheels / base (caster cylinders are base colliders) and
    each cloth. Wheels rolling onto a PhysX surface deformable produced contact forces that flipped the robot
    (stage 3, BED-03 oracle); with the filter the robot drives over floor cloth as if it were flat (DECISIONS D-18)."""
    import isaaclab.sim as sim_utils

    stage = sim_utils.get_current_stage()
    for e in range(env.num_envs):
        root = f"/World/envs/env_{e}"
        _filter_links(stage, root, set(links), _cloth_targets(stage, root, cloth_names))


def add_base_cloth_filter(env_cfg, cloth_names: list[str]):
    env_cfg.events.filter_base_cloth = EventTermCfg(func=filter_base_cloth_collisions, mode="prestartup",
                                                    params={"cloth_names": list(cloth_names)})
    return env_cfg


def add_cloth_assist(env_cfg, cloth_names: list[str], radius: float = 0.05):
    """Call from an env_cfg_callback: adds the action term (0-dim) and the prestartup collision filter."""
    env_cfg.actions.cloth_assist = ClothGraspAssistActionCfg(cloth_names=list(cloth_names), radius=radius)
    env_cfg.events.filter_finger_cloth = EventTermCfg(func=filter_finger_cloth_collisions, mode="prestartup",
                                                      params={"cloth_names": list(cloth_names)})
    return env_cfg
