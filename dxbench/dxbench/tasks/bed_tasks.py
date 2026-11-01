"""BED-01/02/03 Arena tasks in the DX care room + their environments (dx_bed01 / dx_bed02 / dx_bed03).

    python submodules/IsaacLab-Arena/isaaclab_arena/evaluation/policy_runner.py --policy_type zero_action --num_episodes 1 \
        --external_environment_class_path dxbench.tasks.bed_tasks:DxBed03Environment dx_bed03 \
        --scene benchmarks/dx_bed_v0/scenes/bed03_easy_000.json

A scene JSON (dxbench.tasks.scene_spec) fixes the objects, cloth states, robot start and language. The room-independent
parts (env factory, shared events, task base, task registry) are in tasks/task_env.py; the care room (background, its
prestartup events) in assets/scenes.py ROOMS. The DX care-room background keeps its furniture; its own blanket / pillow /
desk T-shirt are deactivated before the simulation starts and the benchmark garments are spawned as Arena deformables
(so the task can read and reset them). Grasping cloth uses the assisted grasp: embodiments/mdp/pad_grasp.py by default
(D-34), cloth_assist.py with --grasp_assist cylinder (DECISIONS D-4/D-5/D-U11).
Scores: dxbench.tasks.predicates (pure numpy, unit-tested).
"""
from __future__ import annotations

import numpy as np

from dxbench.assets import objects as O
from dxbench.tasks import predicates as P
from dxbench.tasks import task_env
from dxbench.tasks.scene_spec import SceneSpec
from dxbench.tasks.task_env import (CLOTH_KINDS, RoomEnvironmentBase, RoomTaskBase, TaskEntry, _T,  # noqa: F401
                                    aim_observer, register_task)


def __getattr__(name):  # bed_tasks.CURRENT_TASK (older scripts) -> the one task_env keeps
    if name == "CURRENT_TASK":
        return task_env.CURRENT_TASK
    raise AttributeError(name)


# ------------------------------------------------------------------------------------------ task base
class BedTaskBase(RoomTaskBase):
    task_id = "BED-0X"
    room = "care_room"


# ------------------------------------------------------------------------------------------ BED-03
class Bed03LaundryTask(BedTaskBase):
    task_id = "BED-03"

    def evaluate(self, env) -> list[dict]:
        clothes = [o.name for o in self.spec.objects if o.kind in CLOTH_KINDS and not _is_folded(o)]
        folded = [o.name for o in self.spec.objects if o.kind in CLOTH_KINDS and _is_folded(o)]
        dis = [o.name for o in self.spec.objects if o.kind in ("doll", "book")]
        bspec = P.basket_spec(next((o.variant for o in self.spec.objects if o.kind == "basket"), None))
        held_by = getattr(env.action_manager._terms.get("cloth_assist"), "held_cloths", None)  # pad_grasp (D-60)
        out = []
        for e in range(env.num_envs):
            bp, bq = self.rigid_pose(env, "laundry_basket", e)
            out.append(P.evaluate_bed03({c: self.nodes(env, c, e) for c in clothes},
                                        {d: self.rigid_pose(env, d, e) for d in dis}, bp, bq,
                                        folded={c: self.nodes(env, c, e) for c in folded}, spec=bspec,
                                        held=held_by(e) if held_by else None))
        return out


# ------------------------------------------------------------------------------------------ BED-02
class Bed02FoldTask(BedTaskBase):
    task_id = "BED-02"

    def __init__(self, spec, episode_length_s):
        super().__init__(spec, episode_length_s)
        desk = self.layout["parts_aabb"].get("CareDesk") or self.layout["furniture_aabb"]["CareDesk"]
        self.table_z = float(desk[1][2])
        lo, hi = np.array(desk[0]), np.array(desk[1])
        # stack spot (D-32, 2026-10-18): straight behind the garment(s) as seen from the robot - 40 cm in from the desk
        # front edge (robot faces -x from the +x side), laterally at the garments' mean y. Standing at the desk front in
        # front of the garment, both hands reach it with the towel held at its two ends (each hand on its own side,
        # +-12 cm, x 0.55-0.75 m: reports/rby1_check/t5/reach_desk_+-0.12.json), so the folded towel is carried with
        # both hands. Garments spawn 25 cm in, so a flat towel starts 15 cm from it (radius 12 cm: not on the stack).
        # Before: 22 cm in, 30 cm left of the desk centre - only the left hand could reach it.
        ys = [o.xyz[1] for o in spec.objects if o.kind in ("towel", "tshirt")]
        cy = float(np.mean(ys)) if ys else 0.5 * float(lo[1] + hi[1])
        self.stack_xy = (float(hi[0] - 0.40), cy)
        self._uv: dict[str, np.ndarray] = {}
        self._start: dict[int, dict] = {}
        self._seen: dict[int, dict] = {}  # D-64: folds reached so far this episode, per env and garment

    def _rest_uv(self, env, name):
        """Flat rest frame (u, v) of a garment's nodes from its spawned (pre-state) mesh; cached, and written to
        outputs/cache/rest_uv_<kind>.npz for offline audits (tools/predicate_audit.py)."""
        if name not in self._uv:
            o = self.spec.obj(name)
            rest = _T(env.scene[name].data.default_nodal_state_w)[0, :, :3] - env.scene.env_origins[0]
            uv = P.rest_uv(rest.cpu().numpy(), o.xyz, o.yaw)
            self._uv[name] = uv
            try:
                from dxbench.paths import REPO

                d = REPO / "outputs" / "cache"
                d.mkdir(parents=True, exist_ok=True)
                f = d / f"rest_uv_{o.kind}.npz"
                if not f.exists():
                    np.savez(f, uv=uv)
            except OSError:
                pass
        return self._uv[name]

    def evaluate(self, env) -> list[dict]:
        """D-41: every garment, topological folds (predicates.evaluate_bed02_garments); D-64: towel folds latch."""
        spec = P.FoldSpec(rest_area=0.0, stack_center=self.stack_xy, table_z=self.table_z)
        out = []
        for e in range(env.num_envs):
            new_ep = e not in self._start or int(env.episode_length_buf[e]) <= 1
            gs = []
            for name in self.cloth_names:
                o = self.spec.obj(name)
                if o.kind not in P.FOLD_LINES:
                    continue
                n = self.nodes(env, name, e)
                if new_ep:
                    self._start.setdefault(e, {})[name] = (
                        P.projected_area(n[:, :2]) >= 0.8 * P.REST_AREA[o.kind], n[:, :2].mean(axis=0))
                    self._seen.setdefault(e, {})[name] = set()
                flat0, c0 = self._start[e][name]
                gs.append({"name": name, "kind": o.kind, "nodes": n, "uv": self._rest_uv(env, name),
                           "start_flat": flat0, "start_centroid": c0, "seen": self._seen[e][name]})
            out.append(P.evaluate_bed02_garments(gs, spec))
        return out

    @staticmethod
    def _flat_dims(o):
        """Flat (long, short) size of the garment variant (m). Real products (D-36): towel 0.8 x 0.4, M(95) T-shirt
        0.75 (cuff to cuff) x 0.625. Stage-3 variants: towels 0.7 x 0.45 / small 0.5 x 0.3, T-shirt 0.93 x 0.66
        scaled 0.85 / 1.0 / 1.15 for s / m / l (assets/make_objects.py)."""
        if o.kind == "towel":
            if _towel_color(o) is not None:
                return O.TOWEL_SIZE
            return (0.5, 0.3) if o.variant == "small" else (0.7, 0.45)
        if o.kind == "tshirt" and (o.variant or "").startswith("m95_"):
            return (0.75, 0.625)
        if o.kind == "tshirt":
            k = {"s": 0.85, "m": 1.0, "l": 1.15}.get((o.variant or "m_")[0], 1.0)
            return (0.93 * k, 0.66 * k)
        return (0.5, 0.4)

    def _fold_flags(self, nodes, o):
        # extent-based fold proxy: each fold roughly halves / thirds one dimension of the flat garment
        long_, short_ = P.obb_extent_xy(nodes[:, :2])
        flat = self._flat_dims(o)
        f1 = long_ <= 0.75 * flat[0]
        f2 = long_ <= 0.5 * flat[0] + 0.05
        f3 = short_ <= 0.6 * flat[1]
        return (bool(f1), bool(f2), bool(f3))


# ------------------------------------------------------------------------------------------ BED-01
class Bed01MakeBedTask(BedTaskBase):
    task_id = "BED-01"

    def __init__(self, spec, episode_length_s):
        super().__init__(spec, episode_length_s)
        m = self.layout["parts_aabb"]["Bed/Mattress"]
        bed = self.layout["furniture_aabb"]["Bed"]
        self.bed = P.BedSpec(mattress_min=tuple(m[0][:2]), mattress_max=tuple(m[1][:2]), mattress_top_z=float(m[1][2]),
                             headboard_center=(0.5 * (m[0][0] + m[1][0]), float(m[1][1]) - 0.25),
                             long_axis=1 if (bed[1][1] - bed[0][1]) > (bed[1][0] - bed[0][0]) else 0,
                             # D-41: cover the mattress up to the DX blanket's head edge (pillow zone left free);
                             # D-66: no pillow in the scene (easy) -> the whole mattress top
                             cover_head_gap=(float(m[1][1]) - float(self.layout["parts_aabb"]["Bed/Blanket"][1][1])
                                             if any(o.kind == "pillow" for o in spec.objects) else 0.0))

    def evaluate(self, env) -> list[dict]:
        return [P.evaluate_bed01(self.nodes(env, "duvet", e), self.bed) for e in range(env.num_envs)]




# ------------------------------------------------------------------------------------------ assets from spec
def _is_folded(o) -> bool:
    return (o.variant or "").startswith("folded")


def _towel_color(o) -> str | None:
    """TOWEL_COLORS key of a towel variant ("10_lavender" or "folded:10_lavender"), None for the stage-3 towels."""
    c = (o.variant or "").split(":")[-1]
    return c if c in O.TOWEL_COLORS else None


def build_assets(spec: SceneSpec) -> list:
    assets = []
    for o in spec.objects:
        xyz, yaw = o.xyz, o.yaw
        if o.kind == "tshirt":
            assets.append(O.tshirt(o.name, o.variant or "m_blue", xyz, yaw))
        elif o.kind == "towel":
            assets.append(O.towel(o.name, xyz, yaw, small=(o.variant == "small"), color=_towel_color(o)))
        elif o.kind == "sock":
            assets.append(O.sock(o.name, xyz, yaw))
        elif o.kind == "duvet":
            assets.append(O.duvet(o.name, xyz, yaw))
        elif o.kind == "pillow":
            assets.append(O.pillow(o.name, xyz, yaw))
        elif o.kind == "basket":
            assets.append(O.basket(o.name, xyz, yaw, o.variant))
        elif o.kind == "doll":
            assets.append(O.doll(o.name, xyz, yaw))
        elif o.kind == "book":
            assets.append(O.book(o.name, xyz, yaw))
        else:
            raise ValueError(o.kind)
    return assets


# episode limit (sim s), default physics rate (D-59, D-62: BED-02 / BED-03 120 Hz, the duvet 480), observer offset
register_task("BED-01", TaskEntry(Bed01MakeBedTask, 240.0, 480, build_assets, (1.6, -2.2, 1.9), room_scale=True))
register_task("BED-02", TaskEntry(Bed02FoldTask, 240.0, 120, build_assets, (1.2, -1.4, 1.1)))
register_task("BED-03", TaskEntry(Bed03LaundryTask, 180.0, 120, build_assets, (3.3, -0.9, 1.6), room_scale=True))


class DxBed01Environment(RoomEnvironmentBase):
    name = "dx_bed01"
    task_key = "BED-01"


class DxBed02Environment(RoomEnvironmentBase):
    name = "dx_bed02"
    task_key = "BED-02"


class DxBed03Environment(RoomEnvironmentBase):
    name = "dx_bed03"
    task_key = "BED-03"
