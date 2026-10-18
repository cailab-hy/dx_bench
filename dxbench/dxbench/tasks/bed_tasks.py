"""BED-01/02/03 Arena tasks + the environment factory `DxBedEnvironment` (dx_bed01 / dx_bed02 / dx_bed03).

    python submodules/IsaacLab-Arena/isaaclab_arena/evaluation/policy_runner.py --policy_type zero_action --num_episodes 1 \
        --external_environment_class_path dxbench.tasks.bed_tasks:DxBed03Environment dx_bed03 \
        --scene benchmarks/dx_bed_v0/scenes/bed03_easy_000.json

A scene JSON (dxbench.tasks.scene_spec) fixes the objects, cloth states, robot start and language. The DX care-room
background keeps its furniture; its own blanket / pillow / desk T-shirt are deactivated before the simulation
starts and the benchmark garments are spawned as Arena deformables (so the task can read and reset them).
Grasping cloth uses the assisted grasp (embodiments/mdp/cloth_assist.py, DECISIONS D-4/D-5).
Scores: dxbench.tasks.predicates (pure numpy, unit-tested).
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg, TerminationTermCfg
from isaaclab.utils import configclass

from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

import dxbench.assets.scenes  # noqa: F401
import dxbench.embodiments.rby1  # noqa: F401
from dxbench.assets import objects as O
from dxbench.assets.scenes import replace_care_room_floor
from dxbench.embodiments.mdp.cloth_assist import add_base_cloth_filter, add_cloth_assist
from dxbench.environments.rby1_envs import physics_callback
from dxbench.paths import CARE_FURNITURE, REPO
from dxbench.tasks import predicates as P
from dxbench.tasks.scene_spec import SceneSpec, load_layout

CLOTH_KINDS = ("tshirt", "towel", "sock", "duvet")
CURRENT_TASK = None  # the task object of the env built in this process (read by scripts/run_episode.py)
DX_DEFORMABLES = ("Bed/Blanket", "Bed/Pillow", "TShirt")


def _T(x):
    import warp as wp

    if hasattr(x, "torch"):
        return x.torch
    return wp.to_torch(x) if isinstance(x, wp.array) else x


def _resolve(path: str | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    return p if p.is_absolute() else (REPO / p)


# ------------------------------------------------------------------------------------------ events
def deactivate_dx_deformables(env, env_ids):
    """Prestartup: switch off the DX bed blanket / pillow / desk T-shirt (the benchmark spawns its own)."""
    import isaaclab.sim as sim_utils

    stage = sim_utils.get_current_stage()
    for rel in DX_DEFORMABLES:
        p = stage.GetPrimAtPath(f"{CARE_FURNITURE}/{rel}")
        if p.IsValid():
            p.SetActive(False)


def load_cloth_states(env, env_ids, states: dict):
    """Reset: write saved (settled) nodal positions for each garment, zero velocities. states: {name: npz path}."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    for name, path in states.items():
        obj = env.scene[name]
        nodes = torch.as_tensor(np.load(path)["nodes"], dtype=torch.float32, device=env.device)
        pos = _T(obj.data.nodal_pos_w).clone()
        vel = _T(obj.data.nodal_vel_w).clone()
        for e in env_ids.tolist() if hasattr(env_ids, "tolist") else env_ids:
            pos[e] = nodes + (env.scene.env_origins[e] - env.scene.env_origins[0])
            vel[e] = 0.0
        obj.write_nodal_pos_to_sim_index(pos)
        obj.write_nodal_velocity_to_sim_index(vel)


# ------------------------------------------------------------------------------------------ task base
class BedTaskBase(TaskBase):
    task_id = "BED-0X"

    def __init__(self, spec: SceneSpec, episode_length_s: float):
        super().__init__(episode_length_s=episode_length_s, task_description=spec.language.get("high"))
        self.spec = spec
        self.cloth_names = [o.name for o in spec.objects if o.kind in CLOTH_KINDS]
        self.layout = load_layout()
        self.last_eval: list[dict] = []

    # --- Arena interface
    def get_scene_cfg(self):
        return None

    def get_events_cfg(self):
        states = {o.name: str(_resolve(o.cloth_state)) for o in self.spec.objects if o.cloth_state}

        @configclass
        class _Events:
            deactivate_dx = EventTermCfg(func=deactivate_dx_deformables, mode="prestartup")
            floor_plane = EventTermCfg(func=replace_care_room_floor, mode="prestartup")

        ev = _Events()
        if states:
            ev.load_cloth_states = EventTermCfg(func=load_cloth_states, mode="reset", params={"states": states})
        return ev

    def get_termination_cfg(self) -> TaskTerminationCfg:
        return TaskTerminationCfg(
            timeout_s=self.episode_length_s,
            success=[CompletionCriteria(name="task_success", predicate_sequence=[self._success_pred],
                                        description=self.spec.language.get("high"))],
            failures={"task_failure": TerminationTermCfg(func=self._failure_term)})

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        raise NotImplementedError

    def get_metrics(self):
        return [SuccessRateMetric()]

    def get_viewer_cfg(self) -> ViewerCfg:
        x, y = self.spec.robot["xy"]
        return ViewerCfg(eye=(x - 1.8, y - 1.8, 2.2), lookat=(x + 0.8, y + 0.6, 0.4))

    # --- evaluation
    def nodes(self, env, name: str, e: int) -> np.ndarray:
        return (_T(env.scene[name].data.nodal_pos_w)[e] - env.scene.env_origins[e]).cpu().numpy()

    def rigid_pose(self, env, name: str, e: int):
        o = env.scene[name]
        return ((_T(o.data.root_link_pos_w)[e] - env.scene.env_origins[e]).cpu().numpy(),
                _T(o.data.root_link_quat_w)[e].cpu().numpy())

    def evaluate(self, env) -> list[dict]:
        raise NotImplementedError

    def _success_pred(self, env) -> torch.Tensor:
        self.last_eval = self.evaluate(env)
        return torch.tensor([r["success"] for r in self.last_eval], device=env.device)

    def _failure_term(self, env) -> torch.Tensor:
        res = self.last_eval or self.evaluate(env)
        return torch.tensor([r["failure"] is not None for r in res], device=env.device)

    def subtask_labels(self) -> list[str]:
        return list(self.spec.language.get("subtasks", []))


# ------------------------------------------------------------------------------------------ BED-03
class Bed03LaundryTask(BedTaskBase):
    task_id = "BED-03"

    def evaluate(self, env) -> list[dict]:
        clothes = [o.name for o in self.spec.objects if o.kind in CLOTH_KINDS and o.variant != "folded"]
        folded = [o.name for o in self.spec.objects if o.kind in CLOTH_KINDS and o.variant == "folded"]
        dis = [o.name for o in self.spec.objects if o.kind in ("doll", "book")]
        out = []
        for e in range(env.num_envs):
            bp, bq = self.rigid_pose(env, "laundry_basket", e)
            out.append(P.evaluate_bed03({c: self.nodes(env, c, e) for c in clothes},
                                        {d: self.rigid_pose(env, d, e) for d in dis}, bp, bq,
                                        folded={c: self.nodes(env, c, e) for c in folded}))
        return out


# ------------------------------------------------------------------------------------------ BED-02
class Bed02FoldTask(BedTaskBase):
    task_id = "BED-02"

    def __init__(self, spec, episode_length_s):
        super().__init__(spec, episode_length_s)
        desk = self.layout["parts_aabb"].get("CareDesk") or self.layout["furniture_aabb"]["CareDesk"]
        self.table_z = float(desk[1][2])
        lo, hi = np.array(desk[0]), np.array(desk[1])
        # stack spot: 22 cm in from the desk front edge, 30 cm left of the desk centre line as seen from the robot
        # (robot faces -x from the +x side; its left is -y). Reachable by the right arm from the desk front (the far
        # corners of the desk are ~1 m from the base) and where the two folds leave the garment (stage 3).
        cy = 0.5 * float(lo[1] + hi[1])
        self.stack_xy = (float(hi[0] - 0.22), cy - 0.30)

    def evaluate(self, env) -> list[dict]:
        out = []
        target = self.cloth_names[0] if self.cloth_names else None
        for e in range(env.num_envs):
            nodes = self.nodes(env, target, e)
            o = self.spec.obj(target)
            fl = self._flat_dims(o)
            rest = fl[0] * fl[1] * (0.65 if o.kind == "tshirt" else 1.0)  # T-shirt silhouette fills ~65 % of its bbox
            spec = P.FoldSpec(rest_area=rest, stack_center=self.stack_xy, table_z=self.table_z)
            out.append(P.evaluate_bed02(nodes, spec, folds_done=self._fold_flags(nodes, o)))
        return out

    @staticmethod
    def _flat_dims(o):
        """Flat (long, short) size of the garment variant (m): towels 0.7 x 0.45 / small 0.5 x 0.3, T-shirt 0.93 x 0.66
        scaled 0.85 / 1.0 / 1.15 for s / m / l variants (assets/make_objects.py)."""
        if o.kind == "towel":
            return (0.5, 0.3) if o.variant == "small" else (0.7, 0.45)
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
                             long_axis=1 if (bed[1][1] - bed[0][1]) > (bed[1][0] - bed[0][0]) else 0)
        self._pillow0 = None
        self._moved = None

    def evaluate(self, env) -> list[dict]:
        out = []
        n = env.num_envs
        if self._pillow0 is None or env.episode_length_buf.max() <= 1:
            self._pillow0 = [self.nodes(env, "pillow", e).mean(axis=0) for e in range(n)]
            self._moved = [False] * n
        for e in range(n):
            pc = self.nodes(env, "pillow", e).mean(axis=0)
            if np.linalg.norm(pc[:2] - self._pillow0[e][:2]) > 0.3:
                self._moved[e] = True
            out.append(P.evaluate_bed01(self.nodes(env, "duvet", e), pc, self._pillow0[e], self.bed,
                                        pillow_was_moved=self._moved[e]))
        return out


TASKS = {"BED-01": (Bed01MakeBedTask, 240.0), "BED-02": (Bed02FoldTask, 240.0), "BED-03": (Bed03LaundryTask, 180.0)}


# ------------------------------------------------------------------------------------------ assets from spec
def build_assets(spec: SceneSpec) -> list:
    assets = []
    for o in spec.objects:
        xyz, yaw = o.xyz, o.yaw
        if o.kind == "tshirt":
            assets.append(O.tshirt(o.name, o.variant or "m_blue", xyz, yaw))
        elif o.kind == "towel":
            assets.append(O.towel(o.name, xyz, yaw, small=(o.variant == "small")))
        elif o.kind == "sock":
            assets.append(O.sock(o.name, xyz, yaw))
        elif o.kind == "duvet":
            assets.append(O.rect_cloth(o.name, (1.5, 1.4), xyz, yaw, rgb=(0.85, 0.9, 0.95)))
        elif o.kind == "pillow":
            assets.append(O.pillow(o.name, xyz, yaw))
        elif o.kind == "basket":
            assets.append(O.basket(o.name, xyz, yaw))
        elif o.kind == "doll":
            assets.append(O.doll(o.name, xyz, yaw))
        elif o.kind == "book":
            assets.append(O.book(o.name, xyz, yaw))
        else:
            raise ValueError(o.kind)
    return assets


class _DxBedEnvironmentBase(ExampleEnvironmentBase):
    name: str = "dx_bed"
    task_key = "BED-03"

    def get_env(self, args_cli: argparse.Namespace):
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.utils.pose import Pose

        spec = SceneSpec.from_json(_resolve(args_cli.scene))
        assert spec.task == self.task_key, f"scene {spec.scene_id} is for {spec.task}, not {self.task_key}"
        room = self.asset_registry.get_asset_by_name("dx_care_room")()
        x, y = spec.robot["xy"]
        yaw = float(spec.robot.get("yaw", 0.0))
        emb = self.asset_registry.get_asset_by_name(args_cli.embodiment)(
            enable_cameras=bool(getattr(args_cli, "enable_cameras", False)), preset=args_cli.gains,
            torso=spec.robot.get("torso", "upright"), use_tiled_camera=not args_cli.untiled_cameras,
            initial_pose=Pose(position_xyz=(x, y, 0.0), rotation_xyzw=(0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))))
        task_cls, ep_len = TASKS[spec.task]
        task = task_cls(spec, args_cli.episode_s or ep_len)
        global CURRENT_TASK
        CURRENT_TASK = task
        cloth = task.cloth_names
        base_cb = physics_callback(args_cli.physics_hz)

        def cb(env_cfg):
            env_cfg = base_cb(env_cfg)
            if cloth and not args_cli.no_assist:
                add_cloth_assist(env_cfg, cloth, radius=args_cli.assist_radius)
            if cloth:
                add_base_cloth_filter(env_cfg, cloth)
            return env_cfg

        return IsaacLabArenaEnvironment(name=self.name, embodiment=emb,
                                        scene=Scene(assets=[room, *build_assets(spec)]), task=task,
                                        env_cfg_callback=cb)

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--scene", required=True, help="scene JSON (path, absolute or relative to the repo)")
        parser.add_argument("--embodiment", default="rby1_ik", choices=["rby1_joint", "rby1_ik"])
        parser.add_argument("--gains", default="tuned", choices=["tuned", "original"])
        parser.add_argument("--physics_hz", type=int, default=480)
        parser.add_argument("--untiled_cameras", action="store_true")
        parser.add_argument("--no_assist", action="store_true", help="disable the assisted cloth grasp")
        parser.add_argument("--assist_radius", type=float, default=0.025)  # D-U11
        parser.add_argument("--episode_s", type=float, default=None)


class DxBed01Environment(_DxBedEnvironmentBase):
    name = "dx_bed01"
    task_key = "BED-01"


class DxBed02Environment(_DxBedEnvironmentBase):
    name = "dx_bed02"
    task_key = "BED-02"


class DxBed03Environment(_DxBedEnvironmentBase):
    name = "dx_bed03"
    task_key = "BED-03"
