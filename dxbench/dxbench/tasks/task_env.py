"""Room-independent parts of the benchmark task environments: scene JSON -> Arena environment, shared events, the task
registry and the base task / environment classes. A room (assets/scenes.py `ROOMS`) gives the background, its layout
JSON and its prestartup events; a task family (tasks/bed_tasks.py, ...) registers its task ids with `register_task`
and subclasses `RoomTaskBase` / `RoomEnvironmentBase`.

Generic scripts (run_episode, run_teleop, the recorder, the oracles) read the task object of the env built in this
process from `task_env.CURRENT_TASK` and aim the third-person camera with `aim_observer`.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

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
from dxbench.assets.scenes import ROOMS
from dxbench.embodiments.mdp.cloth_assist import add_base_cloth_filter, add_cloth_assist
from dxbench.embodiments.mdp.pad_grasp import add_pad_grasp
from dxbench.environments.rby1_envs import physics_callback
from dxbench.paths import REPO
from dxbench.tasks.scene_spec import SceneSpec, load_layout

CLOTH_KINDS = ("tshirt", "towel", "sock", "duvet")
CURRENT_TASK = None  # the task object of the env built in this process (read by scripts/run_episode.py, the oracles)


@dataclass
class TaskEntry:
    """A registered task id: its task class, episode limit (sim s), default physics rate (Hz), the scene-object ->
    Arena-asset builder and the third-person camera offset (m) from its target."""
    task_cls: type
    episode_s: float
    physics_hz: int
    build_assets: Callable
    observer_eye: tuple = (1.3, 1.3, 0.9)
    room_scale: bool = False  # aim the observer at the robot / objects / garments together (room-scale tasks)
    extra: dict = field(default_factory=dict)


TASKS: dict[str, TaskEntry] = {}


def register_task(task_id: str, entry: TaskEntry) -> None:
    TASKS[task_id] = entry


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


# ------------------------------------------------------------------------------------------ shared events
def solver_iterations(env, env_ids):
    """Prestartup (D-54): PhysX solver position iterations of every deformable (garments, towel, duvet, pillow) and
    every enabled non-robot rigid body (basket, ...) = `objects.SOLVER_ITERS`. The GPU solver runs the scene-wide
    maximum for all bodies, so one 16 (the basket's USD default, the tee USD) made the robot solve 16 + 1 times.
    Experiment overrides (2026-10-28 cost study; benchmark runs leave them unset): DX_CLOTH_ITERS, DX_RIGID_ITERS,
    DX_CLOTH_SELFCOL = 0/1 self-collision, DX_CLOTH_PAIRFREQ = collision pair updates per step (0 = adaptive)."""
    import os

    import isaaclab.sim as sim_utils
    from pxr import PhysxSchema, Sdf, Usd, UsdPhysics

    from dxbench.assets.objects import SOLVER_ITERS

    cloth = {"solverPositionIterationCount": (Sdf.ValueTypeNames.Int, int(os.environ.get("DX_CLOTH_ITERS", SOLVER_ITERS)))}
    for env_key, attr, typ, conv in (("DX_CLOTH_SELFCOL", "selfCollision", Sdf.ValueTypeNames.Bool, lambda v: bool(int(v))),
                                     ("DX_CLOTH_PAIRFREQ", "collisionPairUpdateFrequency", Sdf.ValueTypeNames.Int, int)):
        if os.environ.get(env_key):
            cloth[attr] = (typ, conv(os.environ[env_key]))
    rigid_it = int(os.environ.get("DX_RIGID_ITERS", SOLVER_ITERS))
    stage = sim_utils.get_current_stage()
    nd, rigid = 0, []
    for p in Usd.PrimRange(stage.GetPrimAtPath("/World")):
        schemas = p.GetAppliedSchemas()
        if "OmniPhysicsDeformableBodyAPI" in schemas and "PhysxBaseDeformableBodyAPI" in schemas:
            for attr, (typ, v) in cloth.items():
                p.CreateAttribute(f"physxDeformableBody:{attr}", typ).Set(v)
            nd += 1
        elif p.HasAPI(UsdPhysics.RigidBodyAPI) and "/Robot" not in str(p.GetPath()):
            if UsdPhysics.RigidBodyAPI(p).GetRigidBodyEnabledAttr().Get() is False:
                continue
            api = PhysxSchema.PhysxRigidBodyAPI(p) if p.HasAPI(PhysxSchema.PhysxRigidBodyAPI) else \
                PhysxSchema.PhysxRigidBodyAPI.Apply(p)
            api.CreateSolverPositionIterationCountAttr().Set(rigid_it)
            rigid.append(p.GetName())
    print(f"[solver_iterations] {nd} deformable(s) {cloth}, rigid {rigid} -> {rigid_it}", flush=True)


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
class RoomTaskBase(TaskBase):
    """A scene-JSON task in one of the `ROOMS`. Subclasses set `task_id` and `room` and implement `evaluate(env)` ->
    one dict per env with "success", "failure" (None or a reason), "score" and "stages"."""
    task_id = "TASK"
    room = "care_room"

    def __init__(self, spec: SceneSpec, episode_length_s: float):
        super().__init__(episode_length_s=episode_length_s, task_description=spec.language.get("high"))
        self.spec = spec
        self.cloth_names = [o.name for o in spec.objects if o.kind in CLOTH_KINDS]
        self.layout = load_layout(self.room)
        self.last_eval: list[dict] = []
        self._eval_key = None
        self._eval_cache: list[dict] = []

    # --- Arena interface
    def get_scene_cfg(self):
        return None

    def get_events_cfg(self):
        states = {o.name: str(_resolve(o.cloth_state)) for o in self.spec.objects if o.cloth_state}
        # every term must be a class field: Arena's combine_configclass_instances copies dataclass fields only, so
        # the old `ev.load_cloth_states = ...` instance attribute was dropped and every episode since stage 3 started
        # from the spawn shape, not the saved state (found 2026-10-21, D-42). Task events come after the scene's
        # per-object resets, so the saved nodes overwrite the default ones.
        fields = dict(ROOMS[self.room].prestartup(self.spec))
        fields["solver_iterations"] = EventTermCfg(func=solver_iterations, mode="prestartup")  # D-54
        fields["load_cloth_states"] = EventTermCfg(func=load_cloth_states, mode="reset", params={"states": states})
        events = configclass(type("_Events", (), {"__annotations__": {k: EventTermCfg for k in fields}, **fields}))
        return events()

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

    def evaluate_cached(self, env) -> list[dict]:
        """evaluate() once per env state: the success predicate, the failure term and the teleop loop asked for the
        same verdict three times per control step (2026-10-28). Key: physics step count + episode lengths (a reset
        within the step sets them to 0)."""
        key = (env._sim_step_counter, tuple(env.episode_length_buf.tolist()))
        if key != self._eval_key:
            self._eval_cache, self._eval_key = self.evaluate(env), key
        return self._eval_cache

    def _success_pred(self, env) -> torch.Tensor:
        self.last_eval = self.evaluate_cached(env)
        return torch.tensor([r["success"] for r in self.last_eval], device=env.device)

    def _failure_term(self, env) -> torch.Tensor:
        res = self.last_eval or self.evaluate_cached(env)
        return torch.tensor([r["failure"] is not None for r in res], device=env.device)

    def subtask_labels(self) -> list[str]:
        return list(self.spec.language.get("subtasks", []))


# ------------------------------------------------------------------------------------------ environment factory
class RoomEnvironmentBase(ExampleEnvironmentBase):
    """Builds the Arena env of a registered task from `--scene`: the task's room background, the robot at the scene's
    start pose, the scene objects (the task's `build_assets`), the assisted cloth grasp when there is cloth, and the
    optional third-person camera. Subclasses set `name` and `task_key`."""
    name: str = "dx_task"
    task_key = "TASK"

    def get_env(self, args_cli: argparse.Namespace):
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.utils.pose import Pose

        spec = SceneSpec.from_json(_resolve(args_cli.scene))
        assert spec.task == self.task_key, f"scene {spec.scene_id} is for {spec.task}, not {self.task_key}"
        entry = TASKS[spec.task]
        room = self.asset_registry.get_asset_by_name(ROOMS[entry.task_cls.room].background)()
        x, y = spec.robot["xy"]
        yaw = float(spec.robot.get("yaw", 0.0))
        emb = self.asset_registry.get_asset_by_name(args_cli.embodiment)(
            enable_cameras=bool(getattr(args_cli, "enable_cameras", False)), preset=args_cli.gains,
            torso=spec.robot.get("torso", "upright"), head_pitch=float(spec.robot.get("head_pitch", 0.0)),
            use_tiled_camera=not args_cli.untiled_cameras,
            initial_pose=Pose(position_xyz=(x, y, 0.0), rotation_xyzw=(0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))))
        task = entry.task_cls(spec, args_cli.episode_s or entry.episode_s)
        global CURRENT_TASK
        CURRENT_TASK = task
        cloth = task.cloth_names
        base_cb = physics_callback(args_cli.physics_hz or entry.physics_hz, args_cli.ik_every_n)

        def cb(env_cfg):
            env_cfg = base_cb(env_cfg)
            if cloth and not args_cli.no_assist:
                if args_cli.grasp_assist == "pad":
                    add_pad_grasp(env_cfg, cloth, squeeze=args_cli.pad_squeeze, tip_margin=args_cli.pad_tip_margin,
                                  capture_c=args_cli.pad_capture_c,
                                  support_cloths=[o.name for o in spec.objects  # D-37, D-51
                                                  if o.kind == "duvet" or args_cli.pad_support == "all"],
                                  support_r=args_cli.pad_support_r, support_gain=args_cli.pad_support_gain,
                                  gain=args_cli.pad_gain, pin_mode=args_cli.pad_pin_mode)
                else:
                    add_cloth_assist(env_cfg, cloth, radius=args_cli.assist_radius)
            if cloth:
                add_base_cloth_filter(env_cfg, cloth)
            if args_cli.observer:  # third-person RTX camera; run_episode aims it (aim_observer) and saves observer.mp4
                import isaaclab.sim as sim_utils
                from isaaclab.sensors import CameraCfg

                env_cfg.scene.observer = CameraCfg(
                    prim_path="{ENV_REGEX_NS}/observer", update_period=0.0, width=args_cli.observer_res[0],
                    height=args_cli.observer_res[1], data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg(focal_length=16.0, clipping_range=(0.05, 20.0)))
            return env_cfg

        return IsaacLabArenaEnvironment(name=self.name, embodiment=emb,
                                        scene=Scene(assets=[room, *entry.build_assets(spec)]), task=task,
                                        env_cfg_callback=cb)

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--scene", required=True, help="scene JSON (path, absolute or relative to the repo)")
        parser.add_argument("--embodiment", default="rby1_ik", choices=["rby1_joint", "rby1_ik"])
        parser.add_argument("--gains", default="tuned", choices=["tuned", "original"])
        parser.add_argument("--physics_hz", type=int, default=None,
                            help="physics rate (Hz); default per task (TaskEntry.physics_hz: BED-02 / BED-03 120, "
                                 "BED-01 480, D-59, D-62)")
        parser.add_argument("--untiled_cameras", action="store_true")
        parser.add_argument("--no_assist", action="store_true", help="disable the assisted cloth grasp")
        parser.add_argument("--assist_radius", type=float, default=0.025)  # D-U11 (cylinder)
        parser.add_argument("--grasp_assist", default="pad", choices=["cylinder", "pad"],  # D-34
                            help="pad: pad_grasp (only cloth between the pads, tips <= 4 mm from it); "
                                 "cylinder: cloth_assist (radius around the TCP axis, D-U11)")
        parser.add_argument("--pad_squeeze", type=float, default=0.0)  # D-34: 0.7 let pinned nodes drift up to 3 cm
        parser.add_argument("--pad_tip_margin", type=float, default=0.004)
        parser.add_argument("--pad_support_r", type=float, default=0.05, help="pad assist: duvet support radius (m)")
        parser.add_argument("--pad_support_gain", type=float, default=1.0)
        parser.add_argument("--pad_gain", type=float, default=1.0)
        parser.add_argument("--pad_pin_mode", default="velocity", choices=["velocity", "posvel"])
        parser.add_argument("--pad_support", default="all", choices=["duvet", "all"],
                            help="pad assist: cloths that get the support ring (D-51: all; D-37: duvet only)")
        parser.add_argument("--pad_capture_c", type=float, default=0.025,  # D-35
                            help="pad assist: catch only cloth within this distance (m) of the closing midplane")
        parser.add_argument("--episode_s", type=float, default=None)
        parser.add_argument("--ik_every_n", type=int, default=1,
                            help="rby1_ik: solve the arm IK on every n-th physics step (1 = every step; teleop 4)")
        parser.add_argument("--observer", action="store_true", help="add a third-person camera (needs --video)")
        parser.add_argument("--observer_res", type=int, nargs=2, default=(960, 540))
        parser.add_argument("--observer_target", type=float, nargs=3, default=None,
                            help="observer look-at point (m, env frame) instead of the task default")
        parser.add_argument("--observer_eye", type=float, nargs=3, default=None,
                            help="observer eye offset (m, world) from its target; default per task (TaskEntry)")


def aim_observer(U, task, eye_offset=None, target=None):
    """Point the `observer` camera from `eye_offset` (default: the task's TaskEntry.observer_eye) at the garments
    (cloth centroid, 10 cm up) or, for room-scale tasks, at the middle of the garments, the basket and the robot."""
    entry = TASKS.get(getattr(task, "task_id", None))
    eye_offset = eye_offset or (entry.observer_eye if entry else (1.3, 1.3, 0.9))
    pts = [U.scene[c].data.nodal_pos_w.torch[0] for c in task.cloth_names] if task.cloth_names else []
    if pts:
        tgt = torch.cat(pts).mean(0, keepdim=True)
    else:
        tgt = U.scene["robot"].data.root_link_pos_w.torch[:1].clone()
    if entry is not None and entry.room_scale:
        keys = [tgt, U.scene["robot"].data.root_link_pos_w.torch[:1]]
        if "laundry_basket" in U.scene.keys():
            keys.append(U.scene["laundry_basket"].data.root_link_pos_w.torch[:1])
        tgt = torch.cat(keys).mean(0, keepdim=True)
        tgt[:, 2] = 0.35
    else:
        tgt[:, 2] += 0.10
    if target is not None:
        tgt = U.scene.env_origins[:1] + torch.tensor([list(target)], device=tgt.device, dtype=tgt.dtype)
    eye = tgt + torch.tensor([list(eye_offset)], device=tgt.device, dtype=tgt.dtype)
    U.scene["observer"].set_world_poses_from_view(eye, tgt)
