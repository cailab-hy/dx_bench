"""Arena environments for the RB-Y1 embodiment (no task): flat-ground regression and the DX care room.

    python submodules/IsaacLab-Arena/isaaclab_arena/evaluation/policy_runner.py --policy_type zero_action \
        --num_steps 30 --external_environment_class_path dxbench.environments.rby1_envs:DxBedroomEnvironment \
        dx_bedroom --embodiment rby1_joint

Importing this module registers the dxbench assets (embodiments, scenes).
"""
from __future__ import annotations

import argparse

from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

import dxbench.assets.scenes  # noqa: F401  (registers backgrounds)
import dxbench.embodiments.rby1  # noqa: F401  (registers rby1_joint / rby1_ik)
from dxbench.paths import CARE_SPAWN

CTRL_HZ = 15


def physics_callback(physics_hz: int):
    """env_cfg_callback: 15 Hz control (decimation = physics_hz / 15), PhysX GPU (backend kept)."""
    assert physics_hz % CTRL_HZ == 0, "physics rate must be a multiple of 15 Hz"

    def cb(env_cfg):
        env_cfg.sim.dt = 1.0 / physics_hz
        env_cfg.decimation = physics_hz // CTRL_HZ
        env_cfg.sim.render_interval = env_cfg.decimation
        phys = env_cfg.sim.physics
        for k, v in (("gpu_max_rigid_contact_count", 2**23), ("gpu_max_rigid_patch_count", 2**21),
                     ("gpu_max_soft_body_contacts", 2**21), ("gpu_heap_capacity", 2**27),
                     ("gpu_found_lost_pairs_capacity", 2**22)):
            if hasattr(phys, k):
                setattr(phys, k, v)
        return env_cfg

    return cb


def _common_args(parser: argparse.ArgumentParser):
    parser.add_argument("--embodiment", default="rby1_joint", choices=["rby1_joint", "rby1_ik"])
    parser.add_argument("--gains", default="tuned", choices=["tuned", "original"])
    parser.add_argument("--torso", default="upright", choices=["upright", "lean", "crouch", "deep"])
    parser.add_argument("--physics_hz", type=int, default=240)
    parser.add_argument("--untiled_cameras", action="store_true")
    parser.add_argument("--robot_usd", default=None, help="override the robot USD (debugging)")


class _Base(ExampleEnvironmentBase):
    def _embodiment(self, args_cli, initial_pose=None):
        from isaaclab_arena.utils.pose import Pose

        cls = self.asset_registry.get_asset_by_name(args_cli.embodiment)
        return cls(enable_cameras=bool(getattr(args_cli, "enable_cameras", False)), preset=args_cli.gains,
                   torso=args_cli.torso, use_tiled_camera=not args_cli.untiled_cameras, usd_path=args_cli.robot_usd,
                   initial_pose=initial_pose or Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))


class DxFlatEnvironment(_Base):
    name: str = "dx_rby1_flat"

    def get_env(self, args_cli: argparse.Namespace):
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.no_task import NoTask

        ground = self.asset_registry.get_asset_by_name("dx_flat_ground")()
        light = self.asset_registry.get_asset_by_name("light")()
        assets = [ground, light]
        if args_cli.props == "grasp_cube":
            assets += grasp_cube_props(tuple(float(v) for v in args_cli.cube_xyz.split(",")))
        return IsaacLabArenaEnvironment(name=self.name, embodiment=self._embodiment(args_cli),
                                        scene=Scene(assets=assets), task=NoTask(),
                                        env_cfg_callback=physics_callback(args_cli.physics_hz))

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        _common_args(parser)
        parser.add_argument("--props", default="none", choices=["none", "grasp_cube"])
        parser.add_argument("--cube_xyz", default="0.50,-0.25,0.90", help="grasp cube centre in the env frame (m)")


def grasp_cube_props(cube_xyz=(0.50, -0.25, 0.90)):
    """5 cm / 0.1 kg cube on a 3 x 3 cm kinematic pedestal, centre 0.90 m. With the gripper pointing down the right
    arm loses the orientation above TCP z ~1.05 m (wrist arm_5 upper limit 1.92 rad) and IK cannot reach below
    ~0.85 m at x 0.5 m (0.80 m: 6.7 cm reach error) - stage 2 t5 gripper, DECISIONS D-17."""
    import isaaclab.sim as sim_utils
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_base import ObjectType
    from isaaclab_arena.utils.pose import Pose

    cube = Object(name="grasp_cube", object_type=ObjectType.RIGID,
                  spawner_cfg=sim_utils.CuboidCfg(
                      size=(0.05, 0.05, 0.05), rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                      mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
                      collision_props=sim_utils.CollisionPropertiesCfg(),
                      physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
                      visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.2, 0.2))),
                  initial_pose=Pose(position_xyz=cube_xyz, rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    top = cube_xyz[2] - 0.025
    ped = Object(name="pedestal", object_type=ObjectType.RIGID,
                 spawner_cfg=sim_utils.CuboidCfg(
                     size=(0.03, 0.03, top), rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                     collision_props=sim_utils.CollisionPropertiesCfg(),
                     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.8))),
                 initial_pose=Pose(position_xyz=(cube_xyz[0], cube_xyz[1], top / 2), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    return [cube, ped]


class DxBedroomEnvironment(_Base):
    name: str = "dx_bedroom"

    def get_env(self, args_cli: argparse.Namespace):
        import math

        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.no_task import NoTask
        from isaaclab_arena.utils.pose import Pose

        room = self.asset_registry.get_asset_by_name("dx_care_room")()
        yaw = math.radians(args_cli.yaw)
        pose = Pose(position_xyz=CARE_SPAWN, rotation_xyzw=(0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)))
        return IsaacLabArenaEnvironment(name=self.name, embodiment=self._embodiment(args_cli, pose),
                                        scene=Scene(assets=[room]), task=NoTask(),
                                        env_cfg_callback=physics_callback(args_cli.physics_hz))

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        _common_args(parser)
        parser.add_argument("--yaw", type=float, default=0.0)
