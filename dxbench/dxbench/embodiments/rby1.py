"""RB-Y1 embodiments registered with IsaacLab-Arena.

  rby1_joint (policy):  action 24 = right arm 7 + left arm 7 + gripper [right, left] (rby1_format 16-D, absolute)
                        + torso 6 (absolute) + base (v, omega). See docs/action_space.md.
  rby1_ik (teleop):     action 17 = right TCP delta 6 + left TCP delta 6 (pos m, rot axis-angle rad, base frame,
                        Isaac Lab differential IK, relative mode) + gripper 2 + torso preset step 1 + base (v, omega).
Both add a zero-dimensional gravity-compensation term (preset "tuned", D-U1).
Observations: group `policy` (state16, torso, joint vel, base velocity, TCP poses, gripper) and, with cameras,
`camera_obs` {head, left_wrist, right_wrist}_rgb 480x640.
"""
from __future__ import annotations

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import mdp as lab_mdp
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg, JointPositionActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.camera.camera_cfg import CameraCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.utils.cameras import ArenaCameraCfg

from dxbench.embodiments.mdp import actions as A
from dxbench.embodiments.mdp import observations as O
from dxbench.embodiments.rby1_cfg import ARM_L, ARM_R, TORSO, make_rby1_cfg

TCP_OFFSET = (0.0, 0.0, -0.1045)
JOINT_ACTION_NAMES = ARM_R + ARM_L + ["gripper_right", "gripper_left"] + TORSO + ["base_v", "base_omega"]
IK_ACTION_NAMES = ([f"right_tcp_{k}" for k in ("dx", "dy", "dz", "rx", "ry", "rz")]
                   + [f"left_tcp_{k}" for k in ("dx", "dy", "dz", "rx", "ry", "rz")]
                   + ["gripper_right", "gripper_left", "torso_preset_step", "base_v", "base_omega"])


@configclass
class RBY1SceneCfg:
    robot = make_rby1_cfg()


@configclass
class RBY1CameraCfg(ArenaCameraCfg):
    head = CameraCfg(prim_path="{ENV_REGEX_NS}/Robot/link_head_2/first_person_camera", spawn=None,
                     width=640, height=480, data_types=["rgb"], update_latest_camera_pose=True)
    left_wrist = CameraCfg(prim_path="{ENV_REGEX_NS}/Robot/left_gripper/ee_left/left_hand_camera", spawn=None,
                           width=640, height=480, data_types=["rgb"], update_latest_camera_pose=True)
    right_wrist = CameraCfg(prim_path="{ENV_REGEX_NS}/Robot/right_gripper/ee_right/right_hand_camera", spawn=None,
                            width=640, height=480, data_types=["rgb"], update_latest_camera_pose=True)


@configclass
class RBY1JointActionsCfg:
    arms = JointPositionActionCfg(asset_name="robot", joint_names=ARM_R + ARM_L, preserve_order=True,
                                  use_default_offset=False, scale=1.0)
    gripper = A.ParallelGripperActionCfg(asset_name="robot")
    torso = JointPositionActionCfg(asset_name="robot", joint_names=TORSO, preserve_order=True,
                                   use_default_offset=False, scale=1.0)
    base = A.DiffDriveBaseActionCfg(asset_name="robot")
    gravity = A.GravityCompensationActionCfg(asset_name="robot")


def _ik(side: str) -> A.HomingDiffIKActionCfg:
    # request_home() is the joint-space way back to READY (teleop home key, oracles): the relative IK alone got
    # stuck near joint limits after placing (stage 3). Isaac Lab's null-space joint-limit avoidance
    # (joint_limit_avoidance_gain) was tried: it only protects the position rows and tilted the gripper (t5 gripper
    # lift 14.4 -> 10.3 cm), so it stays off.
    return A.HomingDiffIKActionCfg(
        asset_name="robot", joint_names=[f"{side}_arm_[0-6]"], body_name=f"ee_{side}",
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=TCP_OFFSET),
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=1.0)


@configclass
class RBY1IKActionsCfg:
    right_arm = _ik("right")
    left_arm = _ik("left")
    gripper = A.ParallelGripperActionCfg(asset_name="robot")
    torso = A.TorsoPresetActionCfg(asset_name="robot")
    base = A.DiffDriveBaseActionCfg(asset_name="robot")
    gravity = A.GravityCompensationActionCfg(asset_name="robot")


@configclass
class RBY1ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        state16 = ObsTerm(func=O.state16)
        torso = ObsTerm(func=O.torso_pos)
        joint_vel = ObsTerm(func=O.joint_vel_named, params={"names": tuple(ARM_R + ARM_L + TORSO)})
        gripper = ObsTerm(func=O.gripper_opening)
        base_vel = ObsTerm(func=O.base_velocity)
        base_pose = ObsTerm(func=O.base_pose_2d)
        tcp_right = ObsTerm(func=O.tcp_pose_b, params={"side": "right"})
        tcp_left = ObsTerm(func=O.tcp_pose_b, params={"side": "left"})
        last_action = ObsTerm(func=lab_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class RBY1EventCfg:
    reset_robot_joints = EventTerm(func=lab_mdp.reset_joints_by_offset, mode="reset",
                                   params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0),
                                           "asset_cfg": SceneEntityCfg("robot")})


class _RBY1Base(EmbodimentBase):
    default_arm_mode = ArmMode.DUAL_ARM

    def __init__(self, enable_cameras: bool = False, initial_pose=None, concatenate_observation_terms: bool = False,
                 arm_mode=None, preset: str = "tuned", torso: str = "upright", use_tiled_camera: bool = True,
                 usd_path: str | None = None, **kwargs):
        super().__init__(enable_cameras, initial_pose, concatenate_observation_terms, arm_mode, **kwargs)
        self.scene_config = RBY1SceneCfg()
        self.scene_config.robot = make_rby1_cfg(preset=preset, torso=torso, usd_path=usd_path)
        self.camera_config = RBY1CameraCfg()
        self.camera_config.set_use_tiled_camera(use_tiled_camera)
        self.observation_config = RBY1ObservationsCfg()
        self.event_config = RBY1EventCfg()
        self.mimic_env = None
        self.preset = preset
        self.torso = torso

    def get_ee_frame_transformer_names(self) -> list[str]:
        return []

    def get_action_cfg(self):
        cfg = super().get_action_cfg()
        if self.preset != "tuned":
            cfg.gravity.scale = 0.0
        if hasattr(getattr(cfg, "torso", None), "initial"):  # preset action starts where the robot spawns
            cfg.torso.initial = self.torso
        return cfg


@register_asset
class RBY1JointEmbodiment(_RBY1Base):
    name = "rby1_joint"
    action_names = JOINT_ACTION_NAMES

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_config = RBY1JointActionsCfg()


@register_asset
class RBY1IKEmbodiment(_RBY1Base):
    name = "rby1_ik"
    action_names = IK_ACTION_NAMES

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_config = RBY1IKActionsCfg()

