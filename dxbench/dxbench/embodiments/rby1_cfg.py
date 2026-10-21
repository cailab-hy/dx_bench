"""RB-Y1 ArticulationCfg for dxbench (promoted from tools/rby1_check/rby1_cfg_draft.py).

Gains (reports/rby1_check/REPORT.md):
  preset "tuned" (default, D-U1): stiffness = original kp, damping for zeta = 0.7 (zero pose), viscous joint
           friction only; the gravity feed-forward is applied by the GravityCompensationAction term.
  preset "original": stiffness = kp, damping = kd * 0.002 (rby1_task.py torque loop at 500 Hz), full
           motor friction model (static + dynamic + viscous).
Wheels: damping-only velocity drive (D-U5). Gripper: l1/r1 and l2/r2 all position-driven (K 100 N/m, D 1);
the action term mirrors l2 = -l1 (software mimic, D-3).
Physics materials (wheel mu 1.2, caster mu 0) live in the USD (assets/rby1/rby1_dx.usda).
Isaac Lab 3.0 quaternions are (x, y, z, w).
"""
from __future__ import annotations

import re

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from dxbench.paths import RBY1_DX_USD

PD_CONTROL_DT = 1.0 / 500.0

WHEELS = ["right_wheel", "left_wheel"]
TORSO = [f"torso_{i}" for i in range(6)]
ARM_R = [f"right_arm_{i}" for i in range(7)]
ARM_L = [f"left_arm_{i}" for i in range(7)]
HEAD = ["head_0", "head_1"]
FINGERS = ["gripper_finger_r1", "gripper_finger_r2", "gripper_finger_l1", "gripper_finger_l2"]
BODY = TORSO + ARM_R + ARM_L + HEAD  # gravity-compensated, position-controlled joints
# rby1-sdk C++ order (rby1-sim-isaac src/config.py CPP_JOINT_NAMES)
CPP_JOINT_NAMES = WHEELS + TORSO + ARM_R + ARM_L + HEAD

KP = {"torso_0": 3911.0, "torso_1": 3911.0, "torso_2": 3911.0, "torso_3": 573.6, "torso_4": 573.6,
      "torso_5": 573.6, "head_0": 39.11, "head_1": 39.11}
KD = {"torso_0": 3520.0, "torso_1": 3520.0, "torso_2": 3520.0, "torso_3": 1043.0, "torso_4": 1043.0,
      "torso_5": 1043.0, "head_0": 91.26, "head_1": 91.26}
for _s in ("right", "left"):
    for _i, (_kp, _kd) in enumerate(zip((208.6, 208.6, 208.6, 91.27, 39.11, 39.11, 39.11),
                                        (521.5, 521.5, 321.5, 208.6, 91.26, 91.26, 61.26))):
        KP[f"{_s}_arm_{_i}"], KD[f"{_s}_arm_{_i}"] = _kp, _kd
DAMPING_ORIGINAL = {k: v * PD_CONTROL_DT for k, v in KD.items()}
# zeta = 0.7 at the zero pose (0-stage T3a study: PhysX mass-matrix diagonal + armature)
DAMPING_ZETA07 = {"torso_0": 579.0, "torso_1": 572.0, "torso_2": 365.0, "torso_3": 78.2, "torso_4": 85.1,
                  "torso_5": 58.3, "head_0": 6.22, "head_1": 6.23}
for _s in ("right", "left"):
    for _i, _d in enumerate((24.8, 26.0, 11.1, 11.3, 3.95, 4.17, 3.93)):
        DAMPING_ZETA07[f"{_s}_arm_{_i}"] = _d

# URDF effort (head: USD 70, URDF 1000 is a placeholder), URDF velocity
EFFORT = {"torso_[0-2]": 270.0, "torso_[3-5]": 120.0, ".*_arm_[0-2]": 70.0, ".*_arm_3": 40.0,
          ".*_arm_[4-5]": 10.0, ".*_arm_6": 8.0, "head_.*": 70.0, ".*_wheel": 100.0}
VEL = {"torso_[0-2]": 2.0943951, "torso_[3-5]": 3.141592654, ".*_arm_[0-3]": 3.141592654,
       ".*_arm_[4-5]": 6.283185308, ".*_arm_6": 2.094395102, "head_.*": 3.14, ".*_wheel": 15.707963268}
# motor model (motor_profiles.py): static, dynamic, viscous N·m·s/rad, armature
PROFILE = {"torso_[0-2]": (10.0, 10.0, 20.0, 1.7), "torso_[3-5]": (10.0, 10.0, 10.0, 1.7),
           ".*_arm_[0-2]": (5.2, 5.2, 4.7, 0.29), ".*_arm_3": (2.2, 2.2, 1.9, 0.4),
           ".*_arm_[4-5]": (2.2, 2.2, 1.9, 0.2), ".*_arm_6": (1.2, 1.2, 1.0, 0.2),
           "head_.*": (2.0, 2.0, 3.0, 0.5), ".*_wheel": (2.0, 2.0, 3.0, 0.5)}
WHEEL_DAMPING = 100.0
GROUPS = {"wheels": WHEELS, "torso": TORSO, "arms": ARM_R + ARM_L, "head": HEAD}

# initial joint positions (rad); torso presets are torso_1/2/3 combined (never bend torso_1 alone)
TORSO_PRESETS = {"upright": (0.0, 0.0, 0.0), "lean": (0.7854, -1.2217, 0.7854),
                 "crouch": (1.0472, -1.5708, 1.0472), "deep": (1.5708, -2.2689, 1.5708)}
READY_ARMS = {"right_arm_0": -0.5, "right_arm_1": -0.2, "right_arm_3": -1.6, "right_arm_5": 0.3,
              "left_arm_0": -0.5, "left_arm_1": 0.2, "left_arm_3": -1.6, "left_arm_5": 0.3}


def _sel(table: dict, names: list[str]) -> dict:
    return {k: v for k, v in table.items() if any(re.fullmatch(k, n) for n in names)}


def default_joint_pos(torso: str = "upright", arms: str = "ready", gripper_open: float = 1.0,
                      head_pitch: float = 0.0) -> dict[str, float]:
    """Explicit per-joint initial positions (no overlapping regex keys). head_pitch: head_1 (rad, + looks down); no
    action drives the head, so it holds this for the whole episode."""
    q = {n: 0.0 for n in CPP_JOINT_NAMES}
    q["head_1"] = head_pitch
    t1, t2, t3 = TORSO_PRESETS[torso]
    q.update({"torso_1": t1, "torso_2": t2, "torso_3": t3})
    if arms == "ready":
        q.update(READY_ARMS)
    o = 0.05 * gripper_open
    q.update({"gripper_finger_r1": -o, "gripper_finger_r2": o, "gripper_finger_l1": -o, "gripper_finger_l2": o})
    return q


def make_rby1_cfg(preset: str = "tuned", usd_path: str | None = None, torso: str = "upright",
                  arms: str = "ready", self_collision: bool = False, head_pitch: float = 0.0) -> ArticulationCfg:
    actuators = {}
    for g, names in GROUPS.items():
        if g == "wheels":
            stiff, damp = {n: 0.0 for n in names}, {n: WHEEL_DAMPING for n in names}
        else:
            stiff = {n: KP[n] for n in names}
            damp = {n: (DAMPING_ZETA07 if preset == "tuned" else DAMPING_ORIGINAL)[n] for n in names}
        full = preset == "original"
        prof = _sel(PROFILE, names)
        actuators[g] = ImplicitActuatorCfg(
            joint_names_expr=names, stiffness=stiff, damping=damp,
            armature={k: v[3] for k, v in prof.items()},
            friction={k: v[0] for k, v in prof.items()} if full else 0.0,
            dynamic_friction={k: v[1] for k, v in prof.items()} if full else 0.0,
            viscous_friction={k: v[2] for k, v in prof.items()},
            joint_effort_limit=_sel(EFFORT, names), joint_velocity_limit=_sel(VEL, names))
    actuators["gripper"] = ImplicitActuatorCfg(joint_names_expr=FINGERS, stiffness=100.0, damping=1.0,
                                               joint_effort_limit=75.0, joint_velocity_limit=0.05)
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd_path or RBY1_DX_USD),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, linear_damping=0.0,
                                                         angular_damping=0.0, max_linear_velocity=200.0,
                                                         max_angular_velocity=1000.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=self_collision, solver_position_iteration_count=4,
                solver_velocity_iteration_count=1)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), joint_pos=default_joint_pos(torso, arms, head_pitch=head_pitch)),
        soft_joint_pos_limit_factor=1.0,
        actuators=actuators,
    )
