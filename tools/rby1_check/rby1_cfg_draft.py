"""Minimal RB-Y1 ArticulationCfg used by the rby1_check tests (draft for stage 2).

Import only after AppLauncher has started Kit.

Gain interpretation ("원본 재현 게인", source: rby1-sim-isaac src/rby1_task.py + rby1_controller.py)
    original torque = feedback_gain * (kp * e + kd * de * PD_CONTROL_DT) + ff,  PD_CONTROL_DT = 1/500
    -> stiffness = kp [N·m/rad],  damping = kd * 0.002 [N·m·s/rad]
    feedback_gain defaults to 1; SDK commands set it as clip(gain / 10, 0, 1).

Modes
    implicit  (default) ImplicitActuatorCfg: PhysX joint drive with the gains above.
                        Wheels: damping-only velocity drive (simplified, see WHEEL_DAMPING_IMPLICIT).
    ideal_pd  IdealPDActuatorCfg: explicit PD like the original torque loop, effort clipped to the
              USD maxForce (the original clips to articulation max efforts). Wheels use the original
              integrated-position target + kp/kd (the test loop integrates the target).

Both modes apply the original joint motor model (armature, static/dynamic/viscous friction) from
src/motor_profiles.py, solver iterations 4/1 and self-collision off.

preset="tuned" is the proposal from reports/rby1_check/REPORT.md (not the default; gain changes need sign-off).
Isaac Lab 3.0 quaternions (init_state.rot) are (x, y, z, w).
"""
from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from common import ROBOT_USD

PD_CONTROL_DT = 1.0 / 500.0

# --- original gains (rby1_task.py RBY1_MODEL_CONFIGS["a"]) ---------------------------------------
KP = {
    "right_wheel": 262.8, "left_wheel": 262.8,
    "torso_0": 3911.0, "torso_1": 3911.0, "torso_2": 3911.0, "torso_3": 573.6, "torso_4": 573.6, "torso_5": 573.6,
    "head_0": 39.11, "head_1": 39.11,
}
KD = {
    "right_wheel": 3754.9, "left_wheel": 3754.9,
    "torso_0": 3520.0, "torso_1": 3520.0, "torso_2": 3520.0, "torso_3": 1043.0, "torso_4": 1043.0, "torso_5": 1043.0,
    "head_0": 91.26, "head_1": 91.26,
}
_ARM_KP = (208.6, 208.6, 208.6, 91.27, 39.11, 39.11, 39.11)
_ARM_KD = (521.5, 521.5, 321.5, 208.6, 91.26, 91.26, 61.26)
for side in ("right", "left"):
    for i in range(7):
        KP[f"{side}_arm_{i}"] = _ARM_KP[i]
        KD[f"{side}_arm_{i}"] = _ARM_KD[i]
DAMPING = {k: v * PD_CONTROL_DT for k, v in KD.items()}  # N·m·s/rad

# --- limits ---------------------------------------------------------------------------------------
# URDF (rby1-sdk model_v1.2.urdf) effort; head uses USD 70 (URDF 1000 is a placeholder).
EFFORT_URDF = {
    "torso_[0-2]": 270.0, "torso_[3-5]": 120.0,
    ".*_arm_[0-2]": 70.0, ".*_arm_3": 40.0, ".*_arm_[4-5]": 10.0, ".*_arm_6": 8.0,
    "head_.*": 70.0, ".*_wheel": 100.0,
}
# USD drive maxForce (what the original clips to)
EFFORT_USD = {
    "torso_[0-2]": 841.0, "torso_[3-5]": 369.0,
    ".*_arm_[0-2]": 143.0, ".*_arm_3": 70.0, ".*_arm_[4-5]": 70.0, ".*_arm_6": 140.0,
    "head_.*": 70.0, ".*_wheel": 100.0,
}
VEL_URDF = {
    "torso_[0-2]": 2.0943951, "torso_[3-5]": 3.141592654,
    ".*_arm_[0-3]": 3.141592654, ".*_arm_[4-5]": 6.283185308, ".*_arm_6": 2.094395102,
    "head_.*": 3.14, ".*_wheel": 15.707963268,
}

# --- original motor model (motor_profiles.py) -------------------------------------------------------
# (static_effort, dynamic_effort, viscous N·m·s/rad, armature)
_PROFILE = {
    "torso_[0-2]": (10.0, 10.0, 20.0, 1.7),
    "torso_[3-5]": (10.0, 10.0, 10.0, 1.7),
    ".*_arm_[0-2]": (5.2, 5.2, 4.7, 0.29),
    ".*_arm_3": (2.2, 2.2, 1.9, 0.4),
    ".*_arm_[4-5]": (2.2, 2.2, 1.9, 0.2),
    ".*_arm_6": (1.2, 1.2, 1.0, 0.2),
    "head_.*": (2.0, 2.0, 3.0, 0.5),  # "default" profile
    ".*_wheel": (2.0, 2.0, 3.0, 0.5),  # "default" profile
}

# Implicit wheel velocity drive. Viscous friction (3 N·m·s/rad) acts against it, so the steady-state
# speed is D / (D + 3) of the command; D = 100 keeps that within 3 %. The original instead integrates
# a position target with kp 262.8 / kd_eff 7.51 (no steady-state error).
WHEEL_DAMPING_IMPLICIT = 100.0

# Proposed tuning ("tuned" preset, T3a study): damping for zeta = 0.7 from the PhysX mass-matrix diagonal
# + armature at the zero pose (pose dependent; torso/shoulder values dominate). Used with viscous-only
# friction and a gravity-compensation feedforward (effort target = data.gravity_compensation_forces).
DAMPING_ZETA07 = {
    "torso_0": 579.0, "torso_1": 572.0, "torso_2": 365.0, "torso_3": 78.2, "torso_4": 85.1, "torso_5": 58.3,
    "head_0": 6.22, "head_1": 6.23,
}
for side in ("right", "left"):
    for i, d in enumerate((24.8, 26.0, 11.1, 11.3, 3.95, 4.17, 3.93)):
        DAMPING_ZETA07[f"{side}_arm_{i}"] = d

GROUPS = {
    "wheels": [".*_wheel"],
    "torso": ["torso_[0-5]"],
    "arms": [".*_arm_[0-6]"],
    "head": ["head_[0-1]"],
}


def _sel(table: dict, exprs: list[str]) -> dict:
    """Keep the regex-keyed entries of a table that match at least one joint of the group."""
    import re

    names = _names_for(exprs)
    return {k: v for k, v in table.items() if any(re.fullmatch(k, n) for n in names)}


def _names_for(exprs):
    import re

    from common import CPP_JOINT_NAMES

    return [n for n in CPP_JOINT_NAMES if any(re.fullmatch(e, n) for e in exprs)]


def _prof(exprs, idx):
    return {k: v[idx] for k, v in _sel(_PROFILE, exprs).items()}


def make_rby1_cfg(
    mode: str = "implicit",
    usd_path: str | None = None,
    effort: str = "urdf",
    gain_scale: dict[str, tuple[float, float]] | None = None,
    damping_override: dict[str, float] | None = None,
    friction: str = "full",
    self_collision: bool = False,
    preset: str = "original",
) -> ArticulationCfg:
    """Build the test ArticulationCfg.

    gain_scale: {group: (kp_scale, kd_scale)} for tuning experiments.
    damping_override: {joint_name: damping} (e.g. damping-ratio tuned values), implicit/ideal_pd alike.
    friction: "full" = original motor model (static + dynamic + viscous), "viscous" = viscous only,
              "none" = no joint friction.
    preset: "original" = 원본 재현 게인 (default). "tuned" = viscous friction only + DAMPING_ZETA07 +
            l2/r2 driven like l1/r1 (the caller mirrors the targets: l2 = -l1) — add the gravity
            feedforward in the control loop and apply_original_materials(caster_mu=0.0).
    """
    if preset == "tuned":
        friction = "viscous"
        damping_override = {**DAMPING_ZETA07, **(damping_override or {})}
    effort_tbl = EFFORT_URDF if effort == "urdf" else EFFORT_USD
    gain_scale = gain_scale or {}
    actuators = {}
    for g, exprs in GROUPS.items():
        names = _names_for(exprs)
        ks, ds = gain_scale.get(g, (1.0, 1.0))
        stiff = {n: KP[n] * ks for n in names}
        damp = {n: DAMPING[n] * ds for n in names}
        if damping_override:
            damp.update({n: v for n, v in damping_override.items() if n in names})
        common = dict(
            joint_names_expr=exprs,
            armature=_prof(exprs, 3),
            friction=_prof(exprs, 0) if friction == "full" else 0.0,
            dynamic_friction=_prof(exprs, 1) if friction == "full" else 0.0,
            viscous_friction=_prof(exprs, 2) if friction in ("full", "viscous") else 0.0,
            joint_velocity_limit=_sel(VEL_URDF, exprs),
        )
        if mode == "implicit":
            if g == "wheels":
                stiff = {n: 0.0 for n in names}
                damp = {n: WHEEL_DAMPING_IMPLICIT * ds for n in names}
            actuators[g] = ImplicitActuatorCfg(
                **common, stiffness=stiff, damping=damp, joint_effort_limit=_sel(effort_tbl, exprs)
            )
        elif mode == "ideal_pd":
            lim = _sel(effort_tbl, exprs)  # pass effort="usd" to clip like the original
            actuators[g] = IdealPDActuatorCfg(
                **common, stiffness=stiff, damping=damp, actuator_effort_limit=lim, joint_effort_limit=lim
            )
        else:
            raise ValueError(mode)
    # Gripper: one driven finger per hand (l1/r1, 0 = closed, -0.05 = open) as in the original
    # rb_gripper server; l2/r2 keep the weak USD drive (K 0.309, D 1.2e-4) i.e. they are nearly passive.
    actuators["gripper"] = ImplicitActuatorCfg(
        joint_names_expr=["gripper_finger_[lr]1"], stiffness=100.0, damping=1.0, joint_effort_limit=75.0,
        joint_velocity_limit=0.05,
    )
    if preset == "tuned":  # software mimic: drive the second finger like the first (target = -l1)
        actuators["gripper_mimic"] = ImplicitActuatorCfg(
            joint_names_expr=["gripper_finger_[lr]2"], stiffness=100.0, damping=1.0, joint_effort_limit=75.0,
            joint_velocity_limit=0.05,
        )
    else:
        actuators["gripper_passive"] = ImplicitActuatorCfg(
            joint_names_expr=["gripper_finger_[lr]2"], stiffness=None, damping=None, joint_velocity_limit=0.05,
        )
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd_path or ROBOT_USD),
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, retain_accelerations=False, linear_damping=0.0, angular_damping=0.0,
                max_linear_velocity=200.0, max_angular_velocity=1000.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=self_collision,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), joint_pos={".*": 0.0}),
        soft_joint_pos_limit_factor=1.0,
        actuators=actuators,
    )


def apply_original_materials(robot_prim_expr: str = "/World/envs/env_.*/Robot", caster_mu: float = 0.3):
    """Bind the original physics materials: wheels mu 1.2, base caster cylinders mu 0.3.

    The casters are fixed cylinders (no swivel/roll joints), so they slide; caster_mu sets their friction."""
    import isaaclab.sim as su
    from isaaclab.sim.utils import find_matching_prim_paths

    wheel = su.RigidBodyMaterialCfg(static_friction=1.2, dynamic_friction=1.2, restitution=0.0)
    caster = su.RigidBodyMaterialCfg(static_friction=caster_mu, dynamic_friction=caster_mu, restitution=0.0,
                                     friction_combine_mode="min" if caster_mu == 0.0 else "average")
    wheel.func("/World/Physics_Materials/wheel_material", wheel)
    caster.func("/World/Physics_Materials/caster_material", caster)
    n = 0
    for p in find_matching_prim_paths(f"{robot_prim_expr}/wheel_.*/collisions/mesh_0/cylinder"):
        su.bind_physics_material(p, "/World/Physics_Materials/wheel_material")
        n += 1
    for p in find_matching_prim_paths(f"{robot_prim_expr}/base/collisions/mesh_.*/cylinder"):
        su.bind_physics_material(p, "/World/Physics_Materials/caster_material")
        n += 1
    return n


def gain_table() -> list[dict]:
    """Rows for the report: original kp/kd and the converted implicit gains."""
    from common import CPP_JOINT_NAMES

    rows = []
    for n in CPP_JOINT_NAMES:
        rows.append({"joint": n, "kp": KP[n], "kd_raw": KD[n], "damping": DAMPING[n]})
    return rows
