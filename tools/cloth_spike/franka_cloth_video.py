# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
dx_bench copy of IsaacLab `scripts/environments/state_machine/lift_franka_soft.py` (Isaac Lab 3.0, unchanged state
machine) that also records two RTX cameras (overview + close-up on the grasp) into one side-by-side mp4.

    A=~/miniforge3/envs/dx_arena/bin/python
    $A tools/cloth_spike/franka_cloth_video.py --task Isaac-Lift-Cloth-Franka --out outputs/franka_cloth/physx.mp4 presets=physx
    $A tools/cloth_spike/franka_cloth_video.py --task Isaac-Lift-Cloth-Franka --out outputs/franka_cloth/newton.mp4

Original docstring:

Script to demonstrate lifting a deformable object with a robotic arm.

The state machine is implemented in the kernel function `infer_state_machine`.
It uses the `warp` library to run the state machine in parallel on the GPU.

.. code-block:: bash

    # Kitless run with the Newton OpenGL viewer (default).
    uv run python scripts/environments/state_machine/lift_franka_soft.py

    # Headless.
    uv run python scripts/environments/state_machine/lift_franka_soft.py --viz none

"""

import argparse
import sys
from collections.abc import Sequence

import gymnasium as gym
import torch
import warp as wp

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.assets.deformable_object.deformable_object_data import DeformableObjectData
from isaaclab.visualizers import VisualizerCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

# add argparse arguments
parser = argparse.ArgumentParser(description="Pick and lift a deformable with a robotic arm.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--num_steps", type=int, default=1000, help="Number of environment steps to run.")
parser.add_argument("--task", type=str, default="Isaac-Lift-Cloth-Franka", help="The task to run.")
parser.add_argument("--out", type=str, required=True, help="Output mp4 path.")
parser.add_argument("--episodes", type=int, default=2, help="Episodes to record.")
parser.add_argument("--res", type=int, nargs=2, default=(640, 480), help="Camera width height.")
parser.add_argument("--cam_renderer", choices=("rtx", "newton"), default="rtx",
                    help="rtx for PhysX runs; newton (warp) for Newton runs, where RTX frames came out black.")
parser.add_argument("--episode_s", type=float, default=None, help="Episode length (s); default 10, or 15 with --flat.")
parser.add_argument("--assist", choices=("none", "pad"), default="none",
                    help="pad: dxbench pad-box assisted grasp (dxbench/embodiments/mdp/pad_grasp.py) + finger/cloth filter.")
parser.add_argument("--tip_z", type=float, default=0.1119,
                    help="Franka fingertip along panda_hand z (m), collision geometry; measure with --probe_fingers.")
parser.add_argument("--physics_hz", type=float, default=None,
                    help="Physics rate (env step stays 30 Hz); --flat defaults to 480 Hz like the dx_bench tasks.")
parser.add_argument("--probe_fingers", action="store_true", help="Print the finger bounding boxes in the hand frame.")
parser.add_argument("--squeeze", type=float, default=0.0, help="pad assist: closing-direction squeeze fraction.")
parser.add_argument("--snap_s", type=float, default=0.2, help="pad assist: pull-in duration (s).")
parser.add_argument("--flat", action="store_true",
                    help="dx_bench variant: no supports, the cloth lies flat on the table and is pinched there.")
parser.add_argument("--grasp", choices=("corner", "center"), default="corner",
                    help="--flat pinch point: 2 cm inside the corner nearest the robot, or the cloth centre.")
parser.add_argument("--grasp_dz", type=float, default=0.005,
                    help="--flat: TCP height above the grasped node (TCP = panda_hand + 10.34 cm, about 1 cm above the fingertips).")
add_launcher_args(parser)
parser.set_defaults(visualizer=[])
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

# initialize warp
wp.init()


class GripperState:
    """States for the gripper."""

    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)


class PickSmState:
    """States for the pick state machine."""

    REST = wp.constant(0)
    APPROACH_ABOVE_OBJECT = wp.constant(1)
    APPROACH_OBJECT = wp.constant(2)
    GRASP_OBJECT = wp.constant(3)
    LIFT_OBJECT = wp.constant(4)
    OPEN_GRIPPER = wp.constant(5)


@wp.func
def distance_below_threshold(current_pos: wp.vec3, desired_pos: wp.vec3, threshold: float) -> bool:
    return wp.length(current_pos - desired_pos) < threshold


@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    ee_pose: wp.array(dtype=wp.transform),
    object_pose: wp.array(dtype=wp.transform),
    des_object_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),
    offset: wp.array(dtype=wp.transform),
    position_threshold: float,
):
    # retrieve thread id
    tid = wp.tid()
    # retrieve state machine state
    state = sm_state[tid]
    # decide next state
    if state == PickSmState.REST:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.REST:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.APPROACH_ABOVE_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_ABOVE_OBJECT:
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], object_pose[tid])
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            # wait for a while
            if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
                # move to next state and reset wait time
                sm_state[tid] = PickSmState.APPROACH_OBJECT
                sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            # wait for a while
            if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
                # move to next state and reset wait time
                sm_state[tid] = PickSmState.GRASP_OBJECT
                sm_wait_time[tid] = 0.0
    elif state == PickSmState.GRASP_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.GRASP_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.LIFT_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.LIFT_OBJECT:
        des_ee_pose[tid] = des_object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            # wait for a while
            if sm_wait_time[tid] >= PickSmWaitTime.LIFT_OBJECT:
                # move to next state and reset wait time
                sm_state[tid] = PickSmState.OPEN_GRIPPER
                sm_wait_time[tid] = 0.0
    elif state == PickSmState.OPEN_GRIPPER:
        # des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.OPEN_GRIPPER:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.OPEN_GRIPPER
            sm_wait_time[tid] = 0.0
    # increment wait time
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]


class PickSmWaitTime:
    """Additional wait times (in s) for states for before switching.

    Wait times are generous because the low-PD Franka takes a while to settle on each IK target.
    """

    REST = wp.constant(0.2)
    APPROACH_ABOVE_OBJECT = wp.constant(1.0)
    APPROACH_OBJECT = wp.constant(1.5)
    GRASP_OBJECT = wp.constant(1.5)
    LIFT_OBJECT = wp.constant(1.5)
    OPEN_GRIPPER = wp.constant(0.0)


class PickAndLiftSm:
    """A simple state machine in a robot's task space to pick and lift an object.

    The state machine is implemented as a warp kernel. It takes in the current state of
    the robot's end-effector and the object, and outputs the desired state of the robot's
    end-effector and the gripper. The state machine is implemented as a finite state
    machine with the following states:

    1. REST: The robot is at rest.
    2. APPROACH_ABOVE_OBJECT: The robot moves above the object.
    3. APPROACH_OBJECT: The robot moves to the object.
    4. GRASP_OBJECT: The robot grasps the object.
    5. LIFT_OBJECT: The robot lifts the object to the desired pose. This is the final state.
    """

    def __init__(self, dt: float, num_envs: int, device: torch.device | str = "cpu", position_threshold=0.03):
        """Initialize the state machine.

        Args:
            dt: The environment time step.
            num_envs: The number of environments to simulate.
            device: The device to run the state machine on.
        """
        # save parameters
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        self.position_threshold = position_threshold
        # initialize state machine
        self.sm_dt = torch.full((self.num_envs,), self.dt, device=self.device)
        self.sm_state = torch.full((self.num_envs,), 0, dtype=torch.int32, device=self.device)
        self.sm_wait_time = torch.zeros((self.num_envs,), device=self.device)

        # desired state
        self.des_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.des_gripper_state = torch.full((self.num_envs,), 0.0, device=self.device)

        # approach above object offset
        self.offset = torch.zeros((self.num_envs, 7), device=self.device)
        self.offset[:, 2] = 0.1
        self.offset[:, -1] = 1.0  # warp expects quaternion as (x, y, z, w)

        # convert to warp
        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)
        self.offset_wp = wp.from_torch(self.offset, wp.transform)

    def reset_idx(self, env_ids: Sequence[int] = None):
        """Reset the state machine."""
        if env_ids is None:
            env_ids = slice(None)
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0

    def compute(self, ee_pose: torch.Tensor, object_pose: torch.Tensor, des_object_pose: torch.Tensor):
        """Compute the desired state of the robot's end-effector and the gripper."""

        # convert to warp
        ee_pose_wp = wp.from_torch(ee_pose.contiguous(), wp.transform)
        object_pose_wp = wp.from_torch(object_pose.contiguous(), wp.transform)
        des_object_pose_wp = wp.from_torch(des_object_pose.contiguous(), wp.transform)

        # run state machine
        wp.launch(
            kernel=infer_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                ee_pose_wp,
                object_pose_wp,
                des_object_pose_wp,
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
                self.offset_wp,
                self.position_threshold,
            ],
            device=self.device,
        )

        # convert to torch
        return torch.cat([self.des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)


def main():
    # parse configuration via Hydra, so presets can be selected on the CLI (e.g. presets=isaacsim_physx)
    env_cfg, _ = resolve_task_config(args_cli.task, "")
    env_cfg.sim.device = args_cli.device
    env_cfg.scene.num_envs = args_cli.num_envs
    # Scripted demo: keep only the time-out, extended to 10 s so the slow low-PD Franka can finish a
    # pick-and-lift, and drop the failure terminations so a transient bound or velocity spike does
    # not cut a run short.
    env_cfg.episode_length_s = 10.0
    for term_name in list(vars(env_cfg.terminations)):
        if term_name != "time_out":
            setattr(env_cfg.terminations, term_name, None)
    # the state machine emits absolute end-effector poses, so pick the task's own IK action preset
    # (e.g. the cloth closes the gripper fully); the env otherwise defaults to relative joint
    # targets, which RL trains on.
    env_cfg.actions = type(env_cfg)().actions.ik
    if args_cli.flat:
        env_cfg.episode_length_s = 15.0  # the low-PD Franka needs the time to reach the table before closing
    # the task ramps gravity from -1e-4 to -9.81 m/s^2 over 10k RL steps (curriculum); a scripted run never gets
    # there, so the cloth was nearly weightless. Use full gravity from the start.
    if getattr(env_cfg, "curriculum", None) is not None and hasattr(env_cfg.curriculum, "gravity"):
        env_cfg.curriculum.gravity = None
    if hasattr(env_cfg.events, "variable_gravity"):
        env_cfg.events.variable_gravity = None
    env_cfg.sim.gravity = (0.0, 0.0, -9.81)
    hz = args_cli.physics_hz or (480.0 if args_cli.flat else None)
    if hz:
        step_dt_ = env_cfg.sim.dt * env_cfg.decimation
        env_cfg.decimation = max(1, round(step_dt_ * hz))
        env_cfg.sim.dt = step_dt_ / env_cfg.decimation
        env_cfg.sim.render_interval = env_cfg.decimation
    if args_cli.episode_s:
        env_cfg.episode_length_s = args_cli.episode_s
    if args_cli.flat:
        # cloth flat on the table top (z = 0) instead of standing in the gap; supports parked out of reach
        env_cfg.scene.deformable.init_state.pos = (0.4, 0.0, 0.003)
        env_cfg.scene.deformable.init_state.rot = (0.0, 0.0, 0.0, 1.0)
        env_cfg.scene.support_neg_y.init_state.pos = (1.5, -0.02, 0.075)
        env_cfg.scene.support_pos_y.init_state.pos = (1.5, 0.02, 0.075)
        # fixed placement (corner at x 0.37 m, reachable top-down) so runs differ only in what is varied
        env_cfg.events.reset_deformable.params["position_range"] = {"x": (0.05, 0.05), "y": (0.0, 0.0), "z": (0.0, 0.0)}
        props = getattr(env_cfg.scene.deformable.spawn, "deformable_props", None)
        if props is not None and hasattr(props, "sleep_threshold"):
            # a cloth held still by the velocity pins fell asleep and stayed in the air after release
            props.sleep_threshold = 0.0
            props.settling_threshold = 0.0
    if args_cli.assist == "pad":
        from isaaclab.managers import EventTermCfg

        from dxbench.embodiments.mdp.pad_grasp import HandSpec, PadGraspActionCfg, filter_hand_cloth_collisions

        env_cfg.actions.pad_grasp = PadGraspActionCfg(
            cloth_names=["deformable"], gripper_term="gripper_action",
            hands=[HandSpec(body="panda_hand", tip_offset=(0.0, 0.0, args_cli.tip_z),
                            axes=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                            gap_joints=("panda_finger_joint1", "panda_finger_joint2"), close_below=0.0,
                            pad_w=0.033, pad_h=0.0165)],  # finger collision box (--probe_fingers)
            squeeze=args_cli.squeeze, snap_s=args_cli.snap_s)
        env_cfg.events.filter_hand_cloth = EventTermCfg(
            func=filter_hand_cloth_collisions, mode="prestartup",
            params={"cloth_names": ["Deformable"], "links": ("panda_leftfinger", "panda_rightfinger", "panda_hand")})
    env_cfg.viewer.eye = (1.3, 0.6, 0.5)
    env_cfg.viewer.lookat = (0.5, 0.0, 0.05)
    env_cfg.sim.default_visualizer_cfg = VisualizerCfg(eye=env_cfg.viewer.eye, lookat=env_cfg.viewer.lookat)

    # dx_bench: two RTX cameras; poses are set from eye/target after every reset
    import os

    import imageio.v2 as imageio
    import numpy as np

    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg
    from isaaclab_newton.renderers import NewtonWarpRendererCfg
    from isaaclab_physx.renderers import IsaacRtxRendererCfg

    w, h = args_cli.res
    for name in ("cam_wide", "cam_close"):
        setattr(env_cfg.scene, name, CameraCfg(
            prim_path="{ENV_REGEX_NS}/" + name, update_period=0.0, height=h, width=w, data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=24.0 if name == "cam_wide" else 30.0,
                                             clipping_range=(0.01, 5.0)),
            renderer_cfg=IsaacRtxRendererCfg() if args_cli.cam_renderer == "rtx" else NewtonWarpRendererCfg()))

    grasp = {"node": None}

    def aim_cameras(env):
        u = env.unwrapped
        if args_cli.flat:
            nodes = u.scene["deformable"].data.nodal_pos_w.torch[0]
            ref = nodes[:, :2].mean(0) if args_cli.grasp == "center" else nodes[:, :2].min(0).values + 0.02
            grasp["node"] = int((nodes[:, :2] - ref).norm(dim=-1).argmin())
            c = nodes[grasp["node"]][None].clone()
            c[:, 2] += 0.12  # frame the pinch and the lift
            wide_eye = c + torch.tensor([[0.75, 0.65, 0.20]], device=c.device)
            close_eye = c + torch.tensor([[0.20, 0.0, -0.07]], device=c.device)  # re-aimed at the hand every step
            print(f"[franka_cloth_video] grasp node {grasp['node']} at {c[0].tolist()}", flush=True)
        else:
            c = u.scene["deformable"].data.root_pos_w.torch[:1].clone()
            c[:, 2] += 0.07  # the fold over the two supports, where the fingers close (state machine grasp_z)
            wide_eye = c + torch.tensor([[0.55, 0.40, 0.22]], device=c.device)
            close_eye = c + torch.tensor([[0.24, 0.0, 0.06]], device=c.device)  # along the fold ridge: fingers + V
        u.scene["cam_wide"].set_world_poses_from_view(wide_eye, c)
        u.scene["cam_close"].set_world_poses_from_view(close_eye, c)

    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    step_dt = env_cfg.sim.dt * env_cfg.decimation
    writer = imageio.get_writer(args_cli.out, fps=round(1.0 / step_dt), codec="libx264", quality=8,
                                macro_block_size=1)
    episodes = 0

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(args_cli.task, cfg=env_cfg)
        is_cable = "cable" in env.unwrapped.scene.keys()

        # reset environment at start
        env.reset()
        aim_cameras(env)
        if args_cli.probe_fingers:
            from pxr import Gf, Usd, UsdGeom

            import isaaclab.sim as sim_utils_

            stage = sim_utils_.get_current_stage()
            prims = {q.GetName(): q for q in Usd.PrimRange(stage.GetPrimAtPath("/World/envs/env_0/Robot"),
                                                             Usd.TraverseInstanceProxies())
                     if q.GetName() in ("panda_hand", "panda_leftfinger", "panda_rightfinger")}
            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"])
            hand_inv = UsdGeom.Xformable(prims["panda_hand"]).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()).GetInverse()
            from pxr import UsdPhysics

            for k in ("panda_leftfinger", "panda_rightfinger"):
                for kind in ("all", "collision"):
                    subs = [prims[k]] if kind == "all" else [
                        q for q in Usd.PrimRange(prims[k], Usd.TraverseInstanceProxies()) if q.HasAPI(UsdPhysics.CollisionAPI)]
                    pts = []
                    for sp in subs:
                        r = cache.ComputeWorldBound(sp).ComputeAlignedRange()
                        if r.IsEmpty():
                            continue
                        pts += [hand_inv.Transform(Gf.Vec3d(x, y, z)) for x in (r.GetMin()[0], r.GetMax()[0])
                                for y in (r.GetMin()[1], r.GetMax()[1]) for z in (r.GetMin()[2], r.GetMax()[2])]
                    if not pts:
                        print(f"[franka_cloth_video] {k} {kind}: no geometry", flush=True)
                        continue
                    lo = [min(p_[i] for p_ in pts) for i in range(3)]
                    hi = [max(p_[i] for p_ in pts) for i in range(3)]
                    print(f"[franka_cloth_video] {k} {kind} ({len(subs)} prims) bbox in hand frame "
                          f"lo={[round(v, 4) for v in lo]} hi={[round(v, 4) for v in hi]}", flush=True)
            q = env.unwrapped.scene["robot"].data.joint_pos.torch[0, -2:].tolist()
            print(f"[franka_cloth_video] finger q={q}", flush=True)

        # create action buffers (position + quaternion)
        actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
        actions[:, 3] = 1.0
        # desired rotation after grasping
        desired_orientation = torch.zeros((env.unwrapped.num_envs, 4), device=env.unwrapped.device)
        desired_orientation[:, 0] = 1.0

        # Top-down approach: identity quaternion (wxyz, w=1) aligns panda_hand with the Franka root,
        # giving the canonical top-down grasp pose. The bar lies along world-X, so the gripper
        # closes across its short side without any wrist twist.
        object_grasp_orientation = torch.zeros((env.unwrapped.num_envs, 4), device=env.unwrapped.device)
        object_grasp_orientation[:, 0] = 1.0
        # Grasp 1 cm below the deformable's centre of mass, so the fingers close around its lower half.
        # The cloth drapes over a support cube, so its COM sits well below the graspable fold; reach
        # 8 cm higher to close on the raised cloth instead of the table.
        grasp_z = -0.01 + (0.08 if "Cloth" in args_cli.task else 0.0)
        object_local_grasp_position = torch.tensor([0.0, 0.0, grasp_z], device=env.unwrapped.device)

        # create state machine
        # --flat: close only once the TCP is within 8 mm of the pinch point (the official 3 cm closed the hand
        # 1.6 cm above the cloth while the slow arm was still descending)
        pick_sm = PickAndLiftSm(env_cfg.sim.dt * env_cfg.decimation, env.unwrapped.num_envs, env.unwrapped.device,
                                position_threshold=0.008 if args_cli.flat else 0.03)

        for _ in range(args_cli.num_steps):
            # run everything in inference mode
            with torch.inference_mode():
                # step environment
                _, _, terminated, time_outs, _ = env.step(actions)
                dones = terminated | time_outs

                if args_cli.flat:  # close-up follows the fingertips, looking along +x (both fingers in view)
                    tip = env.unwrapped.scene["ee_frame"].data.target_pos_w.torch[:1].clone()
                    tip[:, 2] -= 0.01
                    env.unwrapped.scene["cam_close"].set_world_poses_from_view(
                        tip + torch.tensor([[0.18, 0.0, 0.04]], device=tip.device), tip)
                frames = [env.unwrapped.scene[n].data.output["rgb"].torch[0, ..., :3] for n in ("cam_wide", "cam_close")]
                writer.append_data(np.concatenate([f.cpu().numpy() for f in frames], axis=1).astype(np.uint8))
                n_step = env.unwrapped.episode_length_buf[0].item()
                if n_step == 1:  # nodal positions read right after a reset can be stale: pick / aim on step 1
                    aim_cameras(env)
                if n_step % 15 == 0:
                    nz = env.unwrapped.scene["deformable"].data.nodal_pos_w.torch[0, :, 2]
                    ee_z = env.unwrapped.scene["ee_frame"].data.target_pos_w.torch[0, 0, 2].item()
                    q = env.unwrapped.scene["robot"].data.joint_pos.torch[0, -2:].tolist()
                    print(f"[franka_cloth_video] t={n_step * step_dt:4.1f}s sm={int(pick_sm.sm_state[0])} "
                          f"ee_z={ee_z:.3f} cloth_z min/mean/max={nz.min():.3f}/{nz.mean():.3f}/{nz.max():.3f} "
                          f"fingers={q[0]:.4f},{q[1]:.4f}", flush=True)
                    if args_cli.assist == "pad":
                        term = env.unwrapped.action_manager.get_term("pad_grasp")
                        off = term.pinned_offsets(0, 0, "deformable")
                        cap = term.last_capture.get((0, 0))
                        txt = "none" if off is None else (
                            f"n={len(off)} |w|max={off[:, 0].abs().max():.4f} |c|max={off[:, 1].abs().max():.4f} "
                            f"a=[{off[:, 2].min():.4f},{off[:, 2].max():.4f}]")
                        print(f"[franka_cloth_video]   pad pins {txt} last_capture={cap}", flush=True)
                    if args_cli.flat:
                        gp = env.unwrapped.scene["deformable"].data.nodal_pos_w.torch[0, grasp["node"]]
                        tcp = env.unwrapped.scene["ee_frame"].data.target_pos_w.torch[0, 0]
                        print(f"[franka_cloth_video]   grasp node z={gp[2]:.3f} tcp-node={(tcp - gp).tolist()}", flush=True)

                # reset state machine
                if dones.any():
                    pick_sm.reset_idx(dones.nonzero(as_tuple=False).squeeze(-1))
                    episodes += 1
                    print(f"[franka_cloth_video] episode {episodes} done", flush=True)
                    if episodes >= args_cli.episodes:
                        break
                    aim_cameras(env)

                # observations
                # -- end-effector frame
                ee_frame_sensor = env.unwrapped.scene["ee_frame"]
                tcp_rest_position = (
                    ee_frame_sensor.data.target_pos_w.torch[..., 0, :].clone() - env.unwrapped.scene.env_origins
                )
                tcp_rest_orientation = ee_frame_sensor.data.target_quat_w.torch[..., 0, :].clone()
                # -- object frame
                if is_cable:
                    segment_index = env_cfg.commands.cable_pose.segment_index
                    object_position = (
                        env.unwrapped.scene["cable"].data.segment_pose_w.torch[:, segment_index, :3]
                        - env.unwrapped.scene.env_origins
                    )
                    command_name = "cable_pose"
                else:
                    object_data: DeformableObjectData = env.unwrapped.scene["deformable"].data
                    object_position = object_data.root_pos_w.torch - env.unwrapped.scene.env_origins
                    object_position += object_local_grasp_position
                    if args_cli.flat:  # follow the picked node, TCP grasp_dz above it
                        object_position = (object_data.nodal_pos_w.torch[:, grasp["node"]]
                                           - env.unwrapped.scene.env_origins)
                        object_position[:, 2] += args_cli.grasp_dz
                    command_name = "deformable_pose"

                # -- target object frame
                desired_position = env.unwrapped.command_manager.get_command(command_name)[..., :3]

                # advance state machine
                actions = pick_sm.compute(
                    torch.cat([tcp_rest_position, tcp_rest_orientation], dim=-1),
                    torch.cat([object_position, object_grasp_orientation], dim=-1),
                    torch.cat([desired_position, desired_orientation], dim=-1),
                )

        writer.close()
        print(f"[franka_cloth_video] wrote {args_cli.out}", flush=True)
        # close the environment
        env.close()


if __name__ == "__main__":
    main()
