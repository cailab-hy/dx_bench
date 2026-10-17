"""Raw episode recorder (dx_arena side, no lerobot): JPEG frames + trajectory.npz + initial_state.json + episode.json.

Layout (follows lightwheel scripts/keyboard_recorder.py so its exporters can be reused):
  <root>/pending-<uuid>/  -> renamed to <root>/episode-<uuid>/ when saved, removed when discarded
    initial_state.json   scene JSON path + id, robot joint positions / base pose, object poses, cloth state files
    head/%06d.jpg, left_wrist/%06d.jpg, right_wrist/%06d.jpg   (480 x 640 RGB JPEG q95, before the action of that
                         step; PNG at 3 cameras x 1500 frames was ~4 GB per episode - over the data budget)
    trajectory.npz       per-step arrays (T, ...): see FIELDS
    episode.json         metadata: task, language (high + subtask list), fps, success, score, frames, devices
Only successful episodes are kept when `finish(save_only_success=True)`.
"""
from __future__ import annotations

import json
import shutil
import threading
import time
import uuid
from pathlib import Path
from queue import Queue

import numpy as np

FPS = 15
FIELDS = {
    "observation.state16": "rby1 16-D state: right arm 7, left arm 7, gripper open [r, l] (sim: half gap / 0.05)",
    "observation.torso": "torso_0..5 joint positions (rad)",
    "observation.joint_pos": "all 28 joint positions, Isaac Lab order (joint_names in episode.json)",
    "observation.joint_vel": "all 28 joint velocities",
    "observation.base_vel": "[forward m/s, yaw rate rad/s]",
    "observation.base_pose": "[x, y, yaw] env frame",
    "observation.tcp_right": "[x, y, z, qx, qy, qz, qw] base frame", "observation.tcp_left": "same, left",
    "action": "the embodiment action sent to env.step (rby1_ik 17-D or rby1_joint 24-D)",
    "action.joint_target": "resulting joint position targets of the 28 joints (absolute, rad / m)",
    "action.rby1_16": "absolute 16-D action in rby1 order: arm targets 14 + gripper command [r, l] (1 = open)",
    "action.torso": "torso joint targets (6)", "action.base": "base (v, omega) command",
    "stage_id": "index of the current subtask label (-1 unknown)", "intervention": "1 when a human intervened",
    "sim_time": "simulation time (s)",
}


def _np(x):
    try:
        return x.detach().cpu().numpy()
    except AttributeError:
        return np.asarray(x)


class _ImageWriter:
    def __init__(self, n_threads: int = 3):
        self.q: Queue = Queue(maxsize=256)
        self.threads = [threading.Thread(target=self._run, daemon=True) for _ in range(n_threads)]
        for t in self.threads:
            t.start()

    def _run(self):
        from PIL import Image

        while True:
            item = self.q.get()
            if item is None:
                self.q.task_done()
                break
            path, arr = item
            Image.fromarray(arr).save(path, quality=95)  # JPEG: ~0.12 MB vs ~0.9 MB PNG per 640 x 480 frame
            self.q.task_done()

    def put(self, path, arr):
        self.q.put((path, arr))

    def close(self):
        for _ in self.threads:
            self.q.put(None)
        self.q.join()


class EpisodeRecorder:
    def __init__(self, root, metadata: dict, initial_state: dict, cameras=("head", "left_wrist", "right_wrist")):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.id = uuid.uuid4().hex[:12]
        self.dir = self.root / f"pending-{self.id}"
        self.dir.mkdir()
        self.meta = dict(metadata)
        self.cameras = cameras
        for c in cameras:
            (self.dir / c).mkdir()
        (self.dir / "initial_state.json").write_text(json.dumps(initial_state, indent=1, default=float))
        self.buf: dict[str, list] = {}
        self.t = 0
        self.writer = _ImageWriter()
        self.started = time.time()
        self._env = None
        self.subtasks = list(self.meta.get("language", {}).get("subtasks", []))
        self.intervention = 0

    # ------------------------------------------------------------ construction from a running env
    @classmethod
    def from_env(cls, U, root, policy_name: str = "", scene: str | None = None, device_name: str = "scripted"):
        from dxbench.tasks import bed_tasks

        task = bed_tasks.CURRENT_TASK
        spec = task.spec if task is not None else None
        r = U.scene["robot"]
        init = {"scene_json": scene, "scene_id": spec.scene_id if spec else None,
                "joint_names": list(r.joint_names),
                "joint_positions": r.data.joint_pos.torch[0].tolist(),
                "base_pose": {"pos": (r.data.root_link_pos_w.torch[0] - U.scene.env_origins[0]).tolist(),
                              "quat_xyzw": r.data.root_link_quat_w.torch[0].tolist()},
                "objects": {}, "cloth_states": {}}
        if spec:
            for o in spec.objects:
                init["objects"][o.name] = {"kind": o.kind, "variant": o.variant, "xyz": o.xyz, "yaw": o.yaw}
                if o.cloth_state:
                    init["cloth_states"][o.name] = o.cloth_state
        meta = {"task": spec.task if spec else None, "difficulty": spec.difficulty if spec else None,
                "scene_id": spec.scene_id if spec else None, "language": spec.language if spec else {},
                "fps": FPS, "physics_dt": U.physics_dt, "policy": policy_name, "teleop_device": device_name,
                "embodiment_action_dim": int(U.action_manager.total_action_dim), "joint_names": list(r.joint_names),
                "feature_descriptions": FIELDS, "robot_type": "rby1", "sim": "isaacsim-6.1/isaaclab-3.0/arena-0.3.1"}
        rec = cls(root, meta, init)
        rec._env = U
        return rec

    # ------------------------------------------------------------ per step
    def _add(self, k, v):
        self.buf.setdefault(k, []).append(np.asarray(v, dtype=np.float32))

    def pre_step(self, obs, action, subtask: str | None = None, intervention: int | None = None):
        U = self._env
        pol = obs["policy"]
        self._add("observation.state16", _np(pol["state16"][0]))
        self._add("observation.torso", _np(pol["torso"][0]))
        self._add("observation.base_vel", _np(pol["base_vel"][0]))
        self._add("observation.base_pose", _np(pol["base_pose"][0]))
        self._add("observation.tcp_right", _np(pol["tcp_right"][0]))
        self._add("observation.tcp_left", _np(pol["tcp_left"][0]))
        r = U.scene["robot"]
        self._add("observation.joint_pos", _np(r.data.joint_pos.torch[0]))
        self._add("observation.joint_vel", _np(r.data.joint_vel.torch[0]))
        self._add("action", _np(action[0]))
        self.buf.setdefault("sim_time", []).append(float(self.t) / FPS)
        stage = self.subtasks.index(subtask) if subtask in self.subtasks else -1
        self.buf.setdefault("stage_id", []).append(stage)
        self.buf.setdefault("subtask", []).append(subtask or "")
        self.buf.setdefault("intervention", []).append(int(self.intervention if intervention is None else intervention))
        cam = obs.get("camera_obs", {})
        for c in self.cameras:
            key = f"{c}_rgb"
            if key in cam:
                self.writer.put(self.dir / c / f"{self.t:06d}.jpg", _np(cam[key][0])[..., :3].astype(np.uint8))
        self.t += 1

    def post_step(self):
        """Call after env.step: the joint targets written for this step's action."""
        U = self._env
        r = U.scene["robot"]
        tgt = _np(r.data.joint_pos_target.torch[0])
        self._add("action.joint_target", tgt)
        names = list(r.joint_names)
        arm = [names.index(f"right_arm_{i}") for i in range(7)] + [names.index(f"left_arm_{i}") for i in range(7)]
        try:
            g = _np(U.action_manager.get_term("gripper").raw_actions[0])[:2]
        except Exception:  # noqa: BLE001
            g = np.ones(2)
        self._add("action.rby1_16", np.r_[tgt[arm], np.clip(g, 0, 1)])
        self._add("action.torso", tgt[[names.index(f"torso_{i}") for i in range(6)]])
        try:
            self._add("action.base", _np(U.action_manager.get_term("base").raw_actions[0])[:2])
        except Exception:  # noqa: BLE001
            self._add("action.base", np.zeros(2))

    # ------------------------------------------------------------ end
    def finish(self, success: bool | None, score: float | None, save_only_success: bool = True,
               extra: dict | None = None) -> Path | None:
        self.writer.close()
        keep = bool(success) or not save_only_success
        if not keep:
            shutil.rmtree(self.dir, ignore_errors=True)
            return None
        n = min(len(v) for v in self.buf.values()) if self.buf else 0
        arrays = {}
        for k, v in self.buf.items():
            if k == "subtask":
                arrays[k] = np.array(v[:n])
            else:
                arrays[k] = np.stack([np.asarray(x) for x in v[:n]])
        np.savez_compressed(self.dir / "trajectory.npz", **arrays)
        meta = dict(self.meta)
        meta.update({"episode_id": self.id, "frames": n, "status": "saved", "success": bool(success), "score": score,
                     "wall_s": time.time() - self.started, **(extra or {})})
        (self.dir / "episode.json").write_text(json.dumps(meta, indent=1, default=float))
        final = self.root / f"episode-{self.id}"
        self.dir.rename(final)
        return final

    def discard(self):
        self.writer.close()
        shutil.rmtree(self.dir, ignore_errors=True)
