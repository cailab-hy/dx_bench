"""Raw dx_arena episodes (dxbench/teleop/recorder.py) -> LeRobot v3 dataset, 15 fps.

    ~/miniforge3/envs/dx_policy/bin/python -m dxbench_policy.convert.raw_to_lerobot \
        --raw data/raw/bed03_oracle --out data/lerobot/dx_bed03_oracle_dx24 --repo_id dxbench/bed03_oracle --fmt dx24

Formats
  rby1  real RB-Y1 layout (rby1_format): state/action 16-D (right arm, left arm, gripper [r, l]), gripper state =
        half finger gap / 0.045 clipped to 0..1, gripper action binary; images front 480x640, right/left 640x480
        portrait (landscape sim frames centre-cropped + resized, left rotated 180 - rby1_format.adapt_image).
  dx24  full sim policy layout = `rby1_joint` action space: action 24 = arms 14 (abs rad) + gripper 2 (binary, 1 open)
        + torso 6 (abs rad) + base 2 (v m/s, omega rad/s); state 24 = rby1 state16 + torso 6 + base velocity 2;
        images front / right / left all 480x640 as rendered.
Both add non-policy columns `dx.*` (ignored by LeRobot policies: they do not start with observation./action):
  dx.stage_id (subtask index), dx.intervention, dx.base_pose [x, y, yaw], dx.tcp_right / dx.tcp_left (7, base frame xyzw).
Episodes are added in name order; `meta/dx_sources.json` lists the raw episode of every LeRobot episode (idempotent:
re-running skips raw episodes already exported).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from dxbench_policy import rby1_format as R

CAMS = {"head": "first_person", "right_wrist": "right_hand", "left_wrist": "left_hand"}
KEY = {"head": "front", "right_wrist": "right", "left_wrist": "left"}
TORSO = [f"torso_{i}" for i in range(6)]
DX24_ACTION = R.NAMES + TORSO + ["base_v", "base_omega"]
DX24_STATE = R.NAMES + TORSO + ["base_v", "base_omega"]


def load_raw(ep: Path):
    m = json.loads((ep / "episode.json").read_text())
    a = dict(np.load(ep / "trajectory.npz", allow_pickle=False))
    init = json.loads((ep / "initial_state.json").read_text())
    return m, a, init


def episode_arrays(m: dict, a: dict, fmt: str) -> dict[str, np.ndarray]:
    names = m["joint_names"]
    st16 = R.state16(a["observation.joint_pos"], names)
    arm_t = a["action.rby1_16"][:, :14]
    grip = (a["action.rby1_16"][:, 14:16] >= 0.5).astype(np.float32)
    act16 = np.concatenate([arm_t, grip], -1).astype(np.float32)
    out = {}
    if fmt == "rby1":
        out["observation.state"], out["action"] = st16, act16
    else:
        out["observation.state"] = np.concatenate([st16, a["observation.torso"], a["observation.base_vel"]], -1)
        out["action"] = np.concatenate([act16, a["action.torso"], a["action.base"]], -1)
    out["dx.stage_id"] = a["stage_id"].astype(np.int64)[:, None]
    out["dx.intervention"] = a["intervention"].astype(np.int64)[:, None]
    out["dx.base_pose"] = a["observation.base_pose"]
    out["dx.tcp_right"] = a["observation.tcp_right"]
    out["dx.tcp_left"] = a["observation.tcp_left"]
    return {k: np.asarray(v, dtype=np.float32 if v.dtype.kind == "f" else v.dtype) for k, v in out.items()}


IMAGE_SCALE = 1.0  # set by --image_scale (dx24 only): stored frames resized by this factor (training speed)


def features(fmt: str) -> dict:
    names = R.NAMES if fmt == "rby1" else DX24_STATE
    f = {"observation.state": {"dtype": "float32", "shape": (len(names),), "names": names},
         "action": {"dtype": "float32", "shape": (len(names),), "names": R.NAMES if fmt == "rby1" else DX24_ACTION},
         "dx.stage_id": {"dtype": "int64", "shape": (1,), "names": None},
         "dx.intervention": {"dtype": "int64", "shape": (1,), "names": None},
         "dx.base_pose": {"dtype": "float32", "shape": (3,), "names": ["x", "y", "yaw"]},
         "dx.tcp_right": {"dtype": "float32", "shape": (7,), "names": ["x", "y", "z", "qx", "qy", "qz", "qw"]},
         "dx.tcp_left": {"dtype": "float32", "shape": (7,), "names": ["x", "y", "z", "qx", "qy", "qz", "qw"]}}
    for cam in CAMS:
        shape = R.IMAGE_SHAPES[KEY[cam]] if fmt == "rby1" else (round(480 * IMAGE_SCALE), round(640 * IMAGE_SCALE))
        f[f"observation.images.{KEY[cam]}"] = {"dtype": "video", "shape": (*shape, 3),
                                               "names": ["height", "width", "channels"]}
    return f


def image(ep: Path, cam: str, t: int, fmt: str) -> np.ndarray:
    f = ep / cam / f"{t:06d}.jpg"
    if not f.exists():
        f = ep / cam / f"{t:06d}.png"  # recordings before 2026-10-17 18:00
    with Image.open(f) as im:
        img = np.asarray(im.convert("RGB"))
    if fmt == "rby1":
        return R.adapt_image(img, CAMS[cam], rendered_for_rby1=False)
    if IMAGE_SCALE != 1.0:
        h, w = round(480 * IMAGE_SCALE), round(640 * IMAGE_SCALE)
        img = np.asarray(Image.fromarray(img).resize((w, h), Image.BILINEAR))
    return img


def convert(raw_root, out, repo_id, fmt="dx24", only_success=True, limit=None, exclude=()) -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    raw_root, out = Path(raw_root), Path(out)
    eps = sorted(p for p in raw_root.glob("episode-*") if (p / "episode.json").exists())
    sources_f = out / "meta" / "dx_sources.json"
    if out.exists() and (out / "meta" / "info.json").exists():
        ds = LeRobotDataset.resume(repo_id, root=out) if hasattr(LeRobotDataset, "resume") else None
        if ds is None:
            raise RuntimeError("existing dataset and no resume API; use a new --out")
        sources = json.loads(sources_f.read_text()) if sources_f.exists() else []
    else:
        if out.exists():
            shutil.rmtree(out)
        ds = LeRobotDataset.create(repo_id, fps=R.FPS, features=features(fmt), root=out, robot_type=R.ROBOT_TYPE,
                                   use_videos=True, image_writer_threads=4)
        sources = []
    done = {s["raw"] for s in sources}
    n_new = 0
    for ep in eps:
        if limit is not None and n_new >= limit:
            break
        if ep.name in done or ep.name in set(exclude):
            continue
        m, a, init = load_raw(ep)
        if only_success and not m.get("success"):
            continue
        arr = episode_arrays(m, a, fmt)
        T = len(arr["action"])
        task = m.get("language", {}).get("high") or m.get("task") or "do the task"
        for t in range(T):
            fr = {k: v[t] for k, v in arr.items()}
            for cam in CAMS:
                fr[f"observation.images.{KEY[cam]}"] = image(ep, cam, t, fmt)
            fr["task"] = task
            ds.add_frame(fr)
        ds.save_episode()
        sources.append({"episode_index": len(sources), "raw": ep.name, "frames": T, "task": m.get("task"),
                        "scene_id": m.get("scene_id"), "score": m.get("score"), "language": task})
        n_new += 1
        print(f"[convert] {ep.name} -> episode {len(sources) - 1} ({T} frames)", flush=True)
    ds.finalize()
    sources_f.parent.mkdir(parents=True, exist_ok=True)
    sources_f.write_text(json.dumps(sources, indent=1))
    return {"out": str(out), "episodes": len(sources), "new": n_new, "fmt": fmt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, nargs="+", help="raw roots (episode-* dirs inside)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo_id", default="dxbench/dx_bed")
    ap.add_argument("--fmt", choices=["rby1", "dx24"], default="dx24")
    ap.add_argument("--all", action="store_true", help="also unsuccessful episodes")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--image_scale", type=float, default=1.0, help="dx24: resize stored frames (0.5 -> 240 x 320)")
    ap.add_argument("--exclude", nargs="*", default=[], help="raw episode dir names to leave out (e.g. sim errors)")
    a = ap.parse_args()
    global IMAGE_SCALE
    IMAGE_SCALE = a.image_scale
    res = None
    for r in a.raw:
        res = convert(r, a.out, a.repo_id, a.fmt, only_success=not a.all, limit=a.limit, exclude=a.exclude)
    print(f"[convert] {json.dumps(res)}")


if __name__ == "__main__":
    main()
