"""Open-loop action error of a trained policy on held-out episodes, per task (D-70 T1-T4, no simulator).

    ~/miniforge3/envs/dx_policy/bin/python -m dxbench_policy.train.offline_eval \
        --checkpoint outputs/train/smolvla_t1/checkpoints/last/pretrained_model \
        --dataset data/lerobot/bed_all_d70_dx24 --split outputs/train/t1_split.json --out outputs/train/smolvla_t1_offline.json

Every `--every` frames of each held-out episode the policy (served exactly like serve/server.py) predicts an action
chunk from the recorded observation; its first `--horizon` actions are compared with the recorded ones. Reported per
task, mean absolute error by action group (arms rad, torso rad, base v m/s, base omega rad/s) and gripper agreement,
next to the "hold" baseline (keep the current joint positions, grippers open, base still).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

GROUPS = {"arms_rad": slice(0, 14), "torso_rad": slice(16, 22), "base_v": slice(22, 23), "base_w": slice(23, 24)}


def hold_chunk(state: np.ndarray, h: int) -> np.ndarray:
    a = state[:24].copy()
    a[14:16] = 1.0
    a[22:24] = 0.0
    return np.repeat(a[None], h, 0)


def errors(pred: np.ndarray, true: np.ndarray) -> dict:
    e = {k: float(np.abs(pred[:, s] - true[:, s]).mean()) for k, s in GROUPS.items()}
    e["gripper_agree"] = float(((pred[:, 14:16] >= 0.5) == (true[:, 14:16] >= 0.5)).mean())
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", required=True, help="json with 'holdout' episode indices (t1_split.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=15)
    ap.add_argument("--horizon", type=int, default=10)
    a = ap.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from dxbench_policy.serve.server import LeRobotBackend

    backend = LeRobotBackend(a.checkpoint, chunk=a.horizon)
    ds = LeRobotDataset("dxbench/offline_eval", root=a.dataset)
    src = {e["episode_index"]: e for e in json.loads((Path(a.dataset) / "meta/dx_sources.json").read_text())}
    hold = json.loads(Path(a.split).read_text())["holdout"]
    acc = defaultdict(lambda: {"policy": defaultdict(list), "hold": defaultdict(list), "frames": 0})
    for ep in hold:
        task = src[ep]["task"]
        lo = int(ds.meta.episodes["dataset_from_index"][ep])
        hi = int(ds.meta.episodes["dataset_to_index"][ep])
        actions = np.stack([np.asarray(x) for x in ds.hf_dataset.select(range(lo, hi))["action"]]).astype(np.float32)
        for t in range(0, hi - lo - a.horizon, a.every):
            s = ds[lo + t]
            obs = {"observation.state": s["observation.state"].numpy(), "task": s["task"]}
            for k in [k for k in s if k.startswith("observation.images.")]:
                obs[k] = (s[k].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            pred = backend.infer(obs)[: a.horizon]
            true = actions[t:t + a.horizon]
            for name, p in (("policy", pred), ("hold", hold_chunk(obs["observation.state"], a.horizon))):
                for k, v in errors(p, true).items():
                    acc[task][name][k].append(v)
            acc[task]["frames"] += 1
    out = {task: {"samples": r["frames"], **{name: {k: round(float(np.mean(v)), 4) for k, v in r[name].items()}
                                             for name in ("policy", "hold")}} for task, r in sorted(acc.items())}
    out["_config"] = {"checkpoint": a.checkpoint, "holdout": hold, "every": a.every, "horizon": a.horizon}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
