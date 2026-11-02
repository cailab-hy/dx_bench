"""lerobot-train with task-weighted frame sampling (D-70, weighted_sampler.TaskWeightedSampler).

    DX_TASK_ALPHA=0.43 python -m dxbench_policy.train.train_weighted <lerobot-train arguments>

The arguments are lerobot-train's own; the dataset root (--dataset.root) must hold meta/dx_sources.json. The task
shares are logged at start-up; DX_TASK_ALPHA=1 reproduces lerobot's frame-proportional sampling in expectation.
"""
from __future__ import annotations

import logging
import os
import sys
from collections import Counter

from dxbench_policy.train import weighted_sampler as W


def main():
    root = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--dataset.root=")), None)
    if root is None:
        raise SystemExit("train_weighted: pass --dataset.root=<dataset dir>")
    W.TaskWeightedSampler.groups = W.episode_groups(root)
    W.TaskWeightedSampler.alpha = float(os.environ.get("DX_TASK_ALPHA", "0.43"))
    print(f"[train_weighted] {len(W.TaskWeightedSampler.groups)} episodes {dict(Counter(W.TaskWeightedSampler.groups))}"
          f" alpha {W.TaskWeightedSampler.alpha}", flush=True)
    import lerobot.scripts.lerobot_train as T

    T.EpisodeAwareSampler = W.TaskWeightedSampler
    logging.getLogger(W.__name__).setLevel(logging.INFO)
    T.main()


if __name__ == "__main__":
    main()
