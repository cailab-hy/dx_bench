"""Task-weighted frame sampling for LeRobot training (D-70).

LeRobot 0.6.1 draws every frame with the same probability (EpisodeAwareSampler, a shuffled permutation per epoch), so a
task's share of the batches is its share of the frames - BED-01 (10 s episodes) next to BED-03 (100 s) got ~3 %.
TaskWeightedSampler keeps the sampler interface (epoch, resume state, length) but draws frames with replacement so that
task g gets the share n_g^alpha / sum_h n_h^alpha, n = the task's frames (pi0 weights its task-robot mixtures with
alpha 0.43; alpha 1 = frame-proportional, 0 = equal per task). The task of an episode comes from the dataset's
meta/dx_sources.json (raw_to_lerobot). Each epoch's draw is a pure function of (seed, epoch), like the parent's.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets import EpisodeAwareSampler

logger = logging.getLogger(__name__)


def episode_groups(root: str | Path) -> list[str]:
    """Task of every LeRobot episode, in episode-index order."""
    src = json.loads((Path(root) / "meta" / "dx_sources.json").read_text())
    return [e["task"] for e in sorted(src, key=lambda e: e["episode_index"])]


def task_shares(frames: dict[str, int], alpha: float) -> dict[str, float]:
    w = {g: float(n) ** alpha for g, n in frames.items() if n > 0}
    tot = sum(w.values())
    return {g: v / tot for g, v in w.items()}


class TaskWeightedSampler(EpisodeAwareSampler):
    groups: list[str] | None = None  # per episode; set before lerobot's train() builds the sampler
    alpha: float = 0.43

    def __init__(self, dataset_from_indices, dataset_to_indices, episode_indices_to_use=None, drop_n_first_frames=0,
                 drop_n_last_frames=0, shuffle=False, seed=0, absolute_to_relative_idx=None):
        super().__init__(dataset_from_indices, dataset_to_indices, episode_indices_to_use=episode_indices_to_use,
                         drop_n_first_frames=drop_n_first_frames, drop_n_last_frames=drop_n_last_frames,
                         shuffle=shuffle, seed=seed, absolute_to_relative_idx=absolute_to_relative_idx)
        frm = np.asarray(dataset_from_indices, dtype=np.int64)
        to = np.asarray(dataset_to_indices, dtype=np.int64)
        if self.groups is None or len(self.groups) != len(frm):
            raise ValueError(f"TaskWeightedSampler.groups has {0 if self.groups is None else len(self.groups)} entries "
                             f"for {len(frm)} episodes")
        used = np.ones(len(frm), dtype=bool)  # the parent's episode selection
        if episode_indices_to_use is not None:
            used = np.zeros(len(frm), dtype=bool)
            used[np.asarray(episode_indices_to_use, dtype=np.int64)] = True
        lengths = (to - drop_n_last_frames) - (frm + drop_n_first_frames)
        used &= lengths > 0
        g = np.asarray([str(x) for x in self.groups])[used]
        n = lengths[used]
        frames = {str(k): int(n[g == k].sum()) for k in sorted(set(g))}
        self.shares = task_shares(frames, self.alpha)
        per_frame = np.array([self.shares[str(k)] / frames[str(k)] for k in g], dtype=np.float64)
        self._w = torch.as_tensor(np.repeat(per_frame, n))
        logger.info("task-weighted sampling (alpha %.2f): frames %s -> shares %s", self.alpha, frames,
                    {k: round(v, 3) for k, v in self.shares.items()})

    def _iter_epoch(self, epoch: int, start: int):
        if not self.shuffle:
            yield from super()._iter_epoch(epoch, start)
            return
        order = torch.multinomial(self._w, self._num_frames, replacement=True, generator=self._epoch_generator(epoch))
        for k in range(start, self._num_frames):
            yield self._frame_index(int(order[k]))
