"""Policy-server round trip with real dataset frames (dx_policy env): connect, reset, send N observations, check the
returned chunks.

    ~/miniforge3/envs/dx_policy/bin/python -m dxbench_policy.serve.roundtrip --port 8765 \
        --dataset data/lerobot/bed03_oracle_dx24 [--n 5] [--out reports/stage5/roundtrip.json]

Observations are dataset frames 0, S, 2S, ... (--stride S; state + the three cameras as uint8 HWC, the dataset's stored
size; the server resizes to the policy input), sent exactly as `rby1_lerobot_remote` sends them. The first returned action
of each chunk is compared with the recorded action of that frame (open-loop error per action group): a policy trained on
the dataset should be close on its own training frames, which checks the observation / action plumbing end to end.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np

from dxbench_policy.serve.server import mpn


def frames(dataset: str, n: int, stride: int = 10):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("dxbench/local", root=dataset)
    task = None
    for i in range(n):
        it = ds[min(stride * i, len(ds) - 1)]
        obs = {"observation.state": it["observation.state"].numpy().astype(np.float32),
               "_action": it["action"].numpy().astype(np.float32)}
        for k in ("front", "right", "left"):
            img = it[f"observation.images.{k}"]  # (C, H, W) float in [0, 1]
            obs[f"observation.images.{k}"] = (img.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        task = obs["task"] = it.get("task", "")
        yield obs
    return task


GROUPS = {"arms": slice(0, 14), "grippers": slice(14, 16), "torso": slice(16, 22), "base": slice(22, 24)}


async def run(port: int, dataset: str, n: int, stride: int = 10) -> dict:
    import websockets

    async with websockets.connect(f"ws://127.0.0.1:{port}", compression=None, max_size=None) as ws:
        meta = mpn.unpackb(await ws.recv())
        await ws.send(mpn.packb({"reset": True}))
        r = mpn.unpackb(await ws.recv())
        assert r.get("reset"), r
        lat, shapes, finite, errs = [], [], True, []
        for obs in frames(dataset, n, stride):
            gt = obs.pop("_action")
            t0 = time.perf_counter()
            await ws.send(mpn.packb(obs))
            msg = await ws.recv()
            if isinstance(msg, str):
                raise RuntimeError(f"server error:\n{msg}")
            lat.append(1000 * (time.perf_counter() - t0))
            a = np.asarray(mpn.unpackb(msg)["actions"])
            shapes.append(list(a.shape))
            finite &= bool(np.isfinite(a).all())
            if a.shape[1] == gt.shape[0]:
                errs.append(np.abs(a[0] - gt))
        e = np.array(errs) if errs else None
        return {"open_loop_abs_err": ({g: round(float(e[:, sl].mean()), 4) for g, sl in GROUPS.items()}
                                      if e is not None and e.shape[1] == 24 else None),
                "backend": meta.get("backend"), "fmt": meta.get("fmt"), "action_dim": meta.get("action_dim"),
                "requests": n, "chunk_shapes": shapes, "finite": finite,
                "latency_ms_first": round(lat[0], 1), "latency_ms_mean_rest": round(float(np.mean(lat[1:] or lat)), 1),
                "ok": finite and all(s[1] == meta.get("action_dim") for s in shapes)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = asyncio.run(run(a.port, a.dataset, a.n, a.stride))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"[roundtrip] {json.dumps(res)}")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
