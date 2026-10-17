"""Validate a converted dataset against its raw episodes (dx_policy env).

    ~/miniforge3/envs/dx_policy/bin/python -m dxbench_policy.convert.validate --out data/lerobot/dx_bed03_oracle_dx24 \
        --raw data/raw/bed03_oracle

Checks per episode (meta/dx_sources.json maps LeRobot episodes to raw episodes):
  - frame count == raw trajectory length, timestamps = t / 15 (within 1e-4 s)
  - observation.state / action bit-exact against a recomputation from the raw arrays
  - decoded video frames (first, middle, last) close to the raw PNGs (MSE < 50; rby1 format compares the adapted image)
Writes <out>/meta/dx_validation.json and exits non-zero on any failure.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from dxbench_policy.convert.raw_to_lerobot import CAMS, KEY, episode_arrays, image, load_raw


def validate(out, raw_roots, max_eps=None) -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    out = Path(out)
    sources = json.loads((out / "meta" / "dx_sources.json").read_text())
    ds = LeRobotDataset(repo_id="local/validate", root=out)
    fmt = "rby1" if ds.meta.features["observation.state"]["shape"][0] == 16 else "dx24"
    if fmt == "dx24":  # stored image size -> the converter's scale
        import dxbench_policy.convert.raw_to_lerobot as C

        C.IMAGE_SCALE = ds.meta.features["observation.images.front"]["shape"][0] / 480.0  # read by image()
    raw_dirs = {}
    for r in raw_roots:
        for p in Path(r).glob("episode-*"):
            raw_dirs[p.name] = p
    report, ok_all = [], True
    for s in sources[:max_eps]:
        ep = s["episode_index"]
        m, a, _ = load_raw(raw_dirs[s["raw"]])
        ref = episode_arrays(m, a, fmt)
        lo = int(ds.meta.episodes["dataset_from_index"][ep])
        hi = int(ds.meta.episodes["dataset_to_index"][ep])
        n = hi - lo
        res = {"episode": ep, "raw": s["raw"], "frames": n, "raw_frames": len(ref["action"])}
        res["count_ok"] = n == len(ref["action"])
        hf = ds.hf_dataset.select(range(lo, hi))
        st = np.stack([np.asarray(x) for x in hf["observation.state"]])
        ac = np.stack([np.asarray(x) for x in hf["action"]])
        ts = np.asarray(hf["timestamp"], dtype=np.float64)
        res["state_exact"] = bool(st.shape == ref["observation.state"].shape and np.array_equal(st, ref["observation.state"]))
        res["action_exact"] = bool(ac.shape == ref["action"].shape and np.array_equal(ac, ref["action"]))
        res["timestamps_ok"] = bool(np.allclose(ts, np.arange(n) / ds.fps, atol=1e-4))
        mse = {}
        for t in sorted({0, n // 2, n - 1}):
            item = ds[lo + t]
            for cam in CAMS:
                k = f"observation.images.{KEY[cam]}"
                dec = (item[k].permute(1, 2, 0).numpy() * 255.0).round().astype(np.float32)
                raw = image(raw_dirs[s["raw"]], cam, t, fmt).astype(np.float32)
                mse[f"{KEY[cam]}@{t}"] = float(((dec - raw) ** 2).mean()) if dec.shape == raw.shape else float("inf")
        res["video_mse_max"] = max(mse.values())
        res["video_ok"] = res["video_mse_max"] < 50.0
        res["ok"] = all(res[k] for k in ("count_ok", "state_exact", "action_exact", "timestamps_ok", "video_ok"))
        ok_all &= res["ok"]
        report.append(res)
        print(f"[validate] {json.dumps(res)}", flush=True)
    summary = {"dataset": str(out), "fmt": fmt, "episodes": len(report), "frames": int(ds.meta.total_frames),
               "ok": bool(ok_all), "per_episode": report}
    (out / "meta" / "dx_validation.json").write_text(json.dumps(summary, indent=1))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw", nargs="+", required=True)
    ap.add_argument("--max_eps", type=int, default=None)
    a = ap.parse_args()
    s = validate(a.out, a.raw, a.max_eps)
    print(f"[validate] ok={s['ok']} episodes={s['episodes']} frames={s['frames']}")
    sys.exit(0 if s["ok"] else 1)


if __name__ == "__main__":
    main()
