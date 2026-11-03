"""Count coverage of a training dataset's randomized conditions (D-70: the cheapest diversity measure with a reported
power law - Lin et al. 2024 - and one-to-one with the knobs we set). For each LeRobot dataset the scene of every
episode is read from meta/dx_sources.json (scene_id) and its scene JSON; the D-70 raw episodes also carry the oracle's
strategy and injected recovery (episode.json).

    python3 tools/diversity_report.py data/lerobot/bed02_oracle_easy_d68_dx24 data/lerobot/bed02_oracle_easy_d70_dx24

Prints per dataset: episodes, frames, number of distinct bins filled per variable, and the strategy / recovery mix.
"""
from __future__ import annotations

import collections
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCENE_DIRS = [REPO / "data/scenes_train_d70", REPO / "data/scenes_train"]
RAW = REPO / "data/raw"


def scene(sid: str, prefer_d70: bool) -> dict | None:
    dirs = SCENE_DIRS if prefer_d70 else SCENE_DIRS[::-1]
    for d in dirs:
        f = d / f"{sid}.json"
        if f.exists():
            return json.loads(f.read_text())
    return None


def bins(spec: dict, task: str) -> dict:
    """Binned condition values of one scene (bin widths: 1 cm / 0.01 rad for the small ranges, 0.5 m / 45 deg for
    BED-03's room-wide layout)."""
    r, objs = spec["robot"], {o["name"]: o for o in spec["objects"]}
    light = spec.get("lighting", {})
    out = {"light": (light.get("lighting"), light.get("blinds"), round(light.get("intensity", 1.0), 1)),
           "instruction": spec["language"]["high"]}
    if task == "BED-01":
        d = objs["duvet"]
        out.update(duvet_x=round(d["xyz"][0], 2), duvet_yaw=round(d["yaw"] - math.pi / 2, 2),
                   robot_dy=round((r["xy"][1] - d["xyz"][1]) / 0.02), robot_yaw=round(r["yaw"] / 0.02))
    elif task == "BED-02":
        t = objs["cloth_0"]
        out.update(towel_y=round(t["xyz"][1] / 0.02), towel_yaw=round((t["yaw"] - math.pi / 2) / 0.02),
                   towel_colour=t.get("variant"), robot_xy=(round(r["xy"][0] / 0.02), round(r["xy"][1] / 0.02)),
                   robot_yaw=round(r["yaw"] / 0.02))
    else:
        b = objs["laundry_basket"]
        g = [o for o in spec["objects"] if o["kind"] in ("tshirt", "towel")]
        out.update(basket_cell=(math.floor(b["xyz"][0] / 0.5), math.floor(b["xyz"][1] / 0.5),
                                round(b["yaw"] / (math.pi / 4)) % 8),
                   garment_cells=tuple(sorted((math.floor(o["xyz"][0] / 0.5), math.floor(o["xyz"][1] / 0.5)) for o in g)),
                   start=(round(r["xy"][0]), round(r["xy"][1])), robot_yaw8=round(r["yaw"] / (math.pi / 4)) % 8,
                   shirt=tuple(sorted(o.get("variant") for o in g if o["kind"] == "tshirt")))
    return out


def report(ds: Path) -> dict:
    src = json.loads((ds / "meta/dx_sources.json").read_text())
    task = src[0]["task"]
    d70 = "_d70" in ds.name
    seen = collections.defaultdict(set)
    strat, rec = collections.Counter(), collections.Counter()
    frames = 0
    for e in src:
        frames += e["frames"]
        sp = scene(e["scene_id"], d70)
        if sp is None:
            continue
        for k, v in bins(sp, task).items():
            seen[k].add(v)
        ep = RAW / ds.name.replace("_dx24", "") / e["raw"] / "episode.json"
        meta = json.loads(ep.read_text()).get("oracle_meta") if ep.exists() else None
        strat[(meta or {}).get("strategy", "single")] += 1
        rec[(meta or {}).get("recovery") or ("none" if meta else "n/a")] += 1
    return {"dataset": ds.name, "task": task, "episodes": len(src), "frames": frames,
            "filled_bins": {k: len(v) for k, v in seen.items()}, "strategy": dict(strat), "recovery": dict(rec)}


def main():
    for p in sys.argv[1:]:
        r = report(Path(p))
        print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
