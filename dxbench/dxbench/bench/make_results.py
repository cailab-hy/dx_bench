"""Collect outputs/bench/*/summary.json into benchmarks/dx_bed_v0/RESULTS.md (one row per experiment x task x difficulty).

    ~/miniforge3/envs/dx_arena/bin/python -m dxbench.bench.make_results

Policies served from dx_policy are named by the server backend in the experiment YAML (hold / replay / lerobot + the
checkpoint's policy type), since the dx_arena side is always the same `rby1_lerobot_remote` client.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]


def policy_name(exp_dir: Path, summary: dict) -> str:
    name = summary["policy"].rsplit(".", 1)[-1]
    f = exp_dir / "experiment.yaml"
    srv = (yaml.safe_load(f.read_text()) or {}).get("server") if f.exists() else None
    if not srv:
        return name
    backend = (re.search(r"--backend\s+(\S+)", srv) or [None, "lerobot"])[1]
    if backend == "lerobot":
        ck = (re.search(r"--checkpoint\s+(\S+)", srv) or [None, ""])[1]
        cfg = REPO / ck / "config.json"
        typ = json.loads(cfg.read_text()).get("type", "lerobot") if cfg.exists() else "lerobot"
        run = Path(ck).parts[2] if len(Path(ck).parts) > 2 else ck
        return f"{typ.upper()} (`{run}`) via server"
    if backend == "replay":
        ep = (re.search(r"--episode\s+(\d+)", srv) or [None, "0"])[1]
        return f"replay of demo {ep} via server"
    return f"{backend} via server"


def main():
    rows = []
    for f in sorted((REPO / "outputs" / "bench").glob("*/summary.json")):
        s = json.loads(f.read_text())
        pol = policy_name(f.parent, s)
        for g in s["groups"]:
            rows.append((s["experiment"], pol, g))
    L = ["# dx_bed_v0 sample results", "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `python -m dxbench.bench.make_results` from "
         "`outputs/bench/*/summary.json`. Per-episode tables: `outputs/bench/<experiment>/REPORT.md`. "
         "Experiment definitions: `benchmarks/dx_bed_v0/experiments/<experiment>.yaml`.", "",
         "| experiment | policy | task | difficulty | episodes | success | score mean | stage completion "
         "| stage time (s) | tipped | task failures | errors |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for exp, pol, g in rows:
        st = ", ".join(f"{k.split('/')[-1]} {v:.0%}" for k, v in g["stage_rate"].items())
        tt = ", ".join(f"{k.split('/')[-1]} {v:.0f}" for k, v in (g.get("stage_time_s_mean") or {}).items()
                       if v is not None) or "-"
        nf = len(g.get("failures") or [])
        L.append(f"| {exp} | {pol} | {g['task']} | {g['difficulty']} | {g['episodes']} | "
                 f"{g['success']}/{g['episodes']} ({g['success_rate']:.0%}) | {g['score_mean']:.2f} | {st} | {tt} | "
                 f"{g['tipped']} | {nf} | {g['errors']} |")
    out = REPO / "benchmarks" / "dx_bed_v0" / "RESULTS.md"
    out.write_text("\n".join(L) + "\n")
    print(f"[results] {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
