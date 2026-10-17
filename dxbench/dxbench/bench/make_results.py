"""Collect outputs/bench/*/summary.json into benchmarks/dx_bed_v0/RESULTS.md (one row per experiment x task x difficulty).

    ~/miniforge3/envs/dx_arena/bin/python -m dxbench.bench.make_results
"""
from __future__ import annotations

import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def main():
    rows = []
    for f in sorted((REPO / "outputs" / "bench").glob("*/summary.json")):
        s = json.loads(f.read_text())
        for g in s["groups"]:
            rows.append((s["experiment"], s["policy"].rsplit(".", 1)[-1], g))
    L = ["# dx_bed_v0 sample results", "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `python -m dxbench.bench.make_results` from "
         "`outputs/bench/*/summary.json`. Per-episode tables: `outputs/bench/<experiment>/REPORT.md`.", "",
         "| experiment | policy | task | difficulty | episodes | success | score mean | stage completion | tipped | errors |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for exp, pol, g in rows:
        st = ", ".join(f"{k.split('/')[-1]} {v:.0%}" for k, v in g["stage_rate"].items())
        L.append(f"| {exp} | {pol} | {g['task']} | {g['difficulty']} | {g['episodes']} | "
                 f"{g['success']}/{g['episodes']} ({g['success_rate']:.0%}) | {g['score_mean']:.2f} | {st} | "
                 f"{g['tipped']} | {g['errors']} |")
    out = REPO / "benchmarks" / "dx_bed_v0" / "RESULTS.md"
    out.write_text("\n".join(L) + "\n")
    print(f"[results] {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
