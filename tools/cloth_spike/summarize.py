"""Summarize reports/cloth_spike/runs/*_<tag>.json into a markdown table (stdout).

    python3 tools/cloth_spike/summarize.py [--tag m]
"""
import argparse
import glob
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="m")
a = ap.parse_args()
root = os.path.join(os.path.dirname(__file__), "../../reports/cloth_spike/runs")
rows = []
for f in sorted(glob.glob(os.path.join(root, f"*_{a.tag}.json"))):
    d = json.load(open(f))
    g, i = d["args"], d["info"]
    rows.append((d["run"], g["test"], g["grip"], g["dt"], g.get("cameras"), g.get("res_cm"), g.get("thickness"),
                 g.get("friction"), g.get("cloth_iters"), g.get("duration"), i.get("drift_mean_m"), i.get("drift_max_m"),
                 i.get("finite"), i.get("rtf"), i.get("num_nodes"), d["pass"],
                 i.get("lift_m", i.get("sleeve_end_y", i.get("target_nodes_travel_m")))))
print("| run | test | grip | Hz | cam | res cm | thick | μ | iters | sim s | drift mean cm | drift max cm | finite | RTF | nodes | pass | task metric |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
fmt = lambda v, s=1: "–" if v is None else (f"{v * s:.2f}" if isinstance(v, float) else str(v))
for r in rows:
    print("| " + " | ".join([r[0], r[1], r[2], str(r[3]), "Y" if r[4] else "", fmt(r[5]), fmt(r[6]), fmt(r[7]), fmt(r[8]),
                             fmt(r[9]), fmt(r[10], 100), fmt(r[11], 100), fmt(r[12]), fmt(r[13]), fmt(r[14]),
                             "**pass**" if r[15] else "fail", fmt(r[16])]) + " |")
