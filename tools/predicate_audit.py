"""Predicate audit (step 2, 2026-10-21): evaluate every scene's *initial* state offline (saved cloth states, no Kit).

    ~/miniforge3/envs/dx_arena/bin/python tools/predicate_audit.py [--glob 'benchmarks/dx_bed_v0/scenes/bed02_*.json']

At t = 0 nothing has been done, so every stage except the ones a scene legitimately starts in should be False and
no failure should fire (a failure terminates the episode at once). Prints per (task, difficulty) how many scenes
start with each stage True, and the scenes that start failed.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "dxbench"))
from dxbench.tasks import predicates as P  # noqa: E402

LAYOUT = json.load(open(os.path.join(REPO, "dxbench/dxbench/tasks/care_room_layout.json")))
TOWEL = (0.80, 0.40)
TEE = (0.75, 0.625)


def nodes_of(o):
    return np.load(os.path.join(REPO, o["cloth_state"]))["nodes"] if o.get("cloth_state") else None


_UV = {}


def uv_for(kind):
    """Rest (u, v) of a garment mesh: the cache a sim run writes (bed_tasks._rest_uv), else a settled flat state
    as a proxy (towel: BED-02 easy 000 on the desk, tee: BED-03 easy 000 on the floor)."""
    if kind not in _UV:
        f = os.path.join(REPO, f"outputs/cache/rest_uv_{kind}.npz")
        if os.path.exists(f):
            _UV[kind] = np.load(f)["uv"]
        else:
            sid = "bed02_easy_000" if kind == "towel" else "bed03_easy_000"
            spec = json.load(open(os.path.join(REPO, f"benchmarks/dx_bed_v0/scenes/{sid}.json")))
            o = next(o for o in spec["objects"] if o["kind"] == kind)
            _UV[kind] = P.rest_uv(nodes_of(o), o["xyz"], o["yaw"])
    return _UV[kind]


def eval_scene(spec):
    t = spec["task"]
    objs = {o["name"]: o for o in spec["objects"]}
    if t == "BED-03":
        b = objs["laundry_basket"]
        q = (0.0, 0.0, math.sin(b["yaw"] / 2), math.cos(b["yaw"] / 2))
        bspec = P.basket_spec(b.get("variant"))
        clothes = {k: nodes_of(o) for k, o in objs.items() if o["kind"] in ("tshirt", "towel", "sock")
                   and not (o.get("variant") or "").startswith("folded")}
        folded = {k: nodes_of(o) for k, o in objs.items() if (o.get("variant") or "").startswith("folded")}
        dis = {k: (np.array(o["xyz"]), (0, 0, 0, 1)) for k, o in objs.items() if o["kind"] in ("doll", "book")}
        return P.evaluate_bed03(clothes, dis, b["xyz"], q, folded=folded, spec=bspec)
    if t == "BED-02":
        desk = LAYOUT["parts_aabb"].get("CareDesk") or LAYOUT["furniture_aabb"]["CareDesk"]
        ys = [o["xyz"][1] for o in spec["objects"] if o["kind"] in ("towel", "tshirt")]
        stack = (float(desk[1][0] - 0.40), float(np.mean(ys)))
        gs = []
        for o in spec["objects"]:
            if o["kind"] not in P.FOLD_LINES:
                continue
            n = nodes_of(o)
            gs.append({"name": o["name"], "kind": o["kind"], "nodes": n, "uv": uv_for(o["kind"]),
                       "start_flat": P.projected_area(n[:, :2]) >= 0.8 * P.REST_AREA[o["kind"]],
                       "start_centroid": n[:, :2].mean(axis=0)})
        return P.evaluate_bed02_garments(gs, P.FoldSpec(rest_area=0.0, stack_center=stack, table_z=float(desk[1][2])))
    if t == "BED-01":
        m = LAYOUT["parts_aabb"]["Bed/Mattress"]
        bed = P.BedSpec(mattress_min=tuple(m[0][:2]), mattress_max=tuple(m[1][:2]), mattress_top_z=float(m[1][2]),
                        headboard_center=(0.5 * (m[0][0] + m[1][0]), float(m[1][1]) - 0.25), long_axis=1,
                        cover_head_gap=float(m[1][1]) - float(LAYOUT["parts_aabb"]["Bed/Blanket"][1][1]))
        pillow = np.array(objs["pillow"]["xyz"])
        return P.evaluate_bed01(nodes_of(objs["duvet"]), pillow, pillow, bed, pillow_was_moved=False)
    raise ValueError(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="benchmarks/dx_bed_v0/scenes/*.json")
    a = ap.parse_args()
    true_counts = defaultdict(Counter)
    totals = Counter()
    failed = defaultdict(list)
    scores = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(REPO, a.glob))):
        spec = json.load(open(f))
        key = (spec["task"], spec["difficulty"])
        r = eval_scene(spec)
        totals[key] += 1
        scores[key].append(r["score"])
        for s, v in r["stages"].items():
            true_counts[key][s.split("/")[0]] += bool(v)
        if r.get("failure"):
            failed[key].append((spec["scene_id"], r["failure"]))
    for key in sorted(totals):
        stages = ", ".join(f"{s} {c}/{totals[key]}" for s, c in true_counts[key].items())
        print(f"{key[0]} {key[1]:6s} n={totals[key]}  start score mean {np.mean(scores[key]):.2f}  True at t=0: {stages}"
              + (f"  FAILED at t=0: {len(failed[key])} ({failed[key][0][1]})" if failed[key] else ""))


if __name__ == "__main__":
    main()
