"""Generate scene JSONs per task x difficulty (pure python; needs tasks/care_room_layout.json).

    python -m dxbench.tasks.scene_gen --task BED-03 --difficulty easy --n 10 --seed0 0 --out benchmarks/dx_bed_v0/scenes
Evaluation scenes use seeds 0..999 (benchmarks/dx_bed_v0), training / demo scenes 10000+ (data/scenes_train) so
the two never overlap. Garments are the real products (D-36): M(95) T-shirt in 3 colours, 40 x 80 cm towel in 10
colours, the gray laundry basket. Colours are drawn per garment from all colours in both splits (D-38, user
2026-10-21: evaluation mixes colours) with their own generator, so the layouts stay those of the stage-3 seeds.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from dxbench.paths import CARE_SPAWN
from dxbench.tasks.scene_spec import ObjSpec, SceneSpec, free_floor_ok, language, load_layout, sample_floor

BASKET_SEP = 1.02   # garment centre to basket centre (m)
GARMENT_SEP = 0.85  # garment to garment (m)
SHIRTS = ("m95_blue", "m95_wine", "m95_green")
TOWELS = ("01_charcoalgray", "05_coolgray", "06_white", "07_oatmeal", "08_lightpink", "09_ivory", "10_lavender",
          "11_applegreen", "12_skyblue", "13_sandbeige")
BASKET_VARIANT = "gray"
_COLOR_SALT = 54321


def _color(seed: int, i: int, options: tuple) -> str:
    """Colour of garment i of a scene: its own generator (seed, i), no draw from the layout rng."""
    return options[int(np.random.default_rng([seed, _COLOR_SALT, i]).integers(len(options)))]


def _shirts(split: str, seed: int = 0):
    """Shirt variants indexed by garment number (kept as a sequence for the callers' i % len pattern)."""
    return tuple(_color(seed, 100 + k, SHIRTS) for k in range(8))


def _towel(split: str, seed: int, i: int = 0) -> str:
    return _color(seed, i, TOWELS)


def _robot_facing(target_xy, start=CARE_SPAWN[:2]):
    return {"xy": list(start), "yaw": math.atan2(target_xy[1] - start[1], target_xy[0] - start[0]), "torso": "upright"}


def _on_mattress(rng, layout, margin=0.25):
    lo, hi = layout["parts_aabb"]["Bed/Mattress"]
    return [float(rng.uniform(lo[0] + margin, hi[0] - margin)), float(rng.uniform(lo[1] + margin, hi[1] - margin)),
            float(hi[2]) + 0.02]


def gen_bed03(diff: str, seed: int, split: str) -> SceneSpec:
    rng = np.random.default_rng(seed)
    L = load_layout()
    robot_disc = [((CARE_SPAWN[0], CARE_SPAWN[1]), 0.9)]
    shirts = _shirts(split, seed)
    n_floor = {"easy": 2, "medium": 3, "hard": 4}[diff]
    # spacing from the garment size: basket half diagonal 0.37 m + large T-shirt radius ~0.55 m + 0.10 m, so no
    # garment settles under the basket (a lifted shirt dragged the basket over, stage 3); garments 0.85 m apart
    b_sep, g_sep = BASKET_SEP, GARMENT_SEP
    for attempt in range(400):
        scale = 1.0 if attempt < 300 else 0.9  # rare crowded layouts: shrink the spacing a little
        b = sample_floor(rng, L, 1, clearance=0.75, avoid=robot_disc)[0]
        try:
            pts = sample_floor(rng, L, n_floor, clearance=0.7, sep=g_sep * scale,
                               avoid=robot_disc + [(b, b_sep * scale)])
        except RuntimeError:
            continue
        # easy: basket and the two garments clustered near each other in the open floor
        if diff != "easy" or all(math.hypot(p[0] - b[0], p[1] - b[1]) < 2.0 for p in pts):
            break
    objs = [ObjSpec("laundry_basket", "basket", [b[0], b[1], 0.0], float(rng.uniform(-math.pi, math.pi)), BASKET_VARIANT)]
    kinds = ["tshirt", "towel", "tshirt", "sock", "tshirt", "towel"]
    k = 0
    for i, p in enumerate(pts):
        kind = kinds[k % len(kinds)]
        k += 1
        var = shirts[i % len(shirts)] if kind == "tshirt" else (_towel(split, seed, i) if kind == "towel" else None)
        objs.append(ObjSpec(f"cloth_{i}", kind, [p[0], p[1], 0.01], float(rng.uniform(-math.pi, math.pi)), var))
    if diff in ("medium", "hard"):
        objs.append(ObjSpec(f"cloth_{len(pts)}", "tshirt", _on_mattress(rng, L), float(rng.uniform(-1, 1)),
                            shirts[0]))
    if diff == "hard":
        objs.append(ObjSpec(f"cloth_{len(pts) + 1}", "towel", _on_mattress(rng, L), 0.0, _towel(split, seed, 7)))
        extra = sample_floor(rng, L, 3, clearance=0.6, sep=0.5,
                             avoid=robot_disc + [(b, 0.75)] + [((p[0], p[1]), 0.6) for p in pts])
        objs += [ObjSpec("doll", "doll", [*extra[0], 0.0], float(rng.uniform(-3, 3))),
                 ObjSpec("book", "book", [*extra[1], 0.0], float(rng.uniform(-3, 3))),
                 ObjSpec("folded_towel", "towel", [*extra[2], 0.01], 0.0, "folded:" + _towel(split, seed, 8))]
    first = next(o for o in objs if o.kind in ("tshirt", "towel"))
    return SceneSpec(scene_id=f"bed03_{diff}_{seed:03d}" if split == "eval" else f"bed03_{diff}_train_{seed}",
                     task="BED-03", difficulty=diff, seed=seed, robot=_robot_facing(first.xyz[:2]), objects=objs,
                     language=language("BED-03"))


def gen_bed02(diff: str, seed: int, split: str) -> SceneSpec:
    rng = np.random.default_rng(seed)
    L = load_layout()
    lo, hi = L["parts_aabb"].get("CareDesk", L["furniture_aabb"]["CareDesk"])
    top = float(hi[2])
    cy = 0.5 * (lo[1] + hi[1])
    # garments 0.25-0.30 m from the desk's front edge (robot side, +x): the arm reaches ~0.8 m ahead of the base at
    # desk height and the base must stay ~0.37 m from the edge, so the desk centre (0.75 m ahead) leaves the far
    # towel corners out of reach (reports/rby1_check/t5/reach_desk_*.json)
    cx = float(hi[0]) - 0.25
    objs = []
    if diff == "easy":
        objs.append(ObjSpec("cloth_0", "towel", [cx + float(rng.uniform(-0.03, 0.03)), cy + float(rng.uniform(-0.1, 0.1)),
                                                 top + 0.005], float(rng.uniform(-0.2, 0.2)) + math.pi / 2,
                            _towel(split, seed)))
    elif diff == "medium":
        objs.append(ObjSpec("cloth_0", "tshirt", [cx - 0.05, cy, top + 0.08], float(rng.uniform(-0.6, 0.6)) + math.pi / 2,
                            _shirts(split, seed)[0]))
    else:
        for i, kind in enumerate(["tshirt", "tshirt", "towel", "tshirt", "towel"]):
            objs.append(ObjSpec(f"cloth_{i}", kind, [cx - 0.05, cy - 0.25, top + 0.05 + 0.06 * i], float(rng.uniform(-1, 1)),
                                _shirts(split, seed)[i] if kind == "tshirt" else _towel(split, seed, i)))
    # robot in front of the desk's long side (desk against the west wall: approach from +x). Starts leaning
    # (2026-10-21): upright, the head camera looks over the desk and sees neither the desk top nor the garment.
    robot = {"xy": [float(hi[0]) + 0.75, cy], "yaw": math.pi, "torso": "lean"}
    return SceneSpec(scene_id=f"bed02_{diff}_{seed:03d}" if split == "eval" else f"bed02_{diff}_train_{seed}",
                     task="BED-02", difficulty=diff, seed=seed, robot=robot, objects=objs, language=language("BED-02"))


def gen_bed01(diff: str, seed: int, split: str) -> SceneSpec:
    rng = np.random.default_rng(seed)
    L = load_layout()
    lo, hi = L["parts_aabb"]["Bed/Mattress"]
    top = float(hi[2])
    cx, cy = 0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])
    head_y = float(hi[1]) - 0.25
    if diff == "easy":     # duvet folded at the foot (state file carries the fold; spawn flat over the foot half)
        duvet = ObjSpec("duvet", "duvet", [cx, float(lo[1]) + 0.45, top + 0.02], math.pi / 2)
        pillow_xyz = [cx + float(rng.uniform(-0.1, 0.1)), head_y, top + 0.08]
    elif diff == "medium":  # crumpled on the bed
        duvet = ObjSpec("duvet", "duvet", [cx + float(rng.uniform(-0.1, 0.1)), cy + float(rng.uniform(-0.2, 0.2)),
                                           top + 0.15], float(rng.uniform(-0.4, 0.4)) + math.pi / 2)
        pillow_xyz = [cx + float(rng.uniform(-0.15, 0.15)), head_y - float(rng.uniform(0.0, 0.3)), top + 0.08]
    else:                   # half of the duvet on the floor (open long side), pillow on the floor
        side_x = float(lo[0]) - 0.35
        duvet = ObjSpec("duvet", "duvet", [side_x + 0.2, cy, top + 0.05], math.pi / 2)
        pillow_xyz = [side_x - 0.4, cy + float(rng.uniform(-0.5, 0.5)), 0.08]
    objs = [duvet, ObjSpec("pillow", "pillow", pillow_xyz, float(rng.uniform(-0.3, 0.3)))]
    robot = {"xy": [float(lo[0]) - 0.9, cy], "yaw": 0.0, "torso": "upright"}  # facing the open long side (+x)
    return SceneSpec(scene_id=f"bed01_{diff}_{seed:03d}" if split == "eval" else f"bed01_{diff}_train_{seed}",
                     task="BED-01", difficulty=diff, seed=seed, robot=robot, objects=objs, language=language("BED-01"))


GEN = {"BED-01": gen_bed01, "BED-02": gen_bed02, "BED-03": gen_bed03}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(GEN))
    ap.add_argument("--difficulty", required=True, choices=["easy", "medium", "hard"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--split", default="eval", choices=["eval", "train"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    for s in range(a.seed0, a.seed0 + a.n):
        spec = GEN[a.task](a.difficulty, s, a.split)
        spec.to_json(out / f"{spec.scene_id}.json")
        assert all(free_floor_ok(o.xyz[:2], load_layout(), 0.6) or o.xyz[2] > 0.1 or o.kind in ("duvet", "pillow")
                   for o in spec.objects if o.kind not in ("basket",)) or a.task != "BED-03"
    print(f"[scene_gen] wrote {a.n} {a.task} {a.difficulty} ({a.split}) scenes to {out}")


if __name__ == "__main__":
    main()
