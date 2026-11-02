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
from dxbench.tasks.predicates import TEXT
from dxbench.tasks.scene_spec import (DEFAULT_LIGHTING, ObjSpec, SceneSpec, free_floor_ok, language, load_layout,
                                      sample_floor)

BASKET_SEP = 1.02   # garment centre to basket centre (m)
GARMENT_SEP = 0.85  # garment to garment (m)
SHIRTS = ("m95_wine",)  # user 2026-10-22: only the wine tee arrived (blue / green assets kept)
SHIRTS_TRAIN = ("m95_wine", "m95_blue", "m95_green")  # D-70 (user 2026-11-02): training scenes mix the product colours
# D-70: BED-03 training starts - one of three free spots (door side = CARE_SPAWN, east of the bed, north-west by the
# desk), facing anywhere (the oracle looks around, D-67). Evaluation keeps CARE_SPAWN. The spot between the bed and the
# sofa (0.4, 20.5) was dropped: the base got stuck there on geometry the layout boxes miss (tilt up to 70 deg, 1/7)
BED03_STARTS = ((-2.6, 17.7), (2.6, 17.9), (-3.1, 20.75))
_START_SALT = 4242
TOWELS = ("01_charcoalgray", "05_coolgray", "06_white", "07_oatmeal", "08_lightpink", "09_ivory", "10_lavender",
          "11_applegreen", "12_skyblue", "13_sandbeige")
BASKET_VARIANT = "gray"
HEAD_PITCH_DESK = 0.6  # rad, BED-02 head camera tilt (D-40)
DUVET_SIZE = (2.0, 1.55)  # m, along x before the spawn yaw (pi / 2: 2.0 m along the bed); objects.DUVET_SIZE
HEAD_PITCH_FLOOR = 0.6  # rad, every BED-03 variant (D-67): level, the head camera missed the floor within ~2.1 m (the
# second garment by the basket never came into view while looking around); 34 deg down it sees the floor from ~0.6 m (the
# near variant's first garment at 0.75 m) up to the horizon. One pitch for the task: near / second had 0.85
HEAD_PITCH_BED = 0.5  # rad, BED-01: level, the mattress top 0.9 m ahead is below the head camera's view (2026-10-22)
HEAD_PITCH_NEAR = 0.85  # rad, BED-03 near: the first garment 0.75 m ahead on the floor in the lower middle of the view
_COLOR_SALT = 54321
_DIVERSE_SALT = 97531
BLINDS = ("down", "half", "up")
LIGHTING = ("day", "day_sunny", "night")
# robot start jitter per task: (xy half range m, yaw half range rad). BED-02 stays small: the desk reach is tight.
START_JITTER = {"BED-01": (0.15, 0.15), "BED-02": (0.05, 0.10), "BED-03": (0.20, 0.30)}


def _color(seed: int, i: int, options: tuple) -> str:
    """Colour of garment i of a scene: its own generator (seed, i), no draw from the layout rng."""
    return options[int(np.random.default_rng([seed, _COLOR_SALT, i]).integers(len(options)))]


def _shirts(split: str, seed: int = 0):
    """Shirt variants indexed by garment number (kept as a sequence for the callers' i % len pattern)."""
    return tuple(_color(seed, 100 + k, SHIRTS_TRAIN if split == "train" else SHIRTS) for k in range(8))


def _towel(split: str, seed: int, i: int = 0) -> str:
    return _color(seed, i, TOWELS)


def _robot_facing(target_xy, start=CARE_SPAWN[:2]):
    return {"xy": list(start), "yaw": math.atan2(target_xy[1] - start[1], target_xy[0] - start[0]), "torso": "upright",
            "head_pitch": HEAD_PITCH_FLOOR}


def _on_mattress(rng, layout, margin=0.25):
    lo, hi = layout["parts_aabb"]["Bed/Mattress"]
    return [float(rng.uniform(lo[0] + margin, hi[0] - margin)), float(rng.uniform(lo[1] + margin, hi[1] - margin)),
            float(hi[2]) + 0.02]


def gen_bed03(diff: str, seed: int, split: str) -> SceneSpec:
    rng = np.random.default_rng(seed)
    L = load_layout()
    start, start_yaw = (CARE_SPAWN[0], CARE_SPAWN[1]), None
    if split == "train":  # D-70: own generator, so the layout draws stay those of the seed
        rs = np.random.default_rng([seed, _START_SALT])
        start, start_yaw = BED03_STARTS[int(rs.integers(len(BED03_STARTS)))], float(rs.uniform(-math.pi, math.pi))
    robot_disc = [(start, 0.9)]
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
    robot = _robot_facing(first.xyz[:2])
    if start_yaw is not None:
        robot = dict(robot, xy=list(start), yaw=start_yaw)
    return SceneSpec(scene_id=f"bed03_{diff}_{seed:03d}" if split == "eval" else f"bed03_{diff}_train_{seed}",
                     task="BED-03", difficulty=diff, seed=seed, robot=robot, objects=objs,
                     language=language("BED-03"))


def gen_bed03_near(diff: str, seed: int, split: str) -> SceneSpec:
    """BED-03 "near" (D-39): the easy layout of the same seed, but the robot starts NEAR_DIST in front of the first
    garment, facing it - pick + a short drive + put, so manipulation is learned / scored without the 4-5 m search.
    The start is on the garment's side towards the original start, rotated around the garment until the floor is
    free (robot footprint 0.45 m + margin)."""
    spec = gen_bed03("easy", seed, split)
    L = load_layout()
    g = next(o for o in spec.objects if o.kind in ("tshirt", "towel"))
    b = next(o for o in spec.objects if o.kind == "basket")
    gx, gy = g.xyz[0], g.xyz[1]
    a0 = math.atan2(CARE_SPAWN[1] - gy, CARE_SPAWN[0] - gx)
    others = [((o.xyz[0], o.xyz[1]), 0.55) for o in spec.objects if o is not g and o.kind in ("tshirt", "towel", "sock")]
    others.append(((b.xyz[0], b.xyz[1]), 0.65))
    for k in range(24):
        a = a0 + (k + 1) // 2 * (math.pi / 12) * (1 if k % 2 else -1)
        x, y = gx + NEAR_DIST * math.cos(a), gy + NEAR_DIST * math.sin(a)
        if free_floor_ok((x, y), L, 0.5, extra=others):
            break
    # head pitched down (2026-10-22): level, the head camera looks over a garment 0.75 m ahead on the floor; at the
    # BED-02 0.6 rad the garment is cut by the image bottom
    spec.robot = {"xy": [float(x), float(y)], "yaw": math.atan2(gy - y, gx - x), "torso": "upright",
                  "head_pitch": HEAD_PITCH_FLOOR}
    spec.difficulty = "near"
    spec.scene_id = f"bed03_near_{seed:03d}" if split == "eval" else f"bed03_near_train_{seed}"
    return spec


NEAR_DIST = 0.75  # robot base origin to the first garment (m)


def gen_bed03_second(diff: str, seed: int, split: str) -> SceneSpec:
    """BED-03 "second" (D-49, training only): the easy layout of the same seed with the first garment already
    dropped into the basket and the robot NEAR_DIST in front of the basket, facing it, on the side the first garment
    came from - the state right after the first put, so demos cover turning away, finding the second garment and
    bringing it (the v3 near ACT put the first garment 9/10 but then circled in place). The second garment keeps the
    easy spawn (and its settled cloth state)."""
    spec = gen_bed03("easy", seed, split)
    L = load_layout()
    g = next(o for o in spec.objects if o.kind in ("tshirt", "towel"))
    b = next(o for o in spec.objects if o.kind == "basket")
    bx, by = b.xyz[0], b.xyz[1]
    a0 = math.atan2(g.xyz[1] - by, g.xyz[0] - bx)
    g.xyz = [bx, by, 0.6]  # dropped flat over the basket centre while the cloth states are made
    g.yaw = b.yaw
    others = [((o.xyz[0], o.xyz[1]), 0.55) for o in spec.objects if o.kind in ("tshirt", "towel", "sock") and o is not g]
    for k in range(24):
        a = a0 + (k + 1) // 2 * (math.pi / 12) * (1 if k % 2 else -1)
        x, y = bx + NEAR_DIST * math.cos(a), by + NEAR_DIST * math.sin(a)
        if free_floor_ok((x, y), L, 0.5, extra=others):
            break
    spec.robot = {"xy": [float(x), float(y)], "yaw": math.atan2(by - y, bx - x), "torso": "upright",
                  "head_pitch": HEAD_PITCH_FLOOR}
    spec.difficulty = "second"
    spec.scene_id = f"bed03_second_{seed:03d}" if split == "eval" else f"bed03_second_train_{seed}"
    return spec


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
        # D-56 (2026-10-29, user): towel centre 0.20-0.22 m from the front edge (was 0.25 +- 0.03), yaw +-0.1 rad (was
        # +-0.2): leaning, the robot stops with its base front only ~2 cm under the desk (the chest meets the desk edge)
        # and the right arm reaches ~0.69 m ahead at desk height, so the half fold's far corners (0.70-0.81 m ahead)
        # were missed; the near long edge now lies about at the desk front, without hanging over it
        dy = 0.15 if split == "train" else 0.1  # D-70: the oracle aligns the base to the towel (D-68)
        objs.append(ObjSpec("cloth_0", "towel", [cx + 0.04 + float(rng.uniform(-0.01, 0.01)),
                                                 cy + float(rng.uniform(-dy, dy)),
                                                 top + 0.005], float(rng.uniform(-0.1, 0.1)) + math.pi / 2,
                            _towel(split, seed)))
    elif diff == "medium":
        objs.append(ObjSpec("cloth_0", "tshirt", [cx - 0.05, cy, top + 0.08], float(rng.uniform(-0.6, 0.6)) + math.pi / 2,
                            _shirts(split, seed)[0]))
    else:
        # heap long axes along the desk (y, 1.4 m, yaw within 0.25 rad), 0.40 m from the front edge and 0.6 m from
        # the -y edge (2026-10-21): 0.30 m from the front with any yaw, the top towel slid down the heap's front slope
        # and off the desk in 10/10 hard states (the robot pulls the heap towards itself anyway)
        for i, kind in enumerate(["tshirt", "tshirt", "towel", "tshirt", "towel"]):
            objs.append(ObjSpec(f"cloth_{i}", kind, [cx - 0.15, cy - 0.10, top + 0.05 + 0.06 * i],
                                float(rng.uniform(-0.25, 0.25)) + math.pi / 2,
                                _shirts(split, seed)[i] if kind == "tshirt" else _towel(split, seed, i)))
    # robot in front of the desk's long side (desk against the west wall: approach from +x). Head pitched down
    # (2026-10-21): level, the head camera looks over the desk and sees neither the desk top nor the garment; a
    # leaning start instead put the READY arms onto the desk and the towel.
    robot = {"xy": [float(hi[0]) + 0.75, cy], "yaw": math.pi, "torso": "upright", "head_pitch": HEAD_PITCH_DESK}
    if diff == "easy":  # D-66 (user 2026-11-01): start where the oracle used to end its approach - leaning (kept all
        # episode) and facing the desk; the oracle only turns in place (about the drive axle, AXLE_X 0.228 m ahead of the
        # base origin) to the towel's long axis (yaw pi + dev). After that turn the base stands where the approach used
        # to end: 0.33 m from the desk edge (the drive-in stalled at 0.31-0.32 m, chest at the edge; 1-2 cm back for the
        # spawn) with the towel centre straight ahead. Starting at the towel's y instead left it up to 5 cm to one side
        # and 2 cm further: the left hand missed the far corners (2026-11-01)
        dev = float(objs[0].yaw) - math.pi / 2
        bx = float(hi[0]) + 0.33
        f = (-math.cos(dev), -math.sin(dev))
        by = float(objs[0].xyz[1]) - (float(objs[0].xyz[0]) - bx) / f[0] * f[1]
        axle = 0.228
        robot = dict(robot, xy=[bx + axle * (1 - math.cos(dev)), by - axle * math.sin(dev)], torso="lean",
                     arms="desk_reach")  # hands over the desk, grippers straight down (user 2026-11-01)
    return SceneSpec(scene_id=f"bed02_{diff}_{seed:03d}" if split == "eval" else f"bed02_{diff}_train_{seed}",
                     task="BED-02", difficulty=diff, seed=seed, robot=robot, objects=objs, language=language("BED-02"))


def gen_bed01(diff: str, seed: int, split: str) -> SceneSpec:
    rng = np.random.default_rng(seed)
    L = load_layout()
    lo, hi = L["parts_aabb"]["Bed/Mattress"]
    top = float(hi[2])
    cx, cy = 0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])
    head_y = float(hi[1]) - 0.25
    # D-55 levels after the pi0.5 "Make the Bed" rubric (pillow not scored; it stays at the head end). The 2.0 x 1.55 m
    # duvet made up: head edge 2 cm short of the mattress end, 28 cm over each long side. Start coverage <= ~0.7.
    if diff == "easy":     # slid towards the far side: pull its near edge back (no driving). D-66 (user 2026-11-01):
        # 48-52 cm (was 57-59, D-57), so the edge lies 0.19-0.23 m in from the open side, 0.60-0.65 m ahead of the base
        # at the pull pose - where the crouched arm reaches it with the gripper straight down (at 0.75 m the wrist was
        # 35-53 deg off and the orientation servo shook the arm). Start coverage of the whole top ~0.80 (< 0.95)
        # head edge at the mattress end (not 2 cm short): the 2.0 m duvet has 12 cm over the 1.88 m mattress, and pulled
        # sideways it slid 2-3 cm towards the foot - the head row of cells came uncovered (D-66, 100 % rule)
        # D-70 (training): 47-56 cm (beyond 56 cm the gripper no longer reaches the edge straight down; planned 40 cm,
        # but under 48 cm the oracle left the head / foot rows bare in 7/9 gate scenes - the duvet ends up short along
        # the bed, which pulling across it cannot fix - vs 3/9 above) and +-0.03 rad (turned more, a head corner
        # uncovers the head row). The draws keep the evaluation sequence
        sl, dyaw = ((0.47, 0.56), 0.03) if split == "train" else ((0.48, 0.52), 0.0)
        duvet = ObjSpec("duvet", "duvet", [cx + float(rng.uniform(*sl)), float(hi[1]) - DUVET_SIZE[0] / 2,
                                           top + 0.02], math.pi / 2 + (float(rng.uniform(-dyaw, dyaw)) if dyaw else 0.0))
    elif diff == "medium":  # head part folded back over the foot half (make_cloth_states folds it)
        duvet = ObjSpec("duvet", "duvet", [cx, float(hi[1]) - 0.02 - DUVET_SIZE[0] / 2, top + 0.02], math.pi / 2)
    else:                   # folded, then gathered into a bundle on the foot half or on the open long edge
        duvet = ObjSpec("duvet", "duvet", [cx, float(hi[1]) - 0.02 - DUVET_SIZE[0] / 2, top + 0.02], math.pi / 2,
                        variant="bundle_foot" if seed % 2 == 0 else "bundle_edge")
    pillow_xyz = [cx + float(rng.uniform(-0.05, 0.05)), head_y, top + 0.08]
    objs = [duvet, ObjSpec("pillow", "pillow", pillow_xyz, float(rng.uniform(-0.1, 0.1)))]
    if diff == "easy":  # D-63 (user): no pillow on the easy level (not scored, only clutter); drawn anyway so the seeds
        objs = [duvet]  # keep the same layouts
    # facing the open long side (+x), 0.9 m from it on every level (D-55: at 1.3 m the hard start robot hit the pillar
    # by the window and tipped over). D-66 (user): easy starts crouched (the working posture, kept all episode) 0.35 m
    # behind the oracle's pull pose (base front 6 cm from the frame), facing the bed: drive straight in, align, pull
    robot = {"xy": [float(lo[0]) - 0.9, cy], "yaw": 0.0, "torso": "upright", "head_pitch": HEAD_PITCH_BED}
    if diff == "easy":
        # in line with the duvet's centre (the pull spot, D-66): only a straight drive in
        robot = dict(robot, xy=[float(lo[0]) - 0.025 - 0.06 - 0.32 - 0.35, float(duvet.xyz[1])], torso="crouch",
                     arms="bed_reach")
    return SceneSpec(scene_id=f"bed01_{diff}_{seed:03d}" if split == "eval" else f"bed01_{diff}_train_{seed}",
                     task="BED-01", difficulty=diff, seed=seed, robot=robot, objects=objs, language=language("BED-01"))


def diversify(spec: SceneSpec, split: str) -> SceneSpec:
    """Instruction paraphrase, blinds / room lighting and robot start jitter (D-42, step 2). Own generator per scene,
    so layouts and colours stay those of the same seed. Evaluation easy / near stay canonical (comparable with the
    earlier results); evaluation medium varies the instruction and blinds; hard and every training scene vary all."""
    level = 2 if split == "train" else {"easy": 0, "near": 0, "second": 0, "medium": 1, "hard": 2}[spec.difficulty]
    if level == 0:
        return spec
    rng = np.random.default_rng([spec.seed, _DIVERSE_SALT, {"BED-01": 1, "BED-02": 2, "BED-03": 3}[spec.task],
                                 spec.difficulty == "near"])
    spec.language = dict(spec.language, high=str(rng.choice(TEXT[spec.task].high)))
    light = dict(DEFAULT_LIGHTING, blinds=str(rng.choice(BLINDS)))
    if level == 2:
        light.update(lighting=str(rng.choice(LIGHTING)), intensity=round(float(rng.uniform(0.6, 1.3)), 2))
    spec.lighting = light
    if level == 2:
        dxy, dyaw = START_JITTER[spec.task]
        if spec.difficulty in ("near", "second"):
            dxy, dyaw = 0.08, 0.15
        if spec.task in ("BED-01", "BED-02") and spec.difficulty == "easy":  # D-66: the oracle only drives straight in
            # (BED-01) or turns in place (BED-02) - 10-15 cm off to the side, the hands over the bed caught on it while
            # driving in at an angle (9/18 failed). BED-02: turning in place left any offset (18/22 failed) until the
            # oracle aligned the base to the nominal pose first (D-68, user 2026-11-02: +-5 cm like BED-01, sim2real)
            # D-70: BED-01 +-10 cm / +-0.1 rad (the oracle aligns first, like BED-02), BED-02 +-8 cm / +-0.1 rad
            dxy, dyaw = (0.10, 0.10) if spec.task == "BED-01" else (0.08, 0.10)
        L = load_layout()
        others = [((o.xyz[0], o.xyz[1]), 0.55) for o in spec.objects if o.xyz[2] < 0.1]
        x0, y0 = spec.robot["xy"]
        for _ in range(20):
            x, y = x0 + float(rng.uniform(-dxy, dxy)), y0 + float(rng.uniform(-dxy, dxy))
            if spec.task == "BED-02" and spec.difficulty == "easy":  # D-66: starts at the desk: only away from it
                x = x0 + 0.6 * abs(x - x0)
            if spec.task == "BED-01" and spec.difficulty == "easy":  # D-70: hands 13 cm short of the mattress: not closer
                x = x0 - 0.6 * abs(x - x0)
            if spec.task != "BED-03" or free_floor_ok((x, y), L, 0.5, extra=others):
                spec.robot = dict(spec.robot, xy=[x, y], yaw=float(spec.robot["yaw"] + rng.uniform(-dyaw, dyaw)))
                break
    return spec


GEN = {"BED-01": gen_bed01, "BED-02": gen_bed02, "BED-03": gen_bed03}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(GEN))
    ap.add_argument("--difficulty", required=True, choices=["easy", "medium", "hard", "near", "second"],
                    help="near: BED-03 only, the easy layout with the robot in front of the first garment (D-39)")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--split", default="eval", choices=["eval", "train"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    for s in range(a.seed0, a.seed0 + a.n):
        gen = {"near": gen_bed03_near, "second": gen_bed03_second}.get(a.difficulty, GEN[a.task])
        spec = diversify(gen(a.difficulty, s, a.split), a.split)
        f = out / f"{spec.scene_id}.json"
        if f.exists():  # keep settled cloth states of objects whose spawn did not change
            old = {o.name: o for o in SceneSpec.from_json(f).objects}
            for o in spec.objects:
                p = old.get(o.name)
                if p and (p.kind, p.xyz, p.yaw, p.variant) == (o.kind, o.xyz, o.yaw, o.variant):
                    o.cloth_state = p.cloth_state
        spec.to_json(f)
        assert all(free_floor_ok(o.xyz[:2], load_layout(), 0.6) or o.xyz[2] > 0.1 or o.kind in ("duvet", "pillow")
                   for o in spec.objects if o.kind not in ("basket",)) or a.task != "BED-03"
    print(f"[scene_gen] wrote {a.n} {a.task} {a.difficulty} ({a.split}) scenes to {out}")


if __name__ == "__main__":
    main()
