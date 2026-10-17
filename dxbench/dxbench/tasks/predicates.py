"""Task predicates and partial scores for BED-01/02/03 — pure numpy (no Kit), so they can be unit-tested.

Conventions: world frame Z-up, metres. Cloth state = (V, 3) node positions. Rigid pose = position (3,) + quaternion
(x, y, z, w) (Isaac Lab 3.0 order). Each task has an `evaluate_*` function that returns a dict with
  stages:   {stage_name: bool}   (sub-goal flags in task order, current state)
  score:    partial score in [0, 1] (task card weights)
  success:  bool (full success), failure: str | None
Specs: docs/notes + PROGRESS (BED-01/02/03 task cards, DX Hub benchmark v0.1).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ------------------------------------------------------------------------------------------ geometry
def quat_to_mat(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def tilt_deg(q) -> float:
    R = quat_to_mat(q)
    return float(np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0))))


def to_local(points: np.ndarray, pos, quat) -> np.ndarray:
    """World points -> frame (pos, quat)."""
    return (np.asarray(points) - np.asarray(pos)) @ quat_to_mat(quat)


@dataclass
class Box:
    """Axis-aligned box in a local frame: centre (3,), half extents (3,)."""
    center: np.ndarray
    half: np.ndarray

    def contains(self, p: np.ndarray, margin: float = 0.0) -> np.ndarray:
        return np.all(np.abs(np.asarray(p) - self.center) <= self.half + margin, axis=-1)


def grid_coverage(nodes: np.ndarray, rect_min, rect_max, z_min: float, z_max: float, cell: float = 0.05,
                  reach: float = 0.05) -> float:
    """Fraction of a horizontal rectangle (grid cells) that has a cloth node within `reach` (xy) inside the z band."""
    sel = nodes[(nodes[:, 2] >= z_min) & (nodes[:, 2] <= z_max)]
    xs = np.arange(rect_min[0] + cell / 2, rect_max[0], cell)
    ys = np.arange(rect_min[1] + cell / 2, rect_max[1], cell)
    if len(sel) == 0 or len(xs) == 0 or len(ys) == 0:
        return 0.0
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    g = np.stack([gx.ravel(), gy.ravel()], axis=1)
    # bucket the nodes on a coarse hash grid to keep this O(N)
    covered = np.zeros(len(g), dtype=bool)
    ij = np.floor(sel[:, :2] / reach).astype(int)
    buckets: dict[tuple, list[int]] = {}
    for k, key in enumerate(map(tuple, ij)):
        buckets.setdefault(key, []).append(k)
    gij = np.floor(g / reach).astype(int)
    for n, (gi, gj) in enumerate(gij):
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                idx = buckets.get((gi + di, gj + dj))
                if idx and np.min(np.linalg.norm(sel[idx, :2] - g[n], axis=1)) <= reach:
                    covered[n] = True
                    break
            if covered[n]:
                break
    return float(covered.mean())


def projected_area(nodes_xy: np.ndarray, cell: float = 0.02) -> float:
    """Area (m^2) of the occupied cells of the node xy projection (robust to folds, unlike a convex hull)."""
    if len(nodes_xy) == 0:
        return 0.0
    cells = {tuple(c) for c in np.floor(np.asarray(nodes_xy) / cell).astype(int)}
    return len(cells) * cell * cell


def obb_extent_xy(nodes_xy: np.ndarray) -> tuple[float, float]:
    """Sides (long, short) of the minimum-area oriented rectangle of the xy points (rotating calipers on the hull)."""
    pts = np.asarray(nodes_xy, dtype=float)
    if len(pts) < 3:
        return (0.0, 0.0)
    from scipy.spatial import ConvexHull

    h = pts[ConvexHull(pts).vertices]
    best = (np.inf, 0.0, 0.0)
    for i in range(len(h)):
        e = h[(i + 1) % len(h)] - h[i]
        n = np.linalg.norm(e)
        if n < 1e-9:
            continue
        u = e / n
        v = np.array([-u[1], u[0]])
        a, b = h @ u, h @ v
        w, l = a.max() - a.min(), b.max() - b.min()
        if w * l < best[0]:
            best = (w * l, max(w, l), min(w, l))
    return best[1], best[2]


# ------------------------------------------------------------------------------------------ BED-03 laundry
@dataclass
class BasketSpec:
    """Laundry basket interior in the basket frame (origin at the floor centre of the basket)."""
    inner_half: tuple = (0.29, 0.21)  # m (outer 60 x 44 cm minus 1 cm walls; assets/make_objects.py)
    inner_height: float = 0.31
    tip_deg: float = 45.0


def cloth_in_basket(nodes: np.ndarray, basket_pos, basket_quat, spec: BasketSpec = BasketSpec(),
                    frac: float = 0.6) -> bool:
    """A garment counts as in the basket when >= `frac` of its nodes are inside the basket interior volume."""
    loc = to_local(nodes, basket_pos, basket_quat)
    inside = (np.abs(loc[:, 0]) <= spec.inner_half[0] + 0.01) & (np.abs(loc[:, 1]) <= spec.inner_half[1] + 0.01) \
        & (loc[:, 2] >= -0.02) & (loc[:, 2] <= spec.inner_height + 0.10)
    return bool(inside.mean() >= frac)


def point_in_basket(p, basket_pos, basket_quat, spec: BasketSpec = BasketSpec()) -> bool:
    loc = to_local(np.asarray(p)[None], basket_pos, basket_quat)[0]
    return bool(abs(loc[0]) <= spec.inner_half[0] and abs(loc[1]) <= spec.inner_half[1]
                and -0.02 <= loc[2] <= spec.inner_height + 0.10)


def evaluate_bed03(clothes: dict[str, np.ndarray], distractors: dict[str, tuple], basket_pos, basket_quat,
                   folded: dict[str, np.ndarray] | None = None, spec: BasketSpec = BasketSpec()) -> dict:
    """Score = clothes in basket / all clothes - 0.2 per distractor or folded garment in the basket (clip 0..1).
    Failure: basket tipped over (> 45 deg)."""
    inb = {k: cloth_in_basket(v, basket_pos, basket_quat, spec) for k, v in clothes.items()}
    bad = [k for k, (p, _q) in distractors.items() if point_in_basket(p, basket_pos, basket_quat, spec)]
    bad += [k for k, v in (folded or {}).items() if cloth_in_basket(v, basket_pos, basket_quat, spec)]
    n = max(1, len(clothes))
    score = float(np.clip(sum(inb.values()) / n - 0.2 * len(bad), 0.0, 1.0))
    tipped = tilt_deg(basket_quat) > spec.tip_deg
    return {"stages": {f"in_basket/{k}": v for k, v in inb.items()}, "in_basket": inb, "penalised": bad,
            "score": 0.0 if tipped else score, "success": (not tipped) and all(inb.values()) and not bad,
            "failure": "basket_tipped" if tipped else None}


# ------------------------------------------------------------------------------------------ BED-02 folding
@dataclass
class FoldSpec:
    rest_area: float           # flat reference projected area (m^2) of the garment
    target_size: tuple = (0.35, 0.30)  # completed fold bbox (m), T-shirt card
    stack_center: tuple = (0.0, 0.0)   # world xy of the stack spot
    stack_radius: float = 0.12
    table_z: float = 0.725


def evaluate_bed02(nodes: np.ndarray, spec: FoldSpec, keypoints: dict[str, np.ndarray] | None = None,
                   folds_done: tuple[bool, bool, bool] | None = None) -> dict:
    """Weights: flattened (area >= 80 % of rest) 0.3, three folds 0.15 each, on the stack 0.25.
    Fold flags come from keypoints (left/right sleeve tips onto the body, hem onto the collar) when given,
    else from `folds_done`. Full success also needs the fold bbox within target_size (+2 cm) and no node outside it
    sticking out (sleeves)."""
    area = projected_area(nodes[:, :2])
    flattened = area >= 0.8 * spec.rest_area
    if keypoints is not None:
        kp = keypoints
        body = kp["body_center"][:2]
        f1 = np.linalg.norm(kp["left_sleeve"][:2] - body) < 0.10
        f2 = np.linalg.norm(kp["right_sleeve"][:2] - body) < 0.10
        f3 = np.linalg.norm(kp["hem"][:2] - kp["collar"][:2]) < 0.12
        folds = (bool(f1), bool(f2), bool(f3))
    else:
        folds = tuple(bool(x) for x in (folds_done or (False, False, False)))
    centre = nodes[:, :2].mean(axis=0)
    on_stack = bool(np.linalg.norm(centre - np.asarray(spec.stack_center)) <= spec.stack_radius
                    and nodes[:, 2].min() >= spec.table_z - 0.02)
    long_, short_ = obb_extent_xy(nodes[:, :2])
    size_ok = long_ <= max(spec.target_size) + 0.02 and short_ <= min(spec.target_size) + 0.02
    # a flattened state is reached before folding: once folded, the area shrinks, so count flattened as reached
    # when any fold is done
    flat_flag = flattened or any(folds)
    score = 0.3 * flat_flag + 0.15 * sum(folds) + 0.25 * on_stack
    return {"stages": {"flatten": bool(flat_flag), "fold_left": folds[0], "fold_right": folds[1], "fold_half": folds[2],
                       "on_stack": on_stack}, "area_m2": area, "fold_bbox_m": (long_, short_),
            "score": float(score), "success": bool(all(folds) and on_stack and size_ok), "failure": None}


# ------------------------------------------------------------------------------------------ BED-01 make bed
@dataclass
class BedSpec:
    mattress_min: tuple  # world xy of the mattress top rectangle
    mattress_max: tuple
    mattress_top_z: float
    headboard_center: tuple  # world xy where the pillow should sit (headboard side, centred)
    long_axis: int = 1       # 0: x, 1: y — the bed length axis (DX bed: along y)
    floor_z: float = 0.0


def evaluate_bed01(duvet: np.ndarray, pillow_pos, pillow_initial_pos, spec: BedSpec,
                   pillow_was_moved: bool | None = None) -> dict:
    """Weights: pillow cleared 0.1, duvet covers >= 90 % of the mattress top 0.5, left/right overhang difference
    <= 15 cm 0.2, pillow within 20 cm of the headboard centre 0.2. Failure: duvet touching the floor."""
    cov = grid_coverage(duvet, spec.mattress_min, spec.mattress_max, spec.mattress_top_z - 0.03,
                        spec.mattress_top_z + 0.25)
    covered = cov >= 0.9
    side = 1 - spec.long_axis  # the axis across the bed
    lo, hi = spec.mattress_min[side], spec.mattress_max[side]
    over_lo = max(0.0, lo - float(duvet[:, side].min()))
    over_hi = max(0.0, float(duvet[:, side].max()) - hi)
    even = abs(over_lo - over_hi) <= 0.15
    moved = bool(pillow_was_moved) if pillow_was_moved is not None else \
        bool(np.linalg.norm(np.asarray(pillow_pos)[:2] - np.asarray(pillow_initial_pos)[:2]) > 0.3)
    placed = bool(np.linalg.norm(np.asarray(pillow_pos)[:2] - np.asarray(spec.headboard_center)) <= 0.20
                  and pillow_pos[2] >= spec.mattress_top_z - 0.02)
    touching_floor = bool(duvet[:, 2].min() <= spec.floor_z + 0.03)
    score = 0.1 * moved + 0.5 * covered + 0.2 * (even and covered) + 0.2 * placed
    return {"stages": {"pillow_cleared": moved, "duvet_spread": covered, "duvet_even": bool(even and covered),
                       "pillow_placed": placed}, "coverage": cov, "overhang": (over_lo, over_hi),
            "score": 0.0 if touching_floor else float(score),
            "success": bool(covered and even and placed and moved and not touching_floor),
            "failure": "duvet_on_floor" if touching_floor else None}


# ------------------------------------------------------------------------------------------ language
@dataclass
class TaskText:
    high: list[str]
    subtasks: list[str] = field(default_factory=list)


TEXT = {
    "BED-01": TaskText(high=["Make the bed.", "침대 정리해 줘"],
                       subtasks=["pick up the pillow", "spread the duvet", "grasp the top corners",
                                 "pull toward the headboard", "smooth the duvet", "place the pillow"]),
    "BED-02": TaskText(high=["Fold the clothes.", "옷 개어 줘"],
                       subtasks=["pick up a shirt", "flatten the shirt", "align the shirt", "fold the left side",
                                 "fold the right side", "fold in half", "place on the stack"]),
    "BED-03": TaskText(high=["Put the clothes in the laundry basket.", "방에 있는 옷 정리해 줘"],
                       subtasks=["look for clothes", "go to the shirt", "pick up the shirt", "go to the basket",
                                 "put it in the basket", "carry the basket to the door"]),
}
