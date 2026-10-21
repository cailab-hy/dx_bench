"""Predicate unit tests with hand-built correct / wrong states (no simulator needed).

    ~/miniforge3/envs/dx_arena/bin/python -m pytest dxbench/tests/test_predicates.py -q
"""
import math

import numpy as np
import pytest

from dxbench.tasks import predicates as P

IDQ = (0.0, 0.0, 0.0, 1.0)


def sheet(cx, cy, z, sx, sy, n=20):
    xs = np.linspace(cx - sx / 2, cx + sx / 2, n)
    ys = np.linspace(cy - sy / 2, cy + sy / 2, n)
    g = np.stack(np.meshgrid(xs, ys, indexing="ij"), -1).reshape(-1, 2)
    return np.c_[g, np.full(len(g), z)]


def yaw_q(a):
    return (0.0, 0.0, math.sin(a / 2), math.cos(a / 2))


# ------------------------------------------------------------------ BED-03
BASKET = (np.array([1.0, 2.0, 0.0]), IDQ)


def test_bed03_all_in_basket_success():
    clothes = {"shirt": sheet(1.0, 2.0, 0.08, 0.3, 0.2), "towel": sheet(1.05, 2.0, 0.12, 0.25, 0.2)}
    r = P.evaluate_bed03(clothes, {}, *BASKET)
    assert r["success"] and r["score"] == pytest.approx(1.0)


def test_bed03_partial_and_rotated_basket():
    clothes = {"in": sheet(1.0, 2.0, 0.08, 0.2, 0.2), "out": sheet(2.0, 2.0, 0.0, 0.3, 0.3)}
    r = P.evaluate_bed03(clothes, {}, np.array([1.0, 2.0, 0.0]), yaw_q(0.7))
    assert not r["success"] and r["score"] == pytest.approx(0.5)


def test_bed03_cloth_draped_over_rim_does_not_count():
    # 70 % of the nodes outside the basket footprint
    c = sheet(1.0 + 0.35, 2.0, 0.3, 0.5, 0.2)
    r = P.evaluate_bed03({"c": c}, {}, *BASKET)
    assert r["score"] == 0.0


def test_bed03_distractor_penalty():
    clothes = {"a": sheet(1.0, 2.0, 0.08, 0.2, 0.2), "b": sheet(1.0, 2.05, 0.1, 0.2, 0.2)}
    r = P.evaluate_bed03(clothes, {"book": (np.array([1.0, 2.0, 0.1]), IDQ)}, *BASKET)
    assert r["score"] == pytest.approx(0.8) and not r["success"] and r["penalised"] == ["book"]


def test_bed03_folded_garment_penalty_and_tipped_basket():
    clothes = {"a": sheet(1.0, 2.0, 0.08, 0.2, 0.2)}
    r = P.evaluate_bed03(clothes, {}, *BASKET, folded={"f": sheet(1.0, 2.0, 0.15, 0.2, 0.2)})
    assert r["score"] == pytest.approx(0.8)
    tipped = (0.0, math.sin(math.radians(60) / 2), 0.0, math.cos(math.radians(60) / 2))
    r2 = P.evaluate_bed03(clothes, {}, np.array([1.0, 2.0, 0.0]), tipped)
    assert r2["failure"] == "basket_tipped" and r2["score"] == 0.0


# ------------------------------------------------------------------ BED-02
SPEC = P.FoldSpec(rest_area=0.93 * 0.66 * 0.6, stack_center=(0.4, 0.4), table_z=0.725)


def test_bed02_flat_shirt_scores_flatten_only():
    r = P.evaluate_bed02(sheet(0.0, 0.0, 0.73, 0.93, 0.66, 40), SPEC, folds_done=(False, False, False))
    assert r["stages"]["flatten"] and r["score"] == pytest.approx(0.3) and not r["success"]


def test_bed02_crumpled_shirt_zero():
    rng = np.random.default_rng(0)
    crumpled = np.c_[rng.normal(0, 0.06, (500, 2)), np.full(500, 0.75)]
    r = P.evaluate_bed02(crumpled, SPEC)
    assert not r["stages"]["flatten"] and r["score"] == 0.0


def test_bed02_full_fold_on_stack_success():
    folded = sheet(0.4, 0.4, 0.74, 0.30, 0.25)
    kp = {"body_center": np.array([0.4, 0.4, 0.74]), "left_sleeve": np.array([0.42, 0.41, 0.75]),
          "right_sleeve": np.array([0.38, 0.39, 0.75]), "hem": np.array([0.4, 0.45, 0.76]),
          "collar": np.array([0.4, 0.47, 0.74])}
    r = P.evaluate_bed02(folded, SPEC, keypoints=kp)
    assert r["success"] and r["score"] == pytest.approx(1.0)


def test_bed02_too_big_fold_is_not_full_success():
    folded = sheet(0.4, 0.4, 0.74, 0.50, 0.25)
    r = P.evaluate_bed02(folded, SPEC, folds_done=(True, True, True))
    assert r["score"] == pytest.approx(1.0) and not r["success"]


# ------------------------------------------------------------------ BED-02 topological folds (D-41)
def towel_uv(nx=41, ny=21):
    u, v = np.meshgrid(np.linspace(-0.4, 0.4, nx), np.linspace(-0.2, 0.2, ny), indexing="ij")
    return np.c_[u.ravel(), v.ravel()]


def test_bed02_fold_done_flat_vs_folded_vs_crumpled():
    uv = towel_uv()
    flat = np.c_[uv, np.full(len(uv), 0.73)]
    assert not any(P.fold_done(flat, uv, ax, c) for ax, c in P.FOLD_LINES["towel"].values())
    half = flat.copy()  # far long edge onto the near one: v -> -|v|, a second layer 3 mm up
    top = uv[:, 1] > 0
    half[top, 1] = -uv[top, 1]
    half[top, 2] += 0.003
    assert P.fold_done(half, uv, 1, 0.0) and not P.fold_done(half, uv, 0, 0.2)
    rng = np.random.default_rng(0)  # crumpled: shrunk to 0.6 and bumpy - extent shrinks, mirror pairs stay apart
    crumpled = np.c_[uv * 0.6 + rng.normal(0, 0.01, uv.shape), 0.73 + rng.uniform(0, 0.08, len(uv))]
    assert not any(P.fold_done(crumpled, uv, ax, c) for ax, c in P.FOLD_LINES["towel"].values())


def test_bed02_garments_score_flat_start_and_stack():
    uv = towel_uv()
    flat = np.c_[uv, np.full(len(uv), 0.73)]
    spec = P.FoldSpec(rest_area=0.0, stack_center=(0.0, 0.25), table_z=0.725)
    g = {"name": "cloth_0", "kind": "towel", "nodes": flat, "uv": uv, "start_flat": True, "start_centroid": (0, 0)}
    assert P.evaluate_bed02_garments([g], spec)["score"] == 0.0  # a flat start earns nothing for being flat
    # fully folded (half + both ends) and moved onto the stack spot
    f = flat.copy()
    top = uv[:, 1] > 0
    f[top, 1] = -uv[top, 1]
    for sgn in (1, -1):
        end = sgn * uv[:, 0] > 0.2
        f[end, 0] = sgn * 0.4 - f[end, 0]
    f[:, 1] += 0.25 + 0.1
    r = P.evaluate_bed02_garments([dict(g, nodes=f)], spec)
    assert r["success"] and r["score"] == pytest.approx(1.0), r


# ------------------------------------------------------------------ BED-01 (real 1.5 x 1.4 m duvet, D-41)
BED = P.BedSpec(mattress_min=(1.49, 19.03), mattress_max=(2.47, 20.91), mattress_top_z=0.53,
                headboard_center=(1.98, 20.66), cover_head_gap=0.45)
SPREAD = sheet(1.98, 19.76, 0.55, 1.5, 1.4, 40)  # the DX blanket's made-bed footprint (y 19.06 .. 20.46)


def test_bed01_made_bed_success():
    r = P.evaluate_bed01(SPREAD, (1.98, 20.62, 0.6), (3.0, 18.0, 0.0), BED)
    assert r["success"] and r["score"] == pytest.approx(1.0), r


def test_bed01_whole_mattress_target_was_impossible():
    old = P.BedSpec(**{**BED.__dict__, "cover_head_gap": 0.0})
    assert P.evaluate_bed01(SPREAD, (1.98, 20.62, 0.6), (3.0, 18.0, 0.0), old)["coverage"] < 0.9


def test_bed01_untouched_pillow_scores_nothing():
    duvet = sheet(1.98, 19.2, 0.6, 1.0, 0.4, 30)  # folded at the foot, pillow never moved
    r = P.evaluate_bed01(duvet, (1.98, 20.62, 0.6), (1.98, 20.62, 0.6), BED, pillow_was_moved=False)
    assert not r["stages"]["duvet_spread"] and not r["stages"]["pillow_placed"] and r["score"] == 0.0


def test_bed01_uneven_overhang():
    duvet = sheet(1.98 + 0.25, 19.76, 0.55, 1.5, 1.4, 40)  # shifted 25 cm to one side
    r = P.evaluate_bed01(duvet, (1.98, 20.62, 0.6), (3.0, 18.0, 0.0), BED)
    assert not r["stages"]["duvet_even"]


def test_bed01_duvet_on_floor_fails_only_when_much_of_it_is():
    corner = np.r_[SPREAD, sheet(0.9, 19.9, 0.01, 0.3, 0.3, 5)]  # a corner brushing the floor: no failure
    assert P.evaluate_bed01(corner, (1.98, 20.62, 0.6), (3.0, 18.0, 0.0), BED)["failure"] is None
    half = np.r_[SPREAD[:800], sheet(0.9, 19.9, 0.01, 0.5, 1.4, 30)]
    r = P.evaluate_bed01(half, (1.98, 20.62, 0.6), (3.0, 18.0, 0.0), BED)
    assert r["failure"] == "duvet_on_floor" and r["score"] == 0.0


def test_geometry_helpers():
    assert P.tilt_deg(IDQ) == pytest.approx(0.0)
    l, s = P.obb_extent_xy(sheet(0, 0, 0, 0.3, 0.2)[:, :2] @ np.array([[0.8, -0.6], [0.6, 0.8]]))
    assert l == pytest.approx(0.3, abs=1e-6) and s == pytest.approx(0.2, abs=1e-6)
    assert P.projected_area(sheet(0, 0, 0, 0.5, 0.5, 60)[:, :2]) == pytest.approx(0.25, rel=0.15)


def test_bed03_tapered_gray_basket():
    # real basket (D-36): interior narrows from 23.7 x 17.2 cm (half, rim) to 18.2 x 13.2 cm (half, floor)
    spec = P.BASKET_GRAY
    q = (0.0, 0.0, 0.0, 1.0)
    assert P.point_in_basket((0.22, 0.0, 0.35), (0, 0, 0), q, spec)      # near the rim: inside
    assert not P.point_in_basket((0.22, 0.0, 0.03), (0, 0, 0), q, spec)  # same x at the floor: outside the taper
    assert not P.point_in_basket((0.0, 0.20, 0.2), (0, 0, 0), q, spec)
    blob = np.c_[np.random.default_rng(0).uniform(-0.15, 0.15, (200, 2)), np.full(200, 0.05)]
    assert P.cloth_in_basket(blob, (0, 0, 0), q, spec)
