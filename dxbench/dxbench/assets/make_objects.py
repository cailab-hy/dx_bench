"""Generate the stage-3 object USDs into dxbench/assets/objects/ (Kit pxr; run in dx_arena, headless).

    python dxbench/dxbench/assets/make_objects.py

  laundry_basket.usda  (old, stage 3) rigid 60 x 44 x 32 cm open box, handle slots in the short walls, 0.8 kg
  laundry_basket_gray.usda  real product (D-36): tapered open basket, top 49 x 36, bottom 38 x 28, 40 cm high, gray
  tshirt_m95_{blue,wine,green}.usda  real M(95) T-shirt (D-36): the DX TShirt warped (thin-plate spline on outline
                       landmarks) to chest 49 / length 62.5 / shoulder 42 / sleeve 19 cm, cotton areal density
  tshirt_{s,m,l}_{color}.usda  copies of the DX TShirt with the cloth points baked at 0.85 / 1.0 / 1.15 scale
                       and a different colour (surface deformable, same physics material as the DX asset)
  doll.usda            rigid 0.3 kg (sphere head + capsule body), book.usda rigid 0.5 kg (23 x 16 x 3 cm)
Towels and socks are procedural surface-deformable rectangles (dxbench.assets.objects).
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(ap)
ap.add_argument("--real_only", action="store_true", help="only the real-product assets (D-36)")
args = ap.parse_args()
app = AppLauncher(args).app

from pathlib import Path  # noqa: E402

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt  # noqa: E402

from dxbench.paths import DX_ASSETS  # noqa: E402

OUT = Path(__file__).resolve().parent / "objects"
OUT.mkdir(exist_ok=True)


def new_stage(path: Path, root: str):
    if path.exists():
        path.unlink()
    st = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(st, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(st, 1.0)
    rp = UsdGeom.Xform.Define(st, f"/{root}")
    st.SetDefaultPrim(rp.GetPrim())
    return st, rp.GetPrim()


def preview_material(st, path: str, rgb):
    mat = UsdShade.Material.Define(st, path)
    sh = UsdShade.Shader.Define(st, f"{path}/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def physics_material(st, path, mu):
    m = UsdShade.Material.Define(st, path)
    api = UsdPhysics.MaterialAPI.Apply(m.GetPrim())
    api.CreateStaticFrictionAttr(mu)
    api.CreateDynamicFrictionAttr(mu * 0.85)
    api.CreateRestitutionAttr(0.0)
    return m


def box(st, path, size, pos, mat=None, pmat=None):
    c = UsdGeom.Cube.Define(st, path)
    c.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(c)
    xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
    xf.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    if mat:
        UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(mat)
    if pmat:
        UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(pmat, UsdShade.Tokens.weakerThanDescendants, "physics")
    return c


def rigid_root(prim, mass):
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)


def make_basket():
    st, root = new_stage(OUT / "laundry_basket.usda", "LaundryBasket")
    rigid_root(root, 0.8)
    m = preview_material(st, "/LaundryBasket/Looks/plastic", (0.55, 0.75, 0.9))
    pm = physics_material(st, "/LaundryBasket/Looks/physics", 0.6)
    L, W, H, t = 0.60, 0.44, 0.32, 0.01  # 60 x 44 x 32 cm: a 50 x 35 cm basket was too small to drop a T-shirt into (stage 3)
    box(st, "/LaundryBasket/bottom", (L, W, t), (0, 0, t / 2), m, pm)
    for s in (-1, 1):
        box(st, f"/LaundryBasket/wall_long_{'n' if s > 0 else 's'}", (L, t, H), (0, s * (W - t) / 2, H / 2), m, pm)
        # short wall with a 12 cm x 6 cm handle slot near the top
        x = s * (L - t) / 2
        tag = "e" if s > 0 else "w"
        box(st, f"/LaundryBasket/wall_short_{tag}_low", (t, W, H - 0.08), (x, 0, (H - 0.08) / 2), m, pm)
        post = (W - 0.12) / 2
        for k in (-1, 1):
            box(st, f"/LaundryBasket/wall_short_{tag}_post{'p' if k > 0 else 'm'}", (t, post, 0.08),
                (x, k * (0.06 + post / 2), H - 0.04), m, pm)
        box(st, f"/LaundryBasket/wall_short_{tag}_rim", (t, 0.12, 0.02), (x, 0, H - 0.01), m, pm)
    st.GetRootLayer().Save()


TSHIRT_VARIANTS = {"s_red": (0.85, (0.75, 0.12, 0.12)), "m_blue": (1.0, (0.15, 0.3, 0.75)),
                   "l_gray": (1.15, (0.55, 0.55, 0.55)), "m_white": (1.0, (0.92, 0.92, 0.9))}


def make_tshirts():
    src = DX_ASSETS / "TShirt" / "TShirt.usda"
    for name, (s, rgb) in TSHIRT_VARIANTS.items():
        dst = OUT / f"tshirt_{name}.usda"
        if dst.exists():
            dst.unlink()
        layer = Sdf.Layer.FindOrOpen(str(src))
        new = Sdf.Layer.CreateNew(str(dst))
        new.TransferContent(layer)
        st = Usd.Stage.Open(new)
        # drop the editor render settings carried by the DX asset
        for p in ("/Render",):
            if st.GetPrimAtPath(p):
                st.RemovePrim(p)
        for prim in st.Traverse():
            for a in prim.GetAttributes():
                tn = a.GetTypeName()
                if tn in (Sdf.ValueTypeNames.Point3fArray, Sdf.ValueTypeNames.Float3Array) and a.HasAuthoredValue():
                    v = a.Get()
                    if v is not None and len(v) and "extent" not in a.GetName():
                        a.Set(Vt.Vec3fArray([Gf.Vec3f(p[0] * s, p[1] * s, p[2] * s) for p in v]))
                    elif v is not None and a.GetName() == "extent":
                        a.Set(Vt.Vec3fArray([Gf.Vec3f(p[0] * s, p[1] * s, p[2]) for p in v]))
            if prim.GetTypeName() == "Shader" and prim.GetAttribute("inputs:diffuseColor"):
                prim.GetAttribute("inputs:diffuseColor").Set(Gf.Vec3f(*rgb))
            if prim.GetTypeName() == "Shader" and prim.GetAttribute("inputs:diffuse_color_constant"):
                prim.GetAttribute("inputs:diffuse_color_constant").Set(Gf.Vec3f(*rgb))
        new.customLayerData = {"source": str(src), "scale": s, "color": Gf.Vec3f(*rgb)}
        new.Save()


# ---------------------------------------------------------------------------------------- real products (D-36)
# Laundry basket (Coupang 7638582343, 대원화학 세탁바구니 대형, PP, body white / handle "gray"): top 49 x 36 cm,
# bottom 38 x 28 cm, height 40 cm. Colours sampled from the product photo (median of a lit patch, sRGB -> linear).
def srgb(hexstr):
    c = [int(hexstr[i:i + 2], 16) / 255.0 for i in (1, 3, 5)]
    return tuple(v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in c)


BASKET_REAL = {"top": (0.49, 0.36), "bottom": (0.38, 0.28), "height": 0.40, "wall": 0.008, "mass": 1.0,
               "body": "#d2d2ce", "rim": "#5e5f63", "hole": 0.022, "bar": 0.009, "band": 0.035, "rim_h": 0.022,
               "solid_u": 0.22}  # long walls: solid centre panel |u| < solid_u (photo), short walls fully gridded


def _slab(st, path, quad_in, quad_out, mat, pmat=None):
    """Slab between two parallel quads (4 x 3 each); with `pmat` it is also a convex-hull collider."""
    pts = [Gf.Vec3f(*p) for p in list(quad_in) + list(quad_out)]
    faces = [0, 1, 2, 3, 7, 6, 5, 4, 0, 4, 5, 1, 1, 5, 6, 2, 2, 6, 7, 3, 3, 7, 4, 0]
    m = UsdGeom.Mesh.Define(st, path)
    m.CreatePointsAttr(Vt.Vec3fArray(pts))
    m.CreateFaceVertexCountsAttr(Vt.IntArray([4] * 6))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray(faces))
    m.CreateSubdivisionSchemeAttr("none")
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
    if pmat is not None:
        UsdPhysics.CollisionAPI.Apply(m.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr("convexHull")
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(pmat, UsdShade.Tokens.weakerThanDescendants, "physics")
    else:
        m.CreatePurposeAttr("render")
    return m


def make_basket_real(spec=None, name="laundry_basket_gray"):
    """Tapered open basket, origin at the floor centre, x = long side. Collision: bottom + four solid leaning walls
    (convex slabs). Render: white walls with a square-hole grid (solid centre panels on the long walls), gray rim."""
    b = dict(BASKET_REAL, **(spec or {}))
    (Lt, Wt), (Lb, Wb), H, t = b["top"], b["bottom"], b["height"], b["wall"]
    st, root = new_stage(OUT / f"{name}.usda", "LaundryBasket")
    rigid_root(root, b["mass"])
    white = preview_material(st, "/LaundryBasket/Looks/body", srgb(b["body"]))
    gray = preview_material(st, "/LaundryBasket/Looks/rim", srgb(b["rim"]))
    pm = physics_material(st, "/LaundryBasket/Looks/physics", 0.6)
    box(st, "/LaundryBasket/bottom", (Lb, Wb, t), (0, 0, t / 2), white, pm)
    for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):  # feet
        c = UsdGeom.Cube.Define(st, f"/LaundryBasket/Render/foot_{i}")
        c.CreateSizeAttr(1.0)
        xf = UsdGeom.Xformable(c)
        xf.AddTranslateOp().Set(Gf.Vec3d(sx * (Lb / 2 - 0.03), sy * (Wb / 2 - 0.03), 0.004))
        xf.AddScaleOp().Set(Gf.Vec3f(0.05, 0.04, 0.008))
        UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(white)

    def half(z):
        f = z / H
        return (Lb + (Lt - Lb) * f) / 2, (Wb + (Wt - Wb) * f) / 2

    def P(axis, sign, u, z, inset=0.0):
        hx, hy = half(z)
        hx, hy = hx - inset, hy - inset
        return (sign * hx, u * hy, z) if axis == 0 else (u * hx, sign * hy, z)

    def piece(path, axis, sign, u0, u1, z0, z1, mat, pmat=None):
        q = [[P(axis, sign, u0, z0, k), P(axis, sign, u1, z0, k), P(axis, sign, u1, z1, k), P(axis, sign, u0, z1, k)]
             for k in (0.0, t)]
        _slab(st, path, q[0], q[1], mat, pmat)

    hole, bar, band, rim_h = b["hole"], b["bar"], b["band"], b["rim_h"]
    for axis in (0, 1):
        for s in (-1, 1):
            tag = f"{'short' if axis == 0 else 'long'}_{'p' if s > 0 else 'm'}"
            # collision: one solid slab per wall (cloth must not poke through the render-only holes)
            q = [[P(axis, s, -1, 0, k), P(axis, s, 1, 0, k), P(axis, s, 1, H, k), P(axis, s, -1, H, k)] for k in (0.0, t)]
            col = _slab(st, f"/LaundryBasket/wall_{tag}", q[0], q[1], white, pm)
            col.CreatePurposeAttr("guide")  # collider only; the grid below is what renders
            # render: bands + grid of bars leaving square holes
            r = f"/LaundryBasket/Render/{tag}"
            piece(f"{r}_bottom", axis, s, -1, 1, 0.0, band, white)
            piece(f"{r}_top", axis, s, -1, 1, H - band - rim_h, H - rim_h, white)
            hw = half(H / 2)[1 - axis]  # wall half width at mid height (u = 1)
            cu = 0.06 / hw  # corner posts 6 cm
            piece(f"{r}_cornerA", axis, s, -1, -1 + cu, band, H - band - rim_h, white)
            piece(f"{r}_cornerB", axis, s, 1 - cu, 1, band, H - band - rim_h, white)
            solid = b["solid_u"] if axis == 1 else 0.0
            if solid > 0:
                piece(f"{r}_panel", axis, s, -solid, solid, band, H - band - rim_h, white)
            pitch = hole + bar
            z = band
            k = 0
            while z + hole + bar <= H - band - rim_h + 1e-6:  # horizontal bars
                piece(f"{r}_h{k}", axis, s, -1 + cu, 1 - cu, z + hole, z + hole + bar, white)
                z += pitch
                k += 1
            for side, (ua, ub) in enumerate(((-1 + cu, -solid if solid else 0.0), ((solid if solid else 0.0), 1 - cu))):
                n_u = max(1, int(round((ub - ua) * hw / pitch)))
                for j in range(1, n_u):  # vertical bars
                    u = ua + (ub - ua) * j / n_u
                    du = bar / 2 / hw
                    piece(f"{r}_v{side}_{j}", axis, s, u - du, u + du, band, H - band - rim_h, white)
            # gray rim (the handles of the product are the long sides of this rim)
            q = [[P(axis, s, -1, H - rim_h, k), P(axis, s, 1, H - rim_h, k), P(axis, s, 1, H, k), P(axis, s, -1, H, k)]
                 for k in (-0.012, t)]
            _slab(st, f"{r}_rim", q[0], q[1], gray)
    root.SetCustomDataByKey("dxbench", {"top": Gf.Vec2f(*b["top"]), "bottom": Gf.Vec2f(*b["bottom"]), "height": H})
    st.GetRootLayer().Save()


# Short-sleeve T-shirt M(95) (Coupang 9705760403): chest half 49, length 62.5, shoulder 42, sleeve 19 cm, cotton 100 %.
TEE_REAL = {"chest_half": 0.49, "length": 0.625, "shoulder": 0.42, "sleeve": 0.19, "neck": 0.18,
            "armhole": 0.235, "shoulder_drop": 0.04, "sleeve_angle_deg": 30.0, "cuff": 0.165,
            "mass": 0.140}  # real tee weighed 140 g (user, 2026-10-21); density = mass / (thickness x mesh area)
TEE_COLORS = {"blue": "#2e4880", "wine": "#623944", "green": "#374a4a"}  # 유니미스 30수 무지 반팔티 photos (sRGB)


def _tps(src, dst):
    """2-D thin-plate spline mapping src landmarks to dst (returns f(points (N,2)) -> (N,2))."""
    import numpy as np

    src, dst = np.asarray(src, float), np.asarray(dst, float)
    n = len(src)

    def U(r2):
        with np.errstate(divide="ignore", invalid="ignore"):
            v = r2 * np.log(r2)
        return np.nan_to_num(v)

    K = U(((src[:, None] - src[None]) ** 2).sum(-1))
    P = np.c_[np.ones(n), src]
    A = np.zeros((n + 3, n + 3))
    A[:n, :n], A[:n, n:], A[n:, :n] = K, P, P.T
    sol = np.linalg.solve(A, np.r_[dst, np.zeros((3, 2))])
    w, a = sol[:n], sol[n:]

    def f(x):
        x = np.asarray(x, float)
        return np.c_[np.ones(len(x)), x] @ a + U(((x[:, None] - src[None]) ** 2).sum(-1)) @ w
    return f


def tee_landmarks(spec=None):
    """(DX TShirt outline landmarks, target landmarks), right side mirrored to the left. DX shirt (measured): body
    48 x 66 cm, flat top 41 cm at y = 0.33, armpit (0.24, 0.01), sleeve lower edge to (0.394, 0.015), cuff top
    (0.466, 0.18)."""
    import math

    t = dict(TEE_REAL, **(spec or {}))
    top = t["length"] / 2
    a = math.radians(t["sleeve_angle_deg"])
    sh = (t["shoulder"] / 2, top - t["shoulder_drop"])
    cuff_top = (sh[0] + t["sleeve"] * math.cos(a), sh[1] - t["sleeve"] * math.sin(a))
    cuff_bot = (cuff_top[0] - t["cuff"] * math.sin(a), cuff_top[1] - t["cuff"] * math.cos(a))
    arm = (t["chest_half"] / 2, top - t["armhole"])
    hem = (t["chest_half"] / 2, -top)
    # side seam: DX y -0.33..0.01 maps linearly onto hem..armpit (one mid point let the TPS bow the side in by 1 cm)
    side = [((0.24, y), (hem[0], hem[1] + (y + 0.33) / 0.34 * (arm[1] - hem[1]))) for y in (-0.25, -0.16, -0.08, -0.02)]
    pairs = side + [((0.24, -0.33), hem), ((0.24, 0.01), arm),
             ((0.394, 0.015), cuff_bot), ((0.466, 0.18), cuff_top), ((0.206, 0.33), sh),
             ((0.09, 0.33), (t["neck"] / 2, top))]
    src, dst = [], []
    for (sx, sy), (dx, dy) in pairs:
        src += [(sx, sy), (-sx, sy)]
        dst += [(dx, dy), (-dx, dy)]
    src += [(0.0, 0.33), (0.0, -0.33), (0.0, 0.0)]
    dst += [(0.0, top), (0.0, -top), (0.0, 0.0)]
    return src, dst


def make_real_tshirts(colors=None, spec=None):
    import numpy as np

    t = dict(TEE_REAL, **(spec or {}))
    f = _tps(*tee_landmarks(spec))
    src = DX_ASSETS / "TShirt" / "TShirt.usda"
    for cname, hexc in (colors or TEE_COLORS).items():
        rgb = srgb(hexc)
        dst = OUT / f"tshirt_m95_{cname}.usda"
        if dst.exists():
            dst.unlink()
        layer = Sdf.Layer.FindOrOpen(str(src))
        new = Sdf.Layer.CreateNew(str(dst))
        new.TransferContent(layer)
        st = Usd.Stage.Open(new)
        for p in ("/Render",):
            if st.GetPrimAtPath(p):
                st.RemovePrim(p)
        thick = None
        mats = []
        for prim in st.Traverse():
            for a_ in prim.GetAttributes():
                tn = a_.GetTypeName()
                if tn in (Sdf.ValueTypeNames.Point3fArray, Sdf.ValueTypeNames.Float3Array) and a_.HasAuthoredValue():
                    v = a_.Get()
                    if v is None or not len(v):
                        continue
                    P = np.array(v, dtype=float)
                    if a_.GetName() == "extent":
                        continue
                    P[:, :2] = f(P[:, :2])
                    a_.Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, q)) for q in P]))
            if prim.GetTypeName() == "Mesh":
                pts = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get())
                UsdGeom.Mesh(prim).GetExtentAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, pts.min(0))),
                                                                      Gf.Vec3f(*map(float, pts.max(0)))]))
            if prim.GetTypeName() == "Shader" and prim.GetAttribute("inputs:diffuseColor"):
                prim.GetAttribute("inputs:diffuseColor").Set(Gf.Vec3f(*rgb))
            if prim.GetTypeName() == "Shader" and prim.GetAttribute("inputs:diffuse_color_constant"):
                prim.GetAttribute("inputs:diffuse_color_constant").Set(Gf.Vec3f(*rgb))
            if prim.HasAPI("OmniPhysicsSurfaceDeformableMaterialAPI") or prim.GetAttribute("omniphysics:surfaceThickness"):
                th = prim.GetAttribute("omniphysics:surfaceThickness").Get()
                if th:
                    thick = th
                    mats.append(prim)
        # PhysX mass = density x surface thickness x area (both garment layers of the warped mesh)
        area = 0.0
        for prim in st.Traverse():
            if prim.GetTypeName() == "Mesh" and prim.HasAPI("OmniPhysicsDeformableBodyAPI"):
                P = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
                F = np.array(UsdGeom.Mesh(prim).GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
                area += 0.5 * np.linalg.norm(np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]]), axis=1).sum()
        for prim in mats:
            prim.GetAttribute("omniphysics:density").Set(float(t["mass"] / (thick * area)))
        new.customLayerData = {"source": str(src), "product": "M(95) short-sleeve tee, cotton 100 %",
                               "mass_kg": float(t["mass"]), "color_srgb": hexc, "thickness": float(thick or 0)}
        new.Save()


def make_doll_book():
    st, root = new_stage(OUT / "doll.usda", "Doll")
    rigid_root(root, 0.3)
    m = preview_material(st, "/Doll/Looks/fur", (0.6, 0.42, 0.25))
    pm = physics_material(st, "/Doll/Looks/physics", 0.8)
    body = UsdGeom.Capsule.Define(st, "/Doll/body")
    body.CreateRadiusAttr(0.07)
    body.CreateHeightAttr(0.08)
    body.CreateAxisAttr("Z")
    UsdGeom.Xformable(body).AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.11))
    head = UsdGeom.Sphere.Define(st, "/Doll/head")
    head.CreateRadiusAttr(0.06)
    UsdGeom.Xformable(head).AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.27))
    for g in (body, head):
        UsdPhysics.CollisionAPI.Apply(g.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(m)
        UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(pm, UsdShade.Tokens.weakerThanDescendants, "physics")
    st.GetRootLayer().Save()
    st, root = new_stage(OUT / "book.usda", "Book")
    rigid_root(root, 0.5)
    m = preview_material(st, "/Book/Looks/cover", (0.2, 0.45, 0.3))
    pm = physics_material(st, "/Book/Looks/physics", 0.6)
    box(st, "/Book/block", (0.23, 0.16, 0.03), (0, 0, 0.015), m, pm)
    st.GetRootLayer().Save()


if __name__ == "__main__":
    if args.real_only:
        make_basket_real()
        make_real_tshirts()
    else:
        make_basket()
        make_tshirts()
        make_doll_book()
        make_basket_real()
        make_real_tshirts()
    print("[objects]", sorted(p.name for p in OUT.glob("*.usda")))
    app.close()
