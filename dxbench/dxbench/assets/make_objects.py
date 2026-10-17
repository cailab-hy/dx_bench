"""Generate the stage-3 object USDs into dxbench/assets/objects/ (Kit pxr; run in dx_arena, headless).

    python dxbench/dxbench/assets/make_objects.py

  laundry_basket.usda  rigid, 50 x 35 x 30 cm open box (1 cm walls), handle slots in the short walls, 0.8 kg
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
    make_basket()
    make_tshirts()
    make_doll_book()
    print("[objects]", sorted(p.name for p in OUT.glob("*.usda")))
    app.close()
