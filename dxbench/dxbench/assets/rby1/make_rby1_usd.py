"""Build dxbench/assets/rby1/rby1_dx.usda: the RB-Y1 robot layer used by the dxbench embodiments.

    python dxbench/dxbench/assets/rby1/make_rby1_usd.py          # (dx_arena, Kit pxr; headless)

A thin layer over `submodules/rby1-sim-isaac/assets/generated/rby1_fullbody_collision.usd` (the source files
are not modified). On top of the stage-0 override (tools/rby1_check/make_override.py) it adds
  1. 24 body DOF: wheels/torso/head FixedJoint -> RevoluteJoint + drive, `fixed_base` inactive,
     ArticulationRootAPI moved from the /RBY1 Xform to the `base` rigid body.
  2. Physics materials inside the asset: wheels mu 1.2, base caster cylinders mu 0 (combine "min"; the casters
     are sliding cylinders). Their collision scopes are de-instanced so the binding can be authored.
  3. Convex-hull colliders (purpose guide) for the links without any: link_torso_3, link_head_0/1/2,
     link_{left,right}_arm_6, ee_{left,right} — hulls of the link's own visual meshes.
  4. tool xform: the {side}_gripper Xform is moved so that ee_{side} sits 0.1261 m below link_*_arm_6, matching
     the tool_{side} fixed joint (the source USD has it at 0.1101 m; PhysX snaps to the joint at start).
The gripper mimic (l2 = -l1) cannot be authored: PhysxMimicJointAPI only accepts rotational reference axes and
is deprecated, so dxbench drives l2/r2 with mirrored targets in software (DECISIONS D-3).
Writes a JSON summary next to the layer.
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="rby1_fullbody_collision.usd")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt  # noqa: E402
from scipy.spatial import ConvexHull  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "rby1_dx.usda"
SRC = HERE.parents[3] / "submodules" / "rby1-sim-isaac" / "assets" / "generated" / args.base
REVOLUTE = ["left_wheel", "right_wheel", "torso_0", "torso_1", "torso_2", "torso_3", "torso_4", "torso_5",
            "head_0", "head_1"]
HULL_LINKS = ["link_torso_3", "link_head_0", "link_head_1", "link_head_2", "link_left_arm_6", "link_right_arm_6",
              "left_gripper/ee_left", "right_gripper/ee_right"]
TOOL_OFFSET = 0.1261
PROXIES = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)


def M(prim, cache):
    return np.array(cache.GetLocalToWorldTransform(prim))  # row-vector convention


def main():
    if OUT.exists():
        OUT.unlink()
    layer = Sdf.Layer.CreateNew(str(OUT))
    layer.subLayerPaths.append(os.path.relpath(SRC, OUT.parent))
    stage = Usd.Stage.Open(layer)
    stage.SetEditTarget(Usd.EditTarget(layer))
    root = stage.GetPrimAtPath("/RBY1")
    stage.SetDefaultPrim(root)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    layer.customLayerData = {"generator": "dxbench/assets/rby1/make_rby1_usd.py", "source": SRC.name}
    summary = {"source": str(SRC), "hulls": {}, "materials": {}}

    # 1. joints + articulation root ---------------------------------------------------------------
    for n in REVOLUTE:
        spec = Sdf.CreatePrimInLayer(layer, f"/RBY1/joints/{n}")
        spec.specifier = Sdf.SpecifierDef
        spec.typeName = "PhysicsRevoluteJoint"
        lo = Sdf.TokenListOp()
        lo.prependedItems = ["PhysicsDriveAPI:angular", "PhysicsJointStateAPI:angular"]
        spec.SetInfo("apiSchemas", lo)
    stage.GetPrimAtPath("/RBY1/joints/fixed_base").SetActive(False)
    rspec = layer.GetPrimAtPath("/RBY1") or Sdf.CreatePrimInLayer(layer, "/RBY1")
    lo = Sdf.TokenListOp()
    lo.deletedItems = ["PhysicsArticulationRootAPI", "PhysxArticulationAPI"]
    rspec.SetInfo("apiSchemas", lo)
    bspec = Sdf.CreatePrimInLayer(layer, "/RBY1/base")
    lo = Sdf.TokenListOp()
    lo.prependedItems = ["PhysicsArticulationRootAPI", "PhysxArticulationAPI"]
    bspec.SetInfo("apiSchemas", lo)
    stage.GetPrimAtPath("/RBY1/base").CreateAttribute(
        "physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(False)

    # 2. materials --------------------------------------------------------------------------------
    def material(name, mu, combine):
        mp = f"/RBY1/PhysicsMaterials/{name}"
        mat = UsdShade.Material.Define(stage, mp)
        api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        api.CreateStaticFrictionAttr(mu)
        api.CreateDynamicFrictionAttr(mu)
        api.CreateRestitutionAttr(0.0)
        prim = mat.GetPrim()
        prim.AddAppliedSchema("PhysxMaterialAPI")
        prim.CreateAttribute("physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token).Set(combine)
        return mat

    wheel_mat = material("wheel", 1.2, "average")
    caster_mat = material("caster", 0.0, "min")
    for link, mat, key in (("wheel_l", wheel_mat, "wheel"), ("wheel_r", wheel_mat, "wheel"),
                           ("base", caster_mat, "caster")):
        col = stage.GetPrimAtPath(f"/RBY1/{link}/collisions")
        if col.IsInstanceable():
            col.SetInstanceable(False)
        for p in Usd.PrimRange(col):
            if p.HasAPI(UsdPhysics.CollisionAPI):
                UsdShade.MaterialBindingAPI.Apply(p).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")
                summary["materials"].setdefault(key, []).append(str(p.GetPath()))

    # 3. convex hull colliders for the links without any --------------------------------------------
    cache = UsdGeom.XformCache()
    for rel in HULL_LINKS:
        link = stage.GetPrimAtPath(f"/RBY1/{rel}")
        Lw = M(link, cache)
        Linv = np.linalg.inv(Lw)
        pts = []
        for p in Usd.PrimRange(link, PROXIES):
            if p.IsA(UsdGeom.Mesh) and "collision" not in str(p.GetPath()).lower():
                if p != link and p.HasAPI(UsdPhysics.RigidBodyAPI):
                    continue
                # skip descendants that are other rigid bodies (e.g. fingers under a gripper scope)
                anc, skip = p.GetParent(), False
                while anc and anc != link:
                    if anc.HasAPI(UsdPhysics.RigidBodyAPI):
                        skip = True
                        break
                    anc = anc.GetParent()
                if skip or UsdGeom.Imageable(p).ComputePurpose() == UsdGeom.Tokens.guide:
                    continue
                v = np.array(UsdGeom.Mesh(p).GetPointsAttr().Get() or [], dtype=float)
                if len(v) == 0:
                    continue
                Mw = M(p, cache)
                vh = np.c_[v, np.ones(len(v))] @ Mw @ Linv
                pts.append(vh[:, :3])
        if not pts:
            summary["hulls"][rel] = "no visual mesh"
            continue
        P = np.vstack(pts)
        hull = ConvexHull(P)
        hv = P[hull.vertices]
        remap = {int(i): k for k, i in enumerate(hull.vertices)}
        faces = [[remap[int(i)] for i in s] for s in hull.simplices]
        mesh = UsdGeom.Mesh.Define(stage, f"/RBY1/{rel}/added_collision")
        mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, x)) for x in hv]))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([i for f in faces for i in f]))
        mesh.CreatePurposeAttr(UsdGeom.Tokens.guide)
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
        ext = hv.max(axis=0) - hv.min(axis=0)
        summary["hulls"][rel] = {"vertices": int(len(hv)), "extent_m": [round(float(e), 4) for e in ext]}

    # 4. tool xform: put ee_{side} 0.1261 m below link_{side}_arm_6 (zero pose, frames aligned)
    cache = UsdGeom.XformCache()
    summary["tool"] = {}
    for side in ("left", "right"):
        a6 = M(stage.GetPrimAtPath(f"/RBY1/link_{side}_arm_6"), cache)
        ee = M(stage.GetPrimAtPath(f"/RBY1/{side}_gripper/ee_{side}"), cache)
        # row-vector convention: world point = local @ M -> offset along the arm_6 local -z
        want = (np.array([0.0, 0.0, -TOOL_OFFSET, 1.0]) @ a6)[:3]
        delta = want - ee[3, :3]
        grip = stage.GetPrimAtPath(f"/RBY1/{side}_gripper")
        xf = UsdGeom.Xformable(grip)
        ops = {op.GetOpName(): op for op in xf.GetOrderedXformOps()}
        if "xformOp:translate" in ops:
            op = ops["xformOp:translate"]
            cur = Gf.Vec3d(op.Get())
            parent_inv = np.linalg.inv(M(grip.GetParent(), cache))[:3, :3]
            d_local = delta @ parent_inv  # world delta -> parent frame (row vectors)
            op.Set(type(op.Get())(cur[0] + d_local[0], cur[1] + d_local[1], cur[2] + d_local[2]))
        else:
            xf.AddTranslateOp(opSuffix="dxbench_tool_fix").Set(Gf.Vec3d(*delta))
        summary["tool"][side] = {"ee_minus_arm6_before_m": float(ee[3, 2] - a6[3, 2]),
                                 "delta_m": [round(float(x), 5) for x in delta]}

    layer.Save()
    # verify
    st = Usd.Stage.Open(str(OUT))
    cache = UsdGeom.XformCache()
    for side in ("left", "right"):
        a6 = M(st.GetPrimAtPath(f"/RBY1/link_{side}_arm_6"), cache)
        ee = M(st.GetPrimAtPath(f"/RBY1/{side}_gripper/ee_{side}"), cache)
        summary["tool"][side]["ee_minus_arm6_after_m"] = float(ee[3, 2] - a6[3, 2])
    roots = [str(p.GetPath()) for p in st.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    dof = [p.GetName() for p in st.Traverse() if p.IsA(UsdPhysics.RevoluteJoint) or p.IsA(UsdPhysics.PrismaticJoint)]
    cols = sum(1 for p in st.Traverse(PROXIES) if p.HasAPI(UsdPhysics.CollisionAPI))
    summary.update({"articulation_roots": roots, "dof": len(dof), "colliders": cols})
    (HERE / "rby1_dx_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("articulation_roots", "dof", "colliders", "tool", "hulls")}, indent=1))


if __name__ == "__main__":
    main()
    app.close()
