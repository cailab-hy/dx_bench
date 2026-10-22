"""Visual thickness for surface-deformable cloth (2026-10-22, user: towel / duvet / T-shirt are ~5 mm thick).

PhysX simulates a surface deformable on a zero-thickness triangle sheet (the sim mesh). Its render mesh is a separate
prim bound to the sim mesh through the `bindPose` deformable pose: PhysX skins every render vertex onto the sim
triangles, including an offset along the triangle normal. So the cloth gets its look of thickness from a render
*shell* - the sheet offset by +-h along the vertex normals plus a rim along the open edges - while the simulation
sheet stays thin. Collision thickness is the matching rest offset h on the sim mesh (cloth surfaces then rest 2h apart).

`shell_mesh` builds the shell from a sheet; `thicken_render_mesh` rewrites a cloth's render mesh on the stage before
physics starts (spawn time); `split_single_mesh_cloth` turns a cloth whose one Mesh is both sim and render mesh
(the DX T-shirt) into a sim mesh + shell render mesh under an Xform root.
"""
from __future__ import annotations

import numpy as np


def vertex_normals(P: np.ndarray, F: np.ndarray) -> np.ndarray:
    fn = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])  # area weighted
    n = np.zeros_like(P)
    for k in range(3):
        np.add.at(n, F[:, k], fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(ln, 1e-12)


def boundary_edges(F: np.ndarray) -> np.ndarray:
    """Directed edges (a, b) used by exactly one triangle, in that triangle's winding."""
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    key = np.sort(e, axis=1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    return e[cnt[inv.ravel()] == 1]


def shell_mesh(P: np.ndarray, F: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
    """Closed shell of half thickness h around the sheet (P, F): top = P + h n, bottom = P - h n (reversed winding),
    and a rim of two triangles per open edge. Returns (points 2V x 3, triangles)."""
    P = np.asarray(P, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64).reshape(-1, 3)
    n = vertex_normals(P, F)
    V = len(P)
    pts = np.concatenate([P + h * n, P - h * n])
    tris = [F, F[:, ::-1] + V]
    b = boundary_edges(F)
    if len(b):
        a_, b_ = b[:, 0], b[:, 1]
        # the rim faces outwards: top edge runs a -> b, so the outward quad is (b, a, a', b')
        tris += [np.stack([b_, a_, a_ + V], axis=1), np.stack([b_, a_ + V, b_ + V], axis=1)]
    return pts.astype(np.float32), np.concatenate(tris).astype(np.int32)


def _set_mesh(prim, pts, tris):
    from pxr import Sdf, UsdGeom, Vt

    m = UsdGeom.Mesh(prim)
    m.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(pts))
    m.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(tris.reshape(-1)))
    m.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(len(tris), 3, dtype=np.int32)))
    if m.GetNormalsAttr().HasAuthoredValue():
        m.GetNormalsAttr().Clear()
    for name in ("primvars:normals", "primvars:st", "primvars:st:indices", "primvars:normals:indices"):
        if prim.HasAttribute(name):
            prim.RemoveProperty(name)
    m.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    m.CreateDoubleSidedAttr().Set(False)
    m.GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(np.stack([pts.min(0), pts.max(0)]).astype(np.float32)))
    prim.CreateAttribute("deformablePose:default:omniphysics:points", Sdf.ValueTypeNames.Point3fArray).Set(
        Vt.Vec3fArray.FromNumpy(pts))


def thicken_render_mesh(stage, root_path: str, half_thickness: float) -> bool:
    """Replace the render mesh of the deformable at `root_path` (an Isaac Lab spawned cloth: root Xform with a guide
    sim mesh and one render Mesh carrying the bindPose) by a shell. Returns False when no such render mesh exists."""
    from pxr import Usd, UsdGeom

    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return False
    for prim in Usd.PrimRange(root):
        if prim.GetTypeName() != "Mesh" or prim.HasAPI("OmniPhysicsSurfaceDeformableSimAPI"):
            continue
        if UsdGeom.Imageable(prim).GetPurposeAttr().Get() == UsdGeom.Tokens.guide:
            continue
        bind = prim.GetAttribute("deformablePose:default:omniphysics:points")
        P = np.array(bind.Get() if bind and bind.HasAuthoredValue() else UsdGeom.Mesh(prim).GetPointsAttr().Get())
        F = np.array(UsdGeom.Mesh(prim).GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        pts, tris = shell_mesh(P, F, half_thickness)
        _set_mesh(prim, pts, tris)
        return True
    return False


def split_single_mesh_cloth(stage, mesh_path: str, half_thickness: float):
    """A cloth whose single Mesh is both the simulation and the render mesh (the DX T-shirt: OmniPhysicsDeformableBodyAPI
    + OmniPhysicsSurfaceDeformableSimAPI on one Mesh): add a `render` shell Mesh child bound by bindPose, make the
    original mesh a guide (simulation / collision only). The body API stays on the original prim, which keeps the
    simulation mesh, so prim paths and the deformable view are unchanged. Returns the render prim."""
    from pxr import Sdf, UsdGeom, UsdShade

    sim = stage.GetPrimAtPath(mesh_path)
    m = UsdGeom.Mesh(sim)
    P = np.array(m.GetPointsAttr().Get())
    F = np.array(m.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    pts, tris = shell_mesh(P, F, half_thickness)
    vis = UsdGeom.Mesh.Define(stage, sim.GetPath().AppendChild("render")).GetPrim()
    vis.ApplyAPI("OmniPhysicsDeformablePoseAPI", "default")
    vis.CreateAttribute("deformablePose:default:omniphysics:purposes", Sdf.ValueTypeNames.TokenArray).Set(["bindPose"])
    _set_mesh(vis, pts, tris)
    UsdGeom.Imageable(vis).CreatePurposeAttr().Set(UsdGeom.Tokens.render)  # else it inherits the parent's guide
    look = UsdShade.MaterialBindingAPI(sim).GetDirectBinding().GetMaterial()
    if look:
        UsdShade.MaterialBindingAPI.Apply(vis).Bind(look)
    if not sim.HasAPI("OmniPhysicsDeformablePoseAPI", "default"):
        sim.ApplyAPI("OmniPhysicsDeformablePoseAPI", "default")
        sim.CreateAttribute("deformablePose:default:omniphysics:purposes", Sdf.ValueTypeNames.TokenArray).Set(
            ["bindPose"])
        sim.CreateAttribute("deformablePose:default:omniphysics:points", Sdf.ValueTypeNames.Point3fArray).Set(
            m.GetPointsAttr().Get())
    UsdGeom.Imageable(sim).CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
    return vis


def thick_spawn(spawn_fn, half_thickness: float, single_mesh: bool = False):
    """Wrap an Isaac Lab spawner (`cfg.func`): spawn as usual, then give every spawned cloth (all prims matching the
    possibly regex prim path) its shell render mesh. single_mesh: the cloth's one Mesh is sim + render mesh (DX
    T-shirt USD) - split it (`split_single_mesh_cloth`) and set the collision rest offset to the half thickness."""

    def spawn(prim_path, cfg, translation=None, orientation=None, **kwargs):
        import isaaclab.sim as sim_utils
        from pxr import Usd

        prim = spawn_fn(prim_path, cfg, translation, orientation, **kwargs)
        stage = sim_utils.get_current_stage()
        for path in sim_utils.find_matching_prim_paths(prim_path):
            if single_mesh:
                root = stage.GetPrimAtPath(path)
                mesh = next(p for p in Usd.PrimRange(root) if p.HasAPI("OmniPhysicsSurfaceDeformableSimAPI"))
                split_single_mesh_cloth(stage, str(mesh.GetPath()), half_thickness)
                mesh.GetAttribute("physxCollision:restOffset").Set(half_thickness)
            elif not thicken_render_mesh(stage, path, half_thickness):
                raise RuntimeError(f"no cloth render mesh under {path}")
        return prim

    return spawn


def grid_rectangle_spawn(spacing: float):
    """A `MeshRectangleCfg` spawner with a regular grid of about `spacing` (m) between nodes (alternating diagonals).
    Isaac Lab refines the 2-triangle rectangle by halving edges, so a 2.0 x 1.55 m duvet can only get 65 x 65 nodes
    (3.1 x 2.4 cm - the fold crumpled it in the task) or 129 x 129 (16.6 k nodes); this gives 88 x 68 at 2.3 cm.
    Use with edge_refinement = 1.0 (no further subdivision)."""
    from isaaclab.sim.utils import clone

    @clone
    def spawn(prim_path, cfg, translation=None, orientation=None, **kwargs):
        import trimesh
        from isaaclab.sim.spawners.meshes.meshes import _spawn_mesh_geom_from_mesh
        from isaaclab.sim.utils import get_current_stage

        sx, sy = cfg.size
        nx, ny = max(2, round(sx / spacing) + 1), max(2, round(sy / spacing) + 1)
        xs, ys = np.linspace(-sx / 2, sx / 2, nx), np.linspace(-sy / 2, sy / 2, ny)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        verts = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], axis=1).astype(np.float32)
        tris = []
        for i in range(nx - 1):
            for j in range(ny - 1):
                a, b, c, d = i * ny + j, (i + 1) * ny + j, (i + 1) * ny + j + 1, i * ny + j + 1
                tris += [(a, b, c), (a, c, d)] if (i + j) % 2 == 0 else [(a, b, d), (b, c, d)]
        mesh = trimesh.Trimesh(vertices=verts, faces=np.array(tris), process=False)
        stage = get_current_stage()
        _spawn_mesh_geom_from_mesh(prim_path, cfg, mesh, translation, orientation, None, stage=stage)
        return stage.GetPrimAtPath(prim_path)

    return spawn
