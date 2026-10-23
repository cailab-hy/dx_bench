"""How crinkly is a crumpled cloth? (2026-10-23, user: sim towel / tee / duvet crumple far more finely than the real ones)

    ~/miniforge3/envs/dx_arena/bin/python tools/cloth_spike/crumple_test.py --cloth tee --tag tee_base

No robot. The benchmark cloth is crumpled the way make_cloth_states.py does it, then left to settle:
  towel / tee : flat on the floor, one random point pulled up 45 cm (drifting <= 8 cm) over 1 s and released
                (the BED-03 medium / hard laundry heap); --seed picks the point
  duvet       : flat on a lab-bed box (0.98 x 1.88 m, top 0.53 m), three corners carried in 40 cm arcs onto one spot
                near the mattress middle over 2.5 s (BED-01 medium)
Metrics on the simulation mesh after a 3 s settle: dihedral angle between neighbouring triangles (interior edges):
mean, p90, fraction of edges sharper than 30 / 60 deg (a 2.3 cm-node sheet folded with a 5 cm radius bends ~26 deg per
edge, crumpled paper > 60); heap footprint and height. Renders an oblique and a top view to reports/crumple/<tag>/.
Knobs: --E, --thick (FEM surface thickness; density follows so the mass stays), --bend / --stretch / --shear
(omniphysics:surface*Stiffness, 0 = the solver default), --ld linear damping, --bd bend damping.
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--cloth", default="tee", choices=["towel", "tee", "duvet"])
ap.add_argument("--E", type=float, default=None, help="Young's modulus (Pa); default = the benchmark value")
ap.add_argument("--thick", type=float, default=None, help="FEM surface thickness (m); mass kept")
ap.add_argument("--bend", type=float, default=None, help="omniphysics:surfaceBendStiffness")
ap.add_argument("--stretch", type=float, default=None, help="omniphysics:surfaceStretchStiffness")
ap.add_argument("--shear", type=float, default=None, help="omniphysics:surfaceShearStiffness")
ap.add_argument("--nu", type=float, default=None, help="Poisson's ratio")
ap.add_argument("--ld", type=float, default=3.0, help="linear damping (1/s), D-45")
ap.add_argument("--bd", type=float, default=0.1, help="bend damping")
ap.add_argument("--grid", type=float, default=0.023, help="towel / duvet node spacing (m)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--rest", default=None, help="omniphysics:restBendAnglesDefault (flatDefault / restShapeDefault)")
ap.add_argument("--cantilever", action="store_true",
                help="towel only: lie on the bed box with --overhang hanging over its edge (Peirce bending length test)")
ap.add_argument("--overhang", type=float, default=0.08)
ap.add_argument("--scheme", default=None, help="render mesh subdivisionScheme override (catmullClark / loop / none)")
ap.add_argument("--refine", type=int, default=None, help="render mesh refinementLevel (with --scheme)")
ap.add_argument("--lift", type=float, default=0.45)
ap.add_argument("--settle", type=float, default=3.0)
ap.add_argument("--hz", type=int, default=480)
ap.add_argument("--tag", default=None)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402
from PIL import Image  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg, DeformableObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab_physx.renderers import IsaacRtxRendererCfg  # noqa: E402
from isaaclab_physx.sim.schemas.schemas_cfg import PhysxCollisionCfg  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "dxbench"))
from dxbench.assets.thick_cloth import grid_rectangle_spawn, split_single_mesh_cloth, thicken_render_mesh  # noqa: E402

TAG = args.tag or f"{args.cloth}_s{args.seed}"
OUT = REPO / "reports" / "crumple" / TAG
OUT.mkdir(parents=True, exist_ok=True)
DT = 1.0 / args.hz
HALF = 0.0025
BED = (0.98, 1.88, 0.53)  # lab bed mattress box
# benchmark values (objects.py): size, mass, E, FEM thickness, nu, friction, colour
SPEC = {"towel": ((0.8, 0.4), 0.120, 1.0e5, 0.0015, 0.25, (0.9, 0.75), (0.55, 0.42, 0.62)),
        "duvet": ((2.0, 1.55), 1.2, 1.0e5, 0.002, 0.3, (0.6, 0.5), (0.8879, 0.8879, 0.8714))}
HL = SPEC[args.cloth][0][0] / 2 if args.cloth in SPEC else 0.0  # cantilever: half length along x (spawned unrotated)


def T(x):
    if hasattr(x, "torch"):
        return x.torch
    return wp.to_torch(x) if isinstance(x, wp.array) else x


def cloth_cfg():
    props = sim_utils.DeformableBodyPropertiesCfg(solver_position_iteration_count=16, self_collision=True,
                                                  linear_damping=args.ld)
    if args.cloth == "tee":
        return sim_utils.UsdFileCfg(usd_path=str(REPO / "dxbench/dxbench/assets/objects/tshirt_m95_wine.usda"))
    size, mass, E, th, nu, mu, rgb = SPEC[args.cloth]
    th_sim = args.thick or th
    mat = dict(density=mass / (size[0] * size[1] * th_sim), static_friction=mu[0], dynamic_friction=mu[1],
               youngs_modulus=args.E or E, poissons_ratio=args.nu or nu, surface_thickness=th_sim,
               bend_damping=args.bd, elasticity_damping=0.1)
    for k, v in (("surface_bend_stiffness", args.bend), ("surface_stretch_stiffness", args.stretch),
                 ("surface_shear_stiffness", args.shear)):
        if v is not None:
            mat[k] = v
    cfg = sim_utils.MeshRectangleCfg(
        size=size, edge_refinement=1.0, deformable_props=props,
        physics_material=sim_utils.PhysxSurfaceDeformableBodyMaterialCfg(**mat),
        collision_props=[PhysxCollisionCfg(contact_offset=0.008, rest_offset=HALF)],
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb))
    cfg.func = grid_rectangle_spawn(args.grid)
    return cfg


def build():
    yaw = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)) if args.cloth == "duvet" and not args.cantilever \
        else (0.0, 0.0, 0.0, 1.0)
    box = (2 * HL + 0.2, 2.0, BED[2]) if args.cantilever else BED
    z0 = (BED[2] if args.cloth == "duvet" or args.cantilever else 0.0) + 0.02

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(
            color=(0.62, 0.6, 0.58),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.6, dynamic_friction=0.5)))
        light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=1500.0))
        sun = AssetBaseCfg(prim_path="/World/sun", spawn=sim_utils.DistantLightCfg(intensity=1800.0, angle=2.0),
                           init_state=AssetBaseCfg.InitialStateCfg(rot=(0.25, 0.25, 0.0, 0.93)))
        bed = (AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Bed",
            spawn=sim_utils.CuboidCfg(size=box, collision_props=sim_utils.CollisionPropertiesCfg(),
                                      visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.5, 0.45)),
                                      physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.6,
                                                                                      dynamic_friction=0.5)),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(HL - args.overhang - box[0] / 2 if args.cantilever else 0.0, 0.0, BED[2] / 2)))
            if args.cloth == "duvet" or args.cantilever else None)
        cloth = DeformableObjectCfg(prim_path="{ENV_REGEX_NS}/Cloth", spawn=cloth_cfg(),
                                    init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, z0), rot=yaw))
        cam = CameraCfg(prim_path="{ENV_REGEX_NS}/cam", update_period=0.0, width=1280, height=720, data_types=["rgb"],
                        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.05, 10.0)),
                        renderer_cfg=IsaacRtxRendererCfg())

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args.device, dt=DT, render_interval=32))
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=5.0))
    from pxr import Sdf, Usd, UsdGeom

    stage = sim_utils.get_current_stage()
    root = stage.GetPrimAtPath("/World/envs/env_0/Cloth")
    mesh = next(p for p in Usd.PrimRange(root) if p.HasAPI("OmniPhysicsSurfaceDeformableSimAPI"))
    if args.cloth == "tee":  # the tee USD carries its own material: override on its prims
        split_single_mesh_cloth(stage, str(mesh.GetPath()), HALF)
        mesh.GetAttribute("physxCollision:restOffset").Set(HALF)
        mesh.GetAttribute("physxDeformableBody:linearDamping").Set(float(args.ld))
        for p in Usd.PrimRange(root):
            th = p.GetAttribute("omniphysics:surfaceThickness")
            if not (th and th.IsValid() and th.HasAuthoredValue()):
                continue
            if args.thick:
                d = p.GetAttribute("omniphysics:density")
                d.Set(float(d.Get() * th.Get() / args.thick))
                th.Set(float(args.thick))
            for name, v in (("omniphysics:youngsModulus", args.E), ("omniphysics:poissonsRatio", args.nu),
                            ("omniphysics:surfaceBendStiffness", args.bend),
                            ("omniphysics:surfaceStretchStiffness", args.stretch),
                            ("omniphysics:surfaceShearStiffness", args.shear),
                            ("physxDeformableMaterial:bendDamping", args.bd)):
                if v is not None:
                    a = p.GetAttribute(name)
                    if not (a and a.IsValid()):
                        a = p.CreateAttribute(name, Sdf.ValueTypeNames.Float)
                    a.Set(float(v))
    else:
        thicken_render_mesh(stage, str(root.GetPath()), HALF)
    if args.rest:
        mesh.CreateAttribute("omniphysics:restBendAnglesDefault", Sdf.ValueTypeNames.Token, variability=Sdf.VariabilityUniform).Set(args.rest)
    if args.scheme:
        for p in Usd.PrimRange(root):
            if p.GetTypeName() == "Mesh" and not p.HasAPI("OmniPhysicsSurfaceDeformableSimAPI"):
                UsdGeom.Mesh(p).GetSubdivisionSchemeAttr().Set(args.scheme)
                if args.refine is not None:
                    p.CreateAttribute("refinementEnableOverride", Sdf.ValueTypeNames.Bool).Set(True)
                    p.CreateAttribute("refinementLevel", Sdf.ValueTypeNames.Int).Set(args.refine)
    if os.environ.get("CRUMPLE_DUMP"):
        from pxr import UsdShade
        for p in Usd.PrimRange(root):
            rel = p.GetRelationship("material:binding:physics")
            print(f"[dump] {p.GetPath()} {p.GetTypeName()} apis={p.GetAppliedSchemas()} physbind={rel.GetTargets() if rel else None}")
            for a in p.GetAttributes():
                if a.HasAuthoredValue() and ("omniphysics" in a.GetName() or "physx" in a.GetName()) and "Points" not in a.GetName() and "points" not in a.GetName():
                    print(f"[dump]    {a.GetName()} = {a.Get()}")
    faces = np.array(UsdGeom.Mesh(mesh).GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    sim.reset()
    return sim, scene, faces


def dihedral(P, F):
    """Angle (deg) between the normals of every pair of triangles sharing an edge."""
    n = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    e = {}
    for fi, f in enumerate(F):
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            e.setdefault((min(a, b), max(a, b)), []).append(fi)
    pairs = np.array([v for v in e.values() if len(v) == 2])
    c = np.clip((n[pairs[:, 0]] * n[pairs[:, 1]]).sum(1), -1, 1)
    return np.degrees(np.arccos(c))


def main():
    sim, scene, F = build()
    cl, cam = scene["cloth"], scene["cam"]
    o = scene.env_origins[0]
    rng = np.random.default_rng(args.seed)

    def step(n=1, render=False):
        for i in range(n):
            sim.step(render=render and i == n - 1)
            scene.update(DT)

    def nodes():
        return T(cl.data.nodal_pos_w)[0] - o

    def snap(name, eye, tgt):
        cam.set_world_poses_from_view((o + torch.tensor(eye, device=o.device, dtype=torch.float32))[None],
                                      (o + torch.tensor(tgt, device=o.device, dtype=torch.float32))[None])
        for _ in range(3):
            step(1, render=True)
        Image.fromarray(T(cam.data.output["rgb"])[0, ..., :3].cpu().numpy().astype(np.uint8)).save(OUT / f"{name}.png")

    n0 = nodes().clone() if args.cantilever else None  # spawn: flat, x in [-0.4, 0.4]
    step(int(1.5 / DT))
    n0 = n0 if args.cantilever else nodes().clone()
    flat = {"flat_zspan": float(nodes()[:, 2].max() - nodes()[:, 2].min()),
            "flat_speed_max": float(T(cl.data.nodal_vel_w)[0].norm(dim=-1).max())}
    print(f"[crumple] {TAG} flat: {json.dumps(flat)}", flush=True)
    P0 = n0.cpu().numpy()
    if len(P0) != F.max() + 1:
        print(f"[crumple] node count {len(P0)} != mesh points {F.max() + 1}: dihedral skipped", flush=True)
    d_flat = dihedral(P0, F) if len(P0) == F.max() + 1 else None
    drops = []
    if args.cantilever:  # Peirce: overhang l droops to tip angle theta; c = l (cos(theta / 2) / (8 tan theta))^(1/3)
        P = nodes().cpu().numpy()
        edge = HL - args.overhang
        tip = P[n0.cpu().numpy()[:, 0] > HL - 0.012]  # the free end
        l = args.overhang
        dz = float(BED[2] + HALF - tip[:, 2].mean())
        theta = math.atan2(dz, float(tip[:, 0].mean() - edge))
        c = l * (math.cos(theta / 2) / (8 * math.tan(theta))) ** (1 / 3) if theta > 1e-3 else float("inf")
        res = {"args": vars(args), "tip_drop": dz, "theta_deg": math.degrees(theta), "bending_length_c": c}
        snap("cantilever", (0.35, -0.6, 0.55), (0.35, 0.0, 0.5))
        (OUT / "result.json").write_text(json.dumps(res, indent=1))
        print(f"[crumple] {TAG}: {json.dumps({k: v for k, v in res.items() if k != 'args'})}", flush=True)
        app.close()
        return
    if args.cloth == "duvet":
        c0 = n0.mean(0)
        g = torch.tensor([rng.uniform(-0.15, 0.15), rng.uniform(-0.3, 0.1)], device=o.device, dtype=n0.dtype)
        d = n0[:, :2] - c0[:2]
        corners = [int(torch.argmax(d @ torch.tensor(v, device=o.device, dtype=n0.dtype)))
                   for v in ((1, 1), (1, -1), (-1, 1), (-1, -1))]
        for k in [corners[i] for i in rng.permutation(4)[:3]]:
            ids = torch.nonzero((n0 - n0[k]).norm(dim=-1) < 0.04).flatten()
            pull = (g - n0[k, :2]) * float(rng.uniform(0.75, 0.95))

            def gather(p0, s_, pull=pull):
                t = p0.clone()
                t[:, :2] += pull * s_
                t[:, 2] += 0.4 * math.sin(math.pi * s_)
                return t
            drops.append((ids, n0[ids].clone(), gather, 2.5))
    else:
        k = int(rng.integers(len(n0)))
        drift = torch.as_tensor(rng.normal(0, 0.08, 2), device=o.device, dtype=n0.dtype)
        ids = torch.nonzero((n0 - n0[k]).norm(dim=-1) < 0.03).flatten()

        def lift_path(p0, s_):
            t = p0.clone()
            t[:, :2] += drift * s_
            t[:, 2] += args.lift * s_
            return t
        drops.append((ids, n0[ids].clone(), lift_path, 1.0))
    t_max = max(dd[3] for dd in drops)
    for i in range(int((t_max + 0.5) / DT)):
        pos = T(cl.data.nodal_pos_w)
        vel = T(cl.data.nodal_vel_w).clone()
        for ids, p0, fn, secs in drops:
            n_d = int(secs / DT)
            if i >= n_d:
                continue
            s_ = min(1.0, (i + 1) / (0.85 * n_d))
            s_ = s_ * s_ * (3 - 2 * s_)
            vel[0, ids] = (fn(p0, s_) + o - pos[0, ids]) / DT
        cl.write_nodal_velocity_to_sim_index(vel)
        step(1)
    step(int(args.settle / DT))
    P = nodes().cpu().numpy()
    res = {"args": vars(args)}
    if d_flat is not None:
        d = dihedral(P, F)
        res.update({"dihedral_mean": float(d.mean()), "dihedral_p90": float(np.percentile(d, 90)),
                    "frac_gt30": float((d > 30).mean()), "frac_gt60": float((d > 60).mean()),
                    "flat_dihedral_mean": float(d_flat.mean())})
    base = BED[2] if args.cloth == "duvet" else 0.0
    res.update({"footprint_xy": (P[:, :2].max(0) - P[:, :2].min(0)).tolist(), "height": float(P[:, 2].max() - base),
                "speed_end": float(T(cl.data.nodal_vel_w)[0].norm(dim=-1).mean())})
    c = P.mean(0)
    span = float(max(np.ptp(P[:, 0]), np.ptp(P[:, 1])))
    tz = float(P[:, 2].max()) * 0.5 + base * 0.5
    snap("oblique", (c[0] + 0.5 * span + 0.1, c[1] - 0.5 * span - 0.1, tz + 0.45 * span + 0.15), (c[0], c[1], tz))
    snap("top", (c[0] + 0.01, c[1], tz + 0.9 * span + 0.2), (c[0], c[1], tz))
    (OUT / "result.json").write_text(json.dumps(res, indent=1))
    np.save(OUT / "nodes.npy", P)
    print(f"[crumple] {TAG}: {json.dumps({k: v for k, v in res.items() if k != 'args'})}", flush=True)
    app.close()


if __name__ == "__main__":
    main()
