"""Match the real towel (LUNOO 40 x 80 cm, 120 g, microfibre) to the video ~/Downloads/towel.mp4 (2026-10-21).

    ~/miniforge3/envs/dx_arena/bin/python tools/cloth_spike/towel_match.py --E 1e5 --mu 0.6 --tag e1e5_mu06

No robot. The benchmark towel (assets/objects.py: 0.8 x 0.4 m, 2.3 cm nodes, thickness 1.5 mm, density from the mass)
lies on a desk (top 0.725 m, same friction as the towel so the contact uses `--mu`). A virtual pinch (nodes within
2.5 cm of a point, velocity pins as the assisted grasp) replays the three motions of the video:
  corner : lift one corner 0.40 m while moving it 0.15 m towards the towel centre (video 3-7 s)
  center : pinch the centre, lift 0.20 m (video 25-28 s)
  drag   : pinch a corner 2 cm up, pull it 0.30 m along the long axis on the desk (video 12-14 s)
Each case starts from the flat towel, holds 2 s, saves a top-down render and metrics to reports/towel_match/<tag>/.
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--E", type=float, default=1.0e5, help="Young's modulus (Pa)")
ap.add_argument("--mu", type=float, default=0.9, help="static friction of towel and desk (dynamic = 0.85 x)")
ap.add_argument("--mass", type=float, default=0.120, help="towel mass (kg)")
ap.add_argument("--thickness", type=float, default=0.0015)
ap.add_argument("--bend_damping", type=float, default=0.1)
ap.add_argument("--hz", type=int, default=480)
ap.add_argument("--cases", default="corner,center,drag")
ap.add_argument("--video", action="store_true", help="record each case (top-down | oblique) at 15 fps")
ap.add_argument("--tag", default="base")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import json  # noqa: E402
import math  # noqa: E402
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

OUT = Path(os.path.expanduser(f"~/dx_bench/reports/towel_match/{args.tag}"))
OUT.mkdir(parents=True, exist_ok=True)
TOP = 0.725
SIZE = (0.80, 0.40)
DT = 1.0 / args.hz
LAVENDER = tuple(((c / 255 + 0.055) / 1.055) ** 2.4 for c in (172, 148, 175))


def T(x):
    if hasattr(x, "torch"):
        return x.torch
    return wp.to_torch(x) if isinstance(x, wp.array) else x


def build():
    density = args.mass / (SIZE[0] * SIZE[1] * args.thickness)
    mat = dict(density=density, static_friction=args.mu, dynamic_friction=0.85 * args.mu, youngs_modulus=args.E,
               poissons_ratio=0.25, surface_thickness=args.thickness, bend_damping=args.bend_damping,
               elasticity_damping=0.1)
    spawn = sim_utils.MeshRectangleCfg(
        size=SIZE, edge_refinement=max(1.0, math.hypot(*SIZE) / (1.45 * 0.023)),
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(solver_position_iteration_count=16, self_collision=True),
        physics_material=sim_utils.PhysxSurfaceDeformableBodyMaterialCfg(**mat),
        collision_props=[PhysxCollisionCfg(contact_offset=0.008, rest_offset=0.002)],
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=LAVENDER, roughness=0.9))

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
        light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2500.0))
        desk = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Desk",
            spawn=sim_utils.CuboidCfg(size=(1.8, 1.2, TOP), collision_props=sim_utils.CollisionPropertiesCfg(),
                                      visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.55, 0.5)),
                                      physics_material=sim_utils.RigidBodyMaterialCfg(
                                          static_friction=args.mu, dynamic_friction=0.85 * args.mu)),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, TOP / 2)))
        towel = DeformableObjectCfg(prim_path="{ENV_REGEX_NS}/Towel", spawn=spawn,
                                    init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, TOP + 0.005)))
        cam = CameraCfg(prim_path="{ENV_REGEX_NS}/cam", update_period=0.0, width=720, height=720, data_types=["rgb"],
                        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 5.0)),
                        renderer_cfg=IsaacRtxRendererCfg())
        top = CameraCfg(prim_path="{ENV_REGEX_NS}/top", update_period=0.0, width=720, height=720, data_types=["rgb"],
                        spawn=sim_utils.PinholeCameraCfg(focal_length=14.0, clipping_range=(0.05, 5.0)),
                        renderer_cfg=IsaacRtxRendererCfg())

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args.device, dt=DT, render_interval=32))
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=3.0))
    sim.reset()
    return sim, scene, density


def main():
    sim, scene, density = build()
    towel = scene["towel"]
    cam = scene["cam"]
    top = scene["top"]
    o = scene.env_origins[0]
    # top-down like the phone video (camera ~0.9 m above the desk, towel long axis vertical in the image)
    top.set_world_poses_from_view((o + torch.tensor([0.05, 0.0, TOP + 0.95], device=o.device))[None],
                                  (o + torch.tensor([0.0, 0.0, TOP], device=o.device))[None])
    frames = []
    every = max(1, round(args.hz / 15))
    # oblique view (the video is top-down, but the lifted drape only reads from the side)
    cam.set_world_poses_from_view((o + torch.tensor([0.75, -0.85, TOP + 0.65], device=o.device))[None],
                                  (o + torch.tensor([0.0, 0.0, TOP + 0.12], device=o.device))[None])

    def step(n=1, render=False):
        for i in range(n):
            sim.step(render=render and i == n - 1)
            scene.update(DT)

    def nodes():
        return T(towel.data.nodal_pos_w)[0] - o

    step(int(1.5 / DT))
    flat_state = T(towel.data.nodal_state_w).clone() if hasattr(towel.data, "nodal_state_w") else None
    n0 = nodes().clone()
    res = {"args": vars(args), "density": density, "cases": {}}

    def reset_flat():
        pos = T(towel.data.nodal_pos_w).clone()
        pos[0] = n0 + o
        towel.write_nodal_pos_to_sim_index(pos) if hasattr(towel, "write_nodal_pos_to_sim_index") else None
        vel = torch.zeros_like(T(towel.data.nodal_vel_w))
        towel.write_nodal_velocity_to_sim_index(vel)
        step(int(0.8 / DT))

    def pinch(p, r=0.025):
        n = nodes()
        d = (n[:, :2] - torch.tensor(p[:2], device=n.device)).norm(dim=-1)
        ids = torch.nonzero(d < r).flatten()
        return ids, n[ids].clone()

    def move(ids, start, path, secs, hold=2.0):
        """Pinned nodes follow start + path(s) (s 0..1, smoothstep) over `secs`, then hold."""
        steps = int(secs / DT)
        prev = None
        for k in range(steps + int(hold / DT)):
            s = min(1.0, (k + 1) / steps)
            s = s * s * (3 - 2 * s)
            tgt = start + torch.tensor(path(s), device=start.device, dtype=start.dtype)
            pos = T(towel.data.nodal_pos_w)
            vel = T(towel.data.nodal_vel_w).clone()
            v_ff = (tgt - prev) / DT if prev is not None else torch.zeros_like(tgt)
            prev = tgt.clone()
            vel[0, ids] = (tgt + o - pos[0, ids]) / DT + v_ff
            towel.write_nodal_velocity_to_sim_index(vel)
            rec = args.video and k % every == 0
            step(1, render=rec or (k == steps + int(hold / DT) - 1))
            if rec:
                a = T(top.data.output["rgb"])[0, ..., :3].cpu().numpy().astype(np.uint8)
                b = T(cam.data.output["rgb"])[0, ..., :3].cpu().numpy().astype(np.uint8)
                frames.append(np.concatenate([a, b], axis=1))

    def snap(name):
        img = T(cam.data.output["rgb"])[0, ..., :3].cpu().numpy().astype(np.uint8)
        Image.fromarray(img).save(OUT / f"{name}.png")

    x_lo, x_hi = float(n0[:, 0].min()), float(n0[:, 0].max())
    y_lo, y_hi = float(n0[:, 1].min()), float(n0[:, 1].max())
    for case in args.cases.split(","):
        reset_flat()
        if case == "corner":
            ids, start = pinch((x_hi - 0.01, y_lo + 0.01))
            move(ids, start, lambda s: (-0.15 * s, 0.05 * s, 0.40 * s), 2.5)
            n = nodes()
            on_desk = n[:, 2] < TOP + 0.012
            res["cases"][case] = {"pinned": int(len(ids)), "frac_on_desk": float(on_desk.float().mean()),
                                  "desk_extent_x": float(n[on_desk, 0].max() - n[on_desk, 0].min()) if on_desk.any() else 0.0}
        elif case == "center":
            ids, start = pinch(((x_lo + x_hi) / 2, (y_lo + y_hi) / 2))
            move(ids, start, lambda s: (0.0, 0.0, 0.20 * s), 2.0)
            n = nodes()
            on_desk = n[:, 2] < TOP + 0.012
            # the two flaps: how far apart their desk contact lines are (narrow = limp tent, wide = stiff A-frame)
            res["cases"][case] = {"pinned": int(len(ids)), "frac_on_desk": float(on_desk.float().mean()),
                                  "flap_span_x": float(n[on_desk, 0].max() - n[on_desk, 0].min()) if on_desk.any() else 0.0,
                                  "max_z": float(n[:, 2].max() - TOP)}
        elif case == "drag":
            ids, start = pinch((x_hi - 0.01, y_lo + 0.01))
            far = torch.argmin((n0[:, 0] - x_lo) ** 2 + (n0[:, 1] - y_hi) ** 2)
            f0 = nodes()[far].clone()
            move(ids, start, lambda s: (0.30 * s, 0.0, 0.02 + 0.0 * s), 2.0, hold=1.0)
            n = nodes()
            res["cases"][case] = {"pinned": int(len(ids)), "far_corner_dx": float(n[far, 0] - f0[0]),
                                  "slide_ratio": float((n[far, 0] - f0[0]) / 0.30),
                                  "extent_x": float(n[:, 0].max() - n[:, 0].min()),
                                  "max_z": float(n[:, 2].max() - TOP)}
        snap(case)
        if args.video and frames:
            import imageio

            imageio.mimsave(OUT / f"{case}.mp4", frames, fps=15, codec="libx264", quality=8, macro_block_size=1)
            frames.clear()
        print(f"[towel_match] {args.tag} {case}: {res['cases'][case]}", flush=True)
    (OUT / "result.json").write_text(json.dumps(res, indent=1))
    print(f"[towel_match] wrote {OUT}", flush=True)
    app.close()


if __name__ == "__main__":
    main()
