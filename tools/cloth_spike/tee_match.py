"""Match the real T-shirt (M(95) cotton 30s jersey, 140 g, wine) to the video ~/Downloads/cloth.mp4 (2026-10-22).

    ~/miniforge3/envs/dx_arena/bin/python tools/cloth_spike/tee_match.py --E 1e5 --tag e1e5

No robot. The benchmark tee (assets/objects/tshirt_m95_wine.usda, render shell + 2.5 mm rest offset) lies flat on a
desk (top 0.725 m). Virtual pinches (nodes within 2.5 cm, velocity pins like the assisted grasp) replay the video:
  lift  : one hem corner lifted 35 cm (video 1.5-2.5 s): how much stays on the desk, how wide the lifted cone is
  hang  : both shoulders lifted 70 cm, 40 cm apart (video 13-16 s, held up): hang length / width, does it hang flat
  fold  : the left side (sleeve + 12 cm of body) folded over onto the body and released (video 7-9 s): does it stay
Each case starts from the flat tee, saves an oblique render and metrics to reports/tee_match/<tag>/.
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--E", type=float, default=1.0e5, help="Young's modulus (Pa) at the 1.5 mm FEM thickness")
ap.add_argument("--hz", type=int, default=480)
ap.add_argument("--bend", type=float, default=None, help="omniphysics:surfaceBendStiffness override (D-46)")
ap.add_argument("--cases", default="lift,hang,fold")
ap.add_argument("--tag", default=None)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import json  # noqa: E402
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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "dxbench"))
from dxbench.assets.thick_cloth import split_single_mesh_cloth  # noqa: E402

TAG = args.tag or f"e{args.E:.0e}".replace("+0", "")
OUT = REPO / "reports" / "tee_match" / TAG
OUT.mkdir(parents=True, exist_ok=True)
TOP = 0.725
DT = 1.0 / args.hz
HALF = 0.0025


def T(x):
    if hasattr(x, "torch"):
        return x.torch
    return wp.to_torch(x) if isinstance(x, wp.array) else x


def build():
    @configclass
    class SceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(color=(0.5, 0.5, 0.5)))
        light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=1800.0))
        sun = AssetBaseCfg(prim_path="/World/sun", spawn=sim_utils.DistantLightCfg(intensity=1500.0, angle=2.0),
                           init_state=AssetBaseCfg.InitialStateCfg(rot=(0.25, 0.25, 0.0, 0.93)))
        desk = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Desk",
            spawn=sim_utils.CuboidCfg(size=(1.4, 1.0, TOP), collision_props=sim_utils.CollisionPropertiesCfg(),
                                      visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.62, 0.62, 0.6)),
                                      physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.6,
                                                                                      dynamic_friction=0.5)),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, TOP / 2)))
        tee = DeformableObjectCfg(
            prim_path="{ENV_REGEX_NS}/Tee",
            spawn=sim_utils.UsdFileCfg(usd_path=str(REPO / "dxbench/dxbench/assets/objects/tshirt_m95_wine.usda")),
            init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, TOP + 0.01)))
        cam = CameraCfg(prim_path="{ENV_REGEX_NS}/cam", update_period=0.0, width=1280, height=720, data_types=["rgb"],
                        spawn=sim_utils.PinholeCameraCfg(focal_length=16.0, clipping_range=(0.05, 8.0)),
                        renderer_cfg=IsaacRtxRendererCfg())

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args.device, dt=DT, render_interval=32))
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=4.0))
    from pxr import Usd

    stage = sim_utils.get_current_stage()
    root = stage.GetPrimAtPath("/World/envs/env_0/Tee")
    mesh = next(p for p in Usd.PrimRange(root) if p.HasAPI("OmniPhysicsSurfaceDeformableSimAPI"))
    split_single_mesh_cloth(stage, str(mesh.GetPath()), HALF)
    mesh.GetAttribute("physxCollision:restOffset").Set(HALF)
    for p in Usd.PrimRange(root):
        a = p.GetAttribute("omniphysics:youngsModulus")
        if a and a.IsValid():
            a.Set(float(args.E))
            if args.bend is not None:
                p.GetAttribute("omniphysics:surfaceBendStiffness").Set(float(args.bend))
    sim.reset()
    return sim, scene


def main():
    sim, scene = build()
    tee, cam = scene["tee"], scene["cam"]
    o = scene.env_origins[0]

    def step(n=1, render=False):
        for i in range(n):
            sim.step(render=render and i == n - 1)
            scene.update(DT)

    def nodes():
        return T(tee.data.nodal_pos_w)[0] - o

    def snap(name, eye=(0.75, -0.9, TOP + 0.75), tgt=(0.0, 0.0, TOP + 0.2)):
        cam.set_world_poses_from_view((o + torch.tensor(eye, device=o.device, dtype=torch.float32))[None],
                                      (o + torch.tensor(tgt, device=o.device, dtype=torch.float32))[None])
        for _ in range(3):
            step(1, render=True)
        Image.fromarray(T(cam.data.output["rgb"])[0, ..., :3].cpu().numpy().astype(np.uint8)).save(OUT / f"{name}.png")

    def pinch(xy, r=0.025, n0=None):
        n = nodes() if n0 is None else n0
        d = (n[:, :2] - torch.tensor(xy, device=n.device, dtype=n.dtype)).norm(dim=-1)
        top = n[:, 2] > n[:, 2].min() + 0.004  # the upper layer only (front of the shirt)
        return torch.nonzero((d < r) & top).flatten() if bool(((d < r) & top).any()) else torch.nonzero(d < r).flatten()

    def move(ids, start, path, secs, hold=2.0, release=False):
        steps = int(secs / DT)
        prev = None
        for k in range(steps + int(hold / DT)):
            s = min(1.0, (k + 1) / steps)
            s = s * s * (3 - 2 * s)
            tgt = start + torch.as_tensor(path(s), device=start.device, dtype=start.dtype)
            if k < steps or not release:
                pos = T(tee.data.nodal_pos_w)
                vel = T(tee.data.nodal_vel_w).clone()
                v_ff = (tgt - prev) / DT if prev is not None else torch.zeros_like(tgt)
                prev = tgt.clone()
                vel[0, ids] = (tgt + o - pos[0, ids]) / DT + v_ff
                tee.write_nodal_velocity_to_sim_index(vel)
            step(1)

    step(int(1.5 / DT))
    n0 = nodes().clone()
    x_lo, x_hi = float(n0[:, 0].min()), float(n0[:, 0].max())
    y_lo, y_hi = float(n0[:, 1].min()), float(n0[:, 1].max())
    res = {"args": vars(args), "cases": {}}

    def reset_flat():
        pos = T(tee.data.nodal_pos_w).clone()
        pos[0] = n0 + o
        tee.write_nodal_pos_to_sim_index(pos)
        tee.write_nodal_velocity_to_sim_index(torch.zeros_like(T(tee.data.nodal_vel_w)))
        step(int(1.0 / DT))

    for case in args.cases.split(","):
        reset_flat()
        if case == "lift":
            ids = pinch((x_lo + 0.13, y_lo + 0.01), n0=n0)
            move(ids, n0[ids].clone(), lambda s: (0.0, 0.10 * s, 0.35 * s), 1.5)
            n = nodes()
            on = n[:, 2] < TOP + 0.012
            res["cases"][case] = {"frac_on_desk": float(on.float().mean()),
                                  "desk_extent_xy": (n[on, :2].max(0).values - n[on, :2].min(0).values).tolist()}
            snap(case)
        elif case == "hang":
            ids_l = pinch((-0.20, y_hi - 0.04), n0=n0)
            ids_r = pinch((0.20, y_hi - 0.04), n0=n0)
            ids = torch.cat([ids_l, ids_r])
            move(ids, n0[ids].clone(), lambda s: (0.0, 0.0, 0.70 * s), 2.0, hold=2.0)
            n = nodes()
            res["cases"][case] = {"hang_length": float(n[:, 2].max() - n[:, 2].min()),
                                  "hang_width_x": float(n[:, 0].max() - n[:, 0].min()),
                                  "hang_depth_y": float(n[:, 1].max() - n[:, 1].min()),
                                  "touches_desk": bool((n[:, 2] < TOP + 0.01).any())}
            snap(case, eye=(0.2, -1.5, TOP + 0.6), tgt=(0.0, 0.0, TOP + 0.45))
        elif case == "fold":
            line = x_lo + 0.25  # sleeve + ~12 cm of body
            side = n0[:, 0] < line
            edge = pinch((x_lo + 0.03, 0.0), r=0.30, n0=n0)
            edge = edge[n0[edge, 0] < x_lo + 0.05]
            move(edge, n0[edge].clone(),
                 lambda s: (2 * (line - (x_lo + 0.03)) * s, 0.0, 0.10 * np.sin(np.pi * s)), 2.0, hold=2.0, release=True)
            n = nodes()
            folded = side & (n[:, 0] > line - 0.02)  # the side that is still lying over the body
            res["cases"][case] = {"frac_side_still_folded": float(folded.sum() / max(1, int(side.sum()))),
                                  "max_z_above_desk": float(n[:, 2].max() - TOP)}
            snap(case)
        print(f"[tee_match] {TAG} {case}: {res['cases'][case]}", flush=True)
    (OUT / "result.json").write_text(json.dumps(res, indent=1))
    app.close()


if __name__ == "__main__":
    main()
