"""Reference solutions of BED-01 / BED-02 as videos, no robot (2026-10-26, user: show the correct process; the bed and
the clothes may tidy themselves). The task env is built from an evaluation scene with its saved cloth state; invisible
"hands" move cloth nodes by velocity pins (the same mechanism as the assisted grasp and the state generator) while the
robot stands still. The observer camera is recorded at 15 Hz with the step name and the live task predicates.

    ~/miniforge3/envs/dx_arena/bin/python tools/solution_demo.py --demo_scene benchmarks/dx_bed_v0/scenes/bed01_easy_000.json \
        --demo_out outputs/solution/bed01_easy_000

BED-01 easy : pillow off the bed -> the folded half of the duvet back over the head side -> pillow back on the head end
BED-01 medium / hard : pillow off -> the whole duvet carried back to its spread shape -> pillow back
BED-02 easy (towel) / medium (T-shirt) : (crumpled: flatten) -> fold left -> fold right -> fold in half -> onto the stack
Writes <out>/observer.mp4 (annotated) and <out>/result.json (predicates after every step).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dxbench"))
from dxbench.utils.arena_runtime import ArenaRuntime  # noqa: E402

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"  # face 1 = Korean
STAGE_KO = {"pillow_cleared": "베개 치움", "duvet_spread": "이불 펴짐", "duvet_even": "좌우 고름",
            "pillow_placed": "베개 제자리", "flatten": "펴기", "fold_left": "왼쪽 접기", "fold_right": "오른쪽 접기",
            "fold_half": "반 접기", "on_stack": "쌓기"}


def _args(p):
    p.add_argument("--demo_scene", required=True, help="evaluation scene JSON")
    p.add_argument("--demo_out", required=True)
    p.add_argument("--eye", type=float, nargs=3, default=None, help="observer eye offset from the target (m)")
    p.add_argument("--target", type=float, nargs=3, default=None, help="observer target (env frame, m)")


def main():
    pre = argparse.ArgumentParser(add_help=False)
    _args(pre)
    a0, _ = pre.parse_known_args()
    from dxbench.tasks.scene_spec import SceneSpec

    spec = SceneSpec.from_json(a0.demo_scene)
    env_name = {"BED-01": "dx_bed01", "BED-02": "dx_bed02"}[spec.task]
    rest = [x for x in sys.argv[1:]]
    sys.argv = [sys.argv[0], "--enable_cameras", *rest, "--external_environment_class_path",
                f"dxbench.tasks.bed_tasks:DxBed0{spec.task[-1]}Environment", env_name, "--scene", a0.demo_scene,
                "--no_assist", "--embodiment", "rby1_joint", "--observer", "--observer_res", "1280", "720"]
    rt = ArenaRuntime(add_args=_args, argv=None)
    env = rt.make_env()
    a = rt.args
    U = env.unwrapped
    env.reset()

    import imageio
    import torch
    from PIL import Image, ImageDraw, ImageFont

    from dxbench.embodiments.rby1_cfg import ARM_L, ARM_R, READY_ARMS
    from dxbench.tasks import bed_tasks
    from dxbench.tasks.bed_tasks import aim_observer

    task = bed_tasks.CURRENT_TASK
    org = U.scene.env_origins[0]
    dev = U.device
    dt = U.physics_dt
    out = a.demo_out
    os.makedirs(out, exist_ok=True)
    act = torch.zeros(U.num_envs, 24, device=dev)
    act[:, :14] = torch.tensor([[READY_ARMS.get(n, 0.0) for n in ARM_R + ARM_L]], device=dev)
    act[:, 14:16] = 1.0
    font = ImageFont.truetype(FONT, 30, index=1)
    font_s = ImageFont.truetype(FONT, 22, index=1)
    frames, log = [], []
    caption = {"t": ""}
    eval_state = {"r": None}

    def nodes(name):
        return U.scene[name].data.nodal_pos_w.torch[0] - org

    def evaluate():
        r = task.evaluate(U)[0]
        U.episode_length_buf[:] = 10  # keep the task's "episode start" bookkeeping (pillow start, flat start)
        eval_state["r"] = r
        return r

    def frame():
        img = U.scene["observer"].data.output["rgb"].torch[0, ..., :3].cpu().numpy().astype(np.uint8)
        im = Image.fromarray(img)
        d = ImageDraw.Draw(im, "RGBA")
        d.rectangle([0, 0, im.width, 54], fill=(0, 0, 0, 150))
        d.text((18, 9), caption["t"], font=font, fill=(255, 255, 255))
        r = eval_state["r"]
        if r is not None:
            y = im.height - 46
            d.rectangle([0, y - 6, im.width, im.height], fill=(0, 0, 0, 150))
            x = 18
            for k, v in r["stages"].items():
                s = ("● " if v else "○ ") + STAGE_KO.get(k.split("/")[0], k)
                d.text((x, y), s, font=font_s, fill=(120, 230, 140) if v else (220, 220, 220))
                x += int(d.textlength(s, font=font_s)) + 26
            s = f"점수 {r['score']:.2f}" + ("  성공" if r["success"] else "")
            d.text((im.width - 18 - int(d.textlength(s, font=font)), y - 8), s, font=font,
                   fill=(120, 230, 140) if r["success"] else (255, 255, 255))
        frames.append(np.asarray(im))

    def run(secs, pins=(), eval_every=15):
        """Advance `secs` (15 Hz env steps, `decimation` physics steps each); `pins` = list of (name, ids, fn) with
        fn(k_physics) -> target positions (env frame) of ids, or None to release."""
        n_env = int(round(secs * 15))
        dec = U.cfg.decimation
        for k in range(n_env):
            U.action_manager.process_action(act)
            for i in range(dec):
                U.action_manager.apply_action()
                kk = k * dec + i
                by = {}
                for name, ids, fn in pins:
                    tgt = fn(kk)
                    if tgt is not None:
                        by.setdefault(name, []).append((ids, tgt))
                for name, items in by.items():
                    obj = U.scene[name]
                    pos = obj.data.nodal_pos_w.torch
                    vel = obj.data.nodal_vel_w.torch.clone()
                    for ids, tgt in items:
                        vel[0, ids] = (tgt + org - pos[0, ids]) / dt
                    obj.write_nodal_velocity_to_sim_index(vel)
                U.scene.write_data_to_sim()
                U.sim.step(render=(i == dec - 1))
                U.scene.update(dt)
            if k % eval_every == 0:
                evaluate()
            frame()

    def smooth(s):
        s = min(1.0, max(0.0, s))
        return s * s * (3 - 2 * s)

    def carry(p0, p1, secs, arc=0.3, hold=0.4):
        """Pinned nodes p0 -> p1 along a raised arc in `secs`, held there for `hold` s, then released."""
        n = int(secs / dt)
        m = int((secs + hold) / dt)
        up = torch.tensor([0.0, 0.0, 1.0], device=dev)

        def fn(kk):
            if kk >= m:
                return None
            s = smooth((kk + 1) / n)
            return p0 + (p1 - p0) * s + up * (arc * math.sin(math.pi * s))
        return fn, secs + hold

    def rotate(p, c, d, ang):
        """Rotate points p about the horizontal line through c with direction d by ang (Rodrigues)."""
        k = d / d.norm()
        q = p - c
        return c + q * math.cos(ang) + torch.cross(k.expand_as(q), q, dim=-1) * math.sin(ang) + \
            k * (q @ k)[:, None] * (1 - math.cos(ang))

    def frame_fit(name, uv, keep, A=None):
        """Garment frame on the desk: world xy ~ A @ uv + b. A (rotation / scale) is fitted once on the flat garment;
        before every fold only the offset b is refitted on the nodes not folded yet (`keep`; folded nodes no longer
        sit at their rest uv): the body slides a few cm while a flap is turned (T-shirt: the half fold landed 10 cm
        off with the offset from before the side folds)."""
        n0 = nodes(name)[keep]
        u = torch.as_tensor(uv, device=dev, dtype=n0.dtype)[keep]
        if A is None:
            X = torch.cat([u, torch.ones(len(u), 1, device=dev, dtype=n0.dtype)], 1)
            A = torch.linalg.lstsq(X, n0[:, :2]).solution[:2].T
        b = (n0[:, :2] - u @ A.T).mean(0)
        return A, b

    def fold(name, uv, axis, line, side, gframe, secs=3.0, gap=0.006):
        """Fold the flap of the garment with sign(uv[axis] - line) == side over the rest line uv[axis] = line: every
        flap node turns 180 deg about the line (found from the nodes on it, at the garment's bottom height) like a rigid
        panel, and lands on top of the rest of the garment (the flip reverses the flap's layer order; it is lifted by
        the garment thickness so it does not fight the layers below)."""
        n0 = nodes(name).clone()
        u = torch.as_tensor(uv, device=dev, dtype=n0.dtype)
        far = (u[:, axis] - line) * side
        flap = torch.nonzero(far > 0.0).flatten()
        A, b = gframe
        other = 1 - axis
        q = torch.zeros(2, device=dev, dtype=n0.dtype)
        q[axis] = line
        e = torch.zeros(2, device=dev, dtype=n0.dtype)
        e[other] = 1.0
        zmin = float(n0[:, 2].min())
        c = torch.cat([A @ q + b, torch.tensor([zmin], device=dev, dtype=n0.dtype)])
        d = torch.cat([A @ e, torch.zeros(1, device=dev, dtype=n0.dtype)])
        p0 = n0[flap]
        probe = rotate(p0[int(far[flap].argmax()):][:1], c, d, math.pi / 2)  # the flap node farthest from the line
        sgn = 1.0 if float(probe[0, 2]) > zmin else -1.0
        rest_part = torch.ones(len(n0), dtype=torch.bool, device=dev)
        rest_part[flap] = False
        lift = (float(n0[rest_part, 2].max()) - zmin) + (float(p0[:, 2].max()) - zmin) + gap
        n = int(secs / dt)
        m = int((secs + 0.6) / dt)
        up = torch.tensor([0.0, 0.0, 1.0], device=dev)

        def fn(kk):
            if kk >= m:
                return None
            s = smooth((kk + 1) / n)
            return rotate(p0, c, d, sgn * math.pi * s) + up * (lift * s)
        # the rest of the garment is held where it lies (a hand flat on it): otherwise the bending hinge dragged the
        # lower half of the towel 9 cm towards the fold line and the half fold missed
        stay = torch.nonzero(rest_part).flatten()
        p_stay = n0[stay]

        def hold(kk):
            return p_stay if kk < m else None
        return [(name, flap, fn), (name, stay, hold)], secs + 0.6

    res = {"scene": os.path.basename(a0.demo_scene), "steps": []}
    aim_observer(U, task, a.eye, a.target)
    caption["t"] = "시작 상태"
    run(0.5)
    evaluate()
    run(1.5)

    snaps = {}

    def step_done(label):
        snaps[label.replace(" ", "_")] = {o.name: nodes(o.name).cpu().numpy() for o in spec.objects
                                          if o.kind in ("towel", "tshirt")}
        r = evaluate()
        res["steps"].append({"step": label, "stages": r["stages"], "score": r["score"], "success": r["success"]})
        print(f"[solution] {label}: score {r['score']:.2f} {r['stages']}", flush=True)

    if spec.task == "BED-01":
        m = task.bed
        lo, hi = m.mattress_min, m.mattress_max
        top = m.mattress_top_z
        pil = nodes("pillow")
        pc = pil.mean(0)
        park = torch.tensor([hi[0] + 0.55, 0.5 * (lo[1] + hi[1]) + 0.4, 0.07], device=dev)
        caption["t"] = "1. 베개를 침대 밖으로 치운다"
        fn, secs = carry(pil.clone(), pil + (park - pc), 3.0, arc=0.35)
        run(secs, [("pillow", torch.arange(len(pil), device=dev), fn)])
        run(1.0)
        step_done("pillow off")
        duv = nodes("duvet")
        rest = U.scene["duvet"].data.default_nodal_state_w.torch[0, :, :3] - org
        if spec.difficulty == "easy":
            caption["t"] = "2. 접힌 이불을 머리 쪽으로 펼친다"
            ids = torch.nonzero(rest[:, 1] > rest[:, 1].max() - 0.03).flatten()  # head edge (folded over to the foot)
            tgt = rest[ids].clone()
            tgt[:, 2] = top + 0.02
            fn, secs = carry(duv[ids].clone(), tgt, 4.5, arc=0.45, hold=0.6)
            run(secs, [("duvet", ids, fn)])
        else:
            caption["t"] = "2. 뭉친 이불을 침대 위에 펼친다"
            tgt = rest.clone()
            tgt[:, 2] = torch.clamp(tgt[:, 2], min=top + 0.02)
            allid = torch.arange(len(duv), device=dev)
            fn, secs = carry(duv.clone(), tgt, 5.0, arc=0.35, hold=0.6)
            run(secs, [("duvet", allid, fn)])
        run(2.0)
        step_done("duvet spread")
        caption["t"] = "3. 베개를 머리맡 가운데에 놓는다"
        pil = nodes("pillow")
        pc = pil.mean(0)
        hb = torch.tensor([m.headboard_center[0], m.headboard_center[1], top + 0.11], device=dev)
        fn, secs = carry(pil.clone(), pil + (hb - pc), 3.0, arc=0.4)
        run(secs, [("pillow", torch.arange(len(pil), device=dev), fn)])
        run(2.0)
        step_done("pillow placed")
    else:
        name = next(o.name for o in spec.objects if o.kind in ("towel", "tshirt"))
        kind = spec.obj(name).kind
        from dxbench.tasks import predicates as P

        uv = task._rest_uv(U, name)
        lines = P.FOLD_LINES[kind]
        if not P.projected_area(nodes(name)[:, :2].cpu().numpy()) >= 0.8 * P.REST_AREA[kind]:
            caption["t"] = "0. 구겨진 옷을 펼친다"
            rest = U.scene[name].data.default_nodal_state_w.torch[0, :, :3] - org
            n0 = nodes(name).clone()
            allid = torch.arange(len(n0), device=dev)
            tgt = rest.clone()
            tgt[:, 2] = task.table_z + 0.01 + (rest[:, 2] - rest[:, 2].min())  # keep the two layers apart
            fn, secs = carry(n0, tgt, 3.5, arc=0.12, hold=0.3)
            run(secs, [(name, allid, fn)])
            run(1.5)
            step_done("flatten")
        unfolded = torch.ones(len(uv), dtype=torch.bool, device=dev)
        A0, _ = frame_fit(name, uv, unfolded)
        uvt = torch.as_tensor(uv, device=dev)

        def to_stack(label):
            """Slide the garment onto the stack spot. The stack predicate also wants it moved >= 8 cm from where the
            episode started (a garment that started next to the spot goes 10 cm from its start towards it, staying
            within the 12 cm radius)."""
            caption["t"] = label
            n0 = nodes(name).clone()
            stack = torch.tensor(task.stack_xy, device=dev, dtype=n0.dtype)
            c0 = torch.as_tensor(task._start[0][name][1], device=dev, dtype=n0.dtype)
            dv = stack - c0
            goal = stack if float(dv.norm()) >= 0.10 else c0 + (dv / dv.norm().clamp(min=1e-6)) * 0.10
            shift = torch.cat([goal - n0[:, :2].mean(0), torch.zeros(1, device=dev, dtype=n0.dtype)])
            fn, secs = carry(n0, n0 + shift, 2.5, arc=0.05, hold=0.3)
            run(secs, [(name, torch.arange(len(n0), device=dev), fn)])
            run(1.0)
            step_done("on stack")

        # the 8-layer T-shirt half fold sits at the predicate's 5 cm tolerance (pairs ~4 cm apart, 3 cm of it the
        # layer stack), so it passes in some runs only; folding on the stack spot instead pushed the shirt off the desk
        for key, label, side in (("fold_left", "1. 왼쪽을 접는다", -1.0), ("fold_right", "2. 오른쪽을 접는다", 1.0),
                                 ("fold_half", "3. 반으로 접는다", 1.0), ("stack", "4. 쌓는 자리로 옮긴다", 0.0)):
            if key == "stack":
                to_stack(label)
                continue
            axis, line = lines[key]
            caption["t"] = label
            pins, secs = fold(name, uv, axis, line, side, frame_fit(name, uv, unfolded, A0))
            unfolded &= ~(((uvt[:, axis] - line) * side) > 0.0)
            run(secs, pins)
            run(1.0)
            step_done(key)
        run(1.0)
    r = evaluate()
    caption["t"] = "완료" + (" — 과제 성공" if r["success"] else f" — 점수 {r['score']:.2f}")
    run(2.5)
    res.update({"final": {"stages": r["stages"], "score": r["score"], "success": r["success"]}})
    np.savez(os.path.join(out, "final_nodes.npz"), **{o.name: nodes(o.name).cpu().numpy() for o in spec.objects
                                                      if o.kind in ("towel", "tshirt", "duvet")})
    if spec.task == "BED-02":
        np.save(os.path.join(out, "rest_uv.npy"), task._rest_uv(U, name))
        np.savez(os.path.join(out, "step_nodes.npz"), **{k: v[name] for k, v in snaps.items()})
    with open(os.path.join(out, "result.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    imageio.mimsave(os.path.join(out, "observer.mp4"), frames, fps=15, codec="libx264", quality=7)
    print(f"[solution] final: success={r['success']} score={r['score']:.2f} frames={len(frames)}", flush=True)
    rt.close()


if __name__ == "__main__":
    main()
