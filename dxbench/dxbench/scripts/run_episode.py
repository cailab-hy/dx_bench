"""Run one episode of a dxbench task with a policy; write metrics JSON (+ optional mp4 / raw recording).

    python dxbench/dxbench/scripts/run_episode.py --policy dxbench.policies.oracle_bed03.Bed03OraclePolicy \
        --out outputs/oracle/bed03_easy_000 [--video] [--record data/raw/bed03_oracle] \
        --external_environment_class_path dxbench.tasks.bed_tasks:DxBed03Environment dx_bed03 \
        --scene benchmarks/dx_bed_v0/scenes/bed03_easy_000.json

Policy: any Arena PolicyBase class (dotted path); its cfg is built from defaults + `--policy_kw k=v ...`.
Output <out>/result.json: task, scene, success, score, failure, stage flags at the end, first-reach step of each stage,
steps, wall time, interventions (0 for scripted), safety violations (robot tilt > 10 deg, object drop).
"""
from __future__ import annotations

import importlib
import json
import math
import os
import sys
import time
from pathlib import Path

from dxbench.utils.arena_runtime import ArenaRuntime


def _args(p):
    p.add_argument("--policy", required=True)
    p.add_argument("--policy_kw", nargs="*", default=[])
    p.add_argument("--out", required=True)
    p.add_argument("--video", action="store_true", help="save head + wrist camera mp4 (needs --enable_cameras)")
    p.add_argument("--record", default=None, help="raw episode recording root (stage 4)")
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--record_all", action="store_true", help="keep unsuccessful recordings too (pipeline tests)")
    p.add_argument("--progress_every", type=int, default=75)


def main():
    if ("--video" in sys.argv or "--record" in sys.argv) and "--enable_cameras" not in sys.argv:
        sys.argv.insert(1, "--enable_cameras")  # before parsing: make_env() re-parses sys.argv
    rt = ArenaRuntime(add_args=_args)
    env = rt.make_env()
    a = rt.args  # re-parsed with the environment arguments (--scene, ...)
    U = env.unwrapped
    import numpy as np
    import torch

    from dxbench.paths import REPO

    out = Path(a.out) if os.path.isabs(a.out) else REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    mod, cls = a.policy.rsplit(".", 1)
    pcls = getattr(importlib.import_module(mod), cls)
    cfg_t = next((b.__args__[0] for b in getattr(pcls, "__orig_bases__", ()) if getattr(b, "__args__", None)), None)
    kw = {}
    for item in a.policy_kw:
        k, v = item.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        kw[k] = v
    policy = pcls(cfg_t(**kw)) if cfg_t else pcls()
    obs, _ = env.reset()
    policy.reset()
    if hasattr(policy, "set_task_description"):
        policy.set_task_description(U.get_language_instruction() if hasattr(U, "get_language_instruction") else None)
    rec = None
    if a.record:
        from dxbench.teleop.recorder import EpisodeRecorder

        rec = EpisodeRecorder.from_env(U, a.record, policy_name=a.policy, scene=getattr(a, "scene", None))
    frames = {k: [] for k in ("head", "left_wrist", "right_wrist")} if a.video else None
    observer = frames is not None and getattr(a, "observer", False) and "observer" in U.scene.keys()
    if observer:
        frames["observer"] = []
    max_steps = a.max_steps or int(U.max_episode_length)
    t0 = time.time()
    first_reach: dict[str, int] = {}
    tilt_max, done, info = 0.0, False, {}
    last_eval = None
    task_obj = _find_task(U)
    snap = {}
    for step in range(max_steps):
        act = policy.get_action(env, obs)
        if rec:
            rec.pre_step(obs, act, getattr(policy, "subtask", None))
        obs, rew, term, trunc, info = env.step(act)
        if rec and not (bool(term[0]) or bool(trunc[0])):
            rec.post_step()  # skipped on the terminating step: the env has reset (the frame is trimmed in finish())
        if frames is not None and "camera_obs" in obs:
            for k in frames:
                key = f"{k}_rgb"
                if key in obs["camera_obs"]:
                    frames[k].append(obs["camera_obs"][key][0].cpu().numpy()[..., :3].astype(np.uint8))
        if observer:
            if step == 0:  # cloth nodes are current after the first step
                from dxbench.tasks.bed_tasks import aim_observer

                aim_observer(U, task_obj, a.observer_eye, getattr(a, "observer_target", None))
            frames["observer"].append(U.scene["observer"].data.output["rgb"].torch[0, ..., :3].cpu().numpy().astype(np.uint8))
        if task_obj is not None and step % 5 == 0:
            last_eval = task_obj.evaluate(U)[0]
            if not (bool(term[0]) or bool(trunc[0])):  # garments every 5 steps (not after the reset) -> final_state.npz
                snap = {n: (U.scene[n].data.nodal_pos_w.torch[0] - U.scene.env_origins[0]).cpu().numpy()
                        for n in getattr(task_obj, "cloth_names", [])}
            for k, v in last_eval["stages"].items():
                if v and k not in first_reach:
                    first_reach[k] = step
        if step % a.progress_every == 0:
            tgt = getattr(policy, "target", None)
            cz = None
            if tgt and tgt in U.scene.deformable_objects.keys():
                cz = round(float(U.scene[tgt].data.nodal_pos_w.torch[0, :, 2].mean()), 3)
            inb = {k.split("/")[-1]: v for k, v in (last_eval or {}).get("stages", {}).items()} if last_eval else None
            print(f"[progress] step {step}/{max_steps} wall {time.time() - t0:.0f}s phase={getattr(policy, 'phase', None)} "
                  f"cloth_z={cz} in={inb} err={getattr(policy, 'last_err', None)} "
                  f"target={getattr(policy, 'target', None)} score={last_eval['score'] if last_eval else None} "
                  f"tilt_max={tilt_max:.1f} base={[round(v, 2) for v in (U.scene['robot'].data.root_link_pos_w.torch[0] - U.scene.env_origins[0]).tolist()]}",
                  flush=True)
        r = U.scene["robot"]
        x, y, z, w = r.data.root_link_quat_w.torch[0].tolist()
        tilt_max = max(tilt_max, math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y))))))
        if bool(term[0]) or bool(trunc[0]):
            done = True
            # the env has already reset this env inside step(): use the task's own evaluation from the
            # termination check (pre-reset), not a new evaluate() of the reset scene
            if task_obj is not None and getattr(task_obj, "last_eval", None):
                last_eval = task_obj.last_eval[0]
            break
        if getattr(policy, "phase", None) == "finished" and step > 10:
            break
    if task_obj is not None and not done:
        last_eval = task_obj.evaluate(U)[0]
    term_name = terminated_by(U)
    if last_eval is not None and term_name == "success":
        # the success termination term is the task predicate itself, evaluated before the env reset inside step();
        # our own last evaluation can be a few steps older. Success implies the full score in all three tasks.
        last_eval = dict(last_eval, success=True, score=1.0, failure=None,
                         stages={k: True for k in last_eval.get("stages", {})})
    elif last_eval is not None and term_name not in (None, "time_out", "success"):
        last_eval = dict(last_eval, success=False, score=0.0, failure=last_eval.get("failure") or term_name)
    res = {"policy": a.policy, "scene": getattr(a, "scene", None), "steps": step + 1, "sim_s": (step + 1) * U.step_dt,
           "wall_s": time.time() - t0, "terminated_by": term_name, "done": done,
           "success": bool(last_eval["success"]) if last_eval else None,
           "score": float(last_eval["score"]) if last_eval else None,
           "failure": last_eval.get("failure") if last_eval else None,
           "stages_end": last_eval.get("stages") if last_eval else None, "first_reach_step": first_reach,
           "interventions": 0, "safety": {"robot_tilt_max_deg": tilt_max, "robot_tipped": tilt_max > 10.0},
           "policy_phase_end": getattr(policy, "phase", None), "policy_plan_log": getattr(policy, "plan_log", None),
           "policy_done": getattr(policy, "done_clothes", None)}
    try:  # pad assist (pad_grasp): per-grasp nodes, fingertip-cloth gap, how far the pinned nodes left the pads
        ca = U.action_manager.get_term("cloth_assist")
        if hasattr(ca, "stats"):
            res["grasp_stats"] = ca.stats
            if hasattr(ca, "wheel_events"):
                res["wheel_trap_events"] = ca.wheel_events
    except (KeyError, ValueError):
        pass
    (out / "result.json").write_text(json.dumps(res, indent=1, default=float))
    if task_obj is not None and snap:  # last garment state before the end (<= 5 steps old) for offline predicate audits
        np.savez_compressed(out / "final_state.npz", **{k: v.astype(np.float32) for k, v in snap.items()})
    if frames is not None and frames["head"]:
        import imageio

        for k, fr in frames.items():
            if fr:
                imageio.mimsave(out / f"{k}.mp4", fr, fps=15, codec="libx264", quality=7)
    if rec:
        saved = rec.finish(success=res["success"], score=res["score"], save_only_success=not a.record_all)
        res["recording"] = str(saved) if saved else None
        (out / "result.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"[episode] {json.dumps({k: res[k] for k in ('scene', 'success', 'score', 'failure', 'steps', 'terminated_by')})}")
    sys.stdout.flush()
    rt.close()


def terminated_by(U) -> str | None:
    """Name of the termination term that fired at the last step (env 0), or None."""
    try:
        tm = U.termination_manager
        for n in tm.active_terms:
            if bool(tm.get_term(n)[0]):
                return n
    except Exception:  # noqa: BLE001
        pass
    return None


def _find_task(U):
    """The Arena task object (BedTaskBase) — stored on the env cfg by the environment factory."""
    t = getattr(U.cfg, "dx_task", None)
    if t is None:
        from dxbench.tasks import bed_tasks

        t = bed_tasks.CURRENT_TASK
    return t


if __name__ == "__main__":
    main()
