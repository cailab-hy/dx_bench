"""Teleoperation + raw recording on a dxbench task (dx_arena; rby1_ik embodiment).

    # keyboard (needs the GUI: add --visualizer kit); H = arms home
    python dxbench/dxbench/teleop/run_teleop.py --teleop_device keyboard --record data/raw/bed03_human --visualizer kit \
        --external_environment_class_path dxbench.tasks.bed_tasks:DxBed03Environment dx_bed03 \
        --scene benchmarks/dx_bed_v0/scenes/bed03_easy_000.json --physics_hz 360
    # Quest (adb reverse tcp:8012 tcp:8012; open http://localhost:8012 in the headset browser)
    python dxbench/dxbench/teleop/run_teleop.py --teleop_device vr --record data/raw/bed03_vr ...
    # headless end-to-end VR test (scripted operator through the real WebSocket)
    python dxbench/dxbench/teleop/run_teleop.py --teleop_device scripted_vr --record outputs/teleop_test --save_any ...

Recording starts with the device "start" event (B key / A button). The episode is saved when the task succeeds
(automatic, or ENTER), discarded on "discard" (BACKSPACE / B held) or when the episode ends unsuccessfully.
With --save_any an unsuccessful episode is kept too (tests). Writes <out>/teleop_summary.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from dxbench.utils.arena_runtime import ArenaRuntime


def _args(p):
    p.add_argument("--teleop_device", choices=["keyboard", "vr", "scripted_vr", "scripted_vr_drive"], default="keyboard",
                   help="scripted_vr / scripted_vr_drive: headless VR checks (arm clutch / base + torso sticks)")
    p.add_argument("--record", default="data/raw/teleop")
    p.add_argument("--vr_port", type=int, default=8012)
    p.add_argument("--save_any", action="store_true")
    p.add_argument("--max_episodes", type=int, default=1000)
    p.add_argument("--realtime", action="store_true", help="pace the loop to 15 Hz wall time")


def main():
    if "--enable_cameras" not in sys.argv:
        sys.argv.insert(1, "--enable_cameras")
    rt = ArenaRuntime(add_args=_args)
    env = rt.make_env()
    a = rt.args
    U = env.unwrapped
    from dxbench.paths import REPO
    from dxbench.teleop.devices import KeyboardTeleop, VRTeleopDevice, attach_carb_keyboard
    from dxbench.teleop.recorder import EpisodeRecorder

    out = Path(a.record) if Path(a.record).is_absolute() else REPO / a.record
    out.mkdir(parents=True, exist_ok=True)
    op = None
    if a.teleop_device == "keyboard":
        dev = KeyboardTeleop()
        sub = attach_carb_keyboard(dev)  # noqa: F841 (keep the subscription alive)
    else:
        dev = VRTeleopDevice(a.vr_port)
        if a.teleop_device.startswith("scripted_vr"):
            from dxbench.teleop.scripted_vr import ScriptedVROperator

            op = ScriptedVROperator(a.vr_port, "drive" if a.teleop_device == "scripted_vr_drive" else "arm")
            t0 = time.monotonic()
            while dev.server.status()["clients"] < 1 and time.monotonic() - t0 < 5:
                time.sleep(0.01)
    streamer = None
    if a.teleop_device == "vr":
        from dxbench.teleop.vr.vr_video import VideoStreamer

        streamer = VideoStreamer(dev.server)

    from dxbench.tasks import bed_tasks

    obs, _ = env.reset()
    dev.reset()
    rec, paused, saved, discarded, steps = None, False, [], 0, 0
    summary = {"device": a.teleop_device, "episodes": []}
    tcp_log, base_log, torso_log = [], [], []
    while True:
        t_loop = time.monotonic()
        if op is not None:
            if op.done:
                break
            op.step()
            time.sleep(0.01)  # let the server thread receive the report
        pol = obs["policy"]
        if a.teleop_device == "keyboard":
            act, events = dev.step()
        else:
            tcp = {s: pol[f"tcp_{s}"][0].cpu().numpy() for s in ("right", "left")}
            act, events = dev.step(tcp)
        if "quit" in events:
            break
        if "pause" in events:
            paused = not paused
        if "home" in events:
            for term in ("right_arm", "left_arm"):
                t = U.action_manager.get_term(term)
                if hasattr(t, "request_home"):
                    t.request_home()
        if "start" in events and rec is None:
            rec = EpisodeRecorder.from_env(U, out, policy_name=f"teleop:{a.teleop_device}", scene=getattr(a, "scene", None),
                                           device_name=a.teleop_device)
        if "discard" in events or "reset" in events:
            if rec is not None:
                rec.discard()
                discarded += 1
                rec = None
            obs, _ = env.reset()
            dev.reset()
            continue
        if paused:
            act[:12] = 0.0
            act[14:] = 0.0
        import torch

        act_t = torch.as_tensor(act, device=U.device).unsqueeze(0).repeat(U.num_envs, 1)
        if rec is not None:
            rec.pre_step(obs, act_t, None)
        obs, _, term, trunc, _ = env.step(act_t)
        steps += 1
        if rec is not None and not (bool(term[0]) or bool(trunc[0])):
            rec.post_step()  # not on the terminating step (already reset); finish() trims that frame
        tcp_log.append(obs["policy"]["tcp_right"][0, :3].cpu().numpy().tolist())
        base_log.append(obs["policy"]["base_pose"][0].cpu().numpy().tolist())
        try:
            torso_log.append(int(U.action_manager.get_term("torso").preset_index[0]))
        except (KeyError, AttributeError):
            torso_log.append(-1)
        if streamer is not None and dev.server.status()["clients"]:
            cam = obs.get("camera_obs", {})
            names = {"head": "first_person", "left_wrist": "left_hand", "right_wrist": "right_hand"}
            imgs = {names[k]: cam[f"{k}_rgb"][0].cpu().numpy()[..., :3].astype(np.uint8) for k in names
                    if f"{k}_rgb" in cam}
            line1 = "REC" if rec else ("PAUSED" if paused else "IDLE")
            try:  # driving / torso state for the operator (the sim runs slower than real time, so it looks off)
                tt = U.action_manager.get_term("torso")
                line1 += f"   TORSO {tt.PRESET_ORDER[int(tt.preset_index[0])]}"
            except (KeyError, AttributeError, IndexError, ValueError):
                pass
            text = line1 + (f"\n{dev.base.status()}" if hasattr(dev, "base") else "")
            streamer.submit(imgs, text, "rec" if rec else "idle")
        task = bed_tasks.CURRENT_TASK
        ended = bool(term[0]) or bool(trunc[0])
        if task is None:
            ev = {"success": False, "score": 0.0}
        elif ended:
            # the env already reset inside step(): the termination term that fired is the verdict
            from dxbench.scripts.run_episode import terminated_by

            tb = terminated_by(U)
            ev = {"success": tb == "success", "score": 1.0 if tb == "success" else 0.0}
        else:
            ev = task.evaluate(U)[0]
        if rec is not None and (ev["success"] or "save" in events or ended):
            path = rec.finish(success=bool(ev["success"]), score=float(ev["score"]), save_only_success=not a.save_any)
            summary["episodes"].append({"saved": str(path) if path else None, "task_success": bool(ev["success"]),
                                        "score": float(ev["score"]), "frames": rec.t})
            if path:
                saved.append(str(path))
            else:
                discarded += 1
            rec = None
            if len(saved) >= a.max_episodes:
                break
            if ended:
                obs, _ = env.reset()
                dev.reset()
        elif ended:
            obs, _ = env.reset()
            dev.reset()
        if a.realtime:
            time.sleep(max(0.0, 1 / 15 - (time.monotonic() - t_loop)))
    if rec is not None:  # script ended while recording
        path = rec.finish(success=False, score=0.0, save_only_success=not a.save_any)
        if path:
            saved.append(str(path))
        summary["episodes"].append({"saved": str(path) if path else None, "frames": rec.t})
    summary.update({"saved": saved, "discarded": discarded, "steps": steps})
    if op is not None:
        tl = np.asarray(tcp_log)
        g1, end = op.marks.get("grip1"), op.marks.get("end")
        if op.script == "drive":
            bl = np.asarray(base_log)
            mk = {k: min(v, len(bl) - 1) for k, v in op.marks.items()}

            def fwd(k0, k1):  # distance along the heading at k0
                d = bl[mk[k1], :2] - bl[mk[k0], :2]
                return float(d @ np.array([np.cos(bl[mk[k0], 2]), np.sin(bl[mk[k0], 2])]))
            dyaw = float(np.angle(np.exp(1j * (bl[mk["slow0"], 2] - bl[mk["turn0"], 2]))))
            summary["scripted"] = {
                "forward_m": fwd("drive0", "turn0"), "turn_rad": dyaw,
                "turn_drift_m": float(np.linalg.norm(bl[mk["slow0"], :2] - bl[mk["turn0"], :2])),
                "slow_m": fwd("slow0", "lock0"),
                "locked_m": float(np.linalg.norm(bl[mk["torso0"], :2] - bl[mk["lock0"], :2])),
                "torso_steps": int(torso_log[mk["end"]] - torso_log[mk["torso0"]]),
                "expected": op.expected_drive, "marks": op.marks}
        elif g1 is not None and end is not None:
            moved = tl[min(end, len(tl) - 1)] - tl[g1]
            summary["scripted"] = {"tcp_offset": moved.tolist(), "expected": op.expected_offset.tolist(),
                                   "err_m": float(np.linalg.norm(moved - op.expected_offset)), "marks": op.marks}
        op.close()
    (out / "teleop_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"[teleop] {json.dumps({k: summary[k] for k in ('device', 'saved', 'discarded', 'steps')})} "
          f"scripted={json.dumps(summary.get('scripted'))}", flush=True)
    rt.close()


if __name__ == "__main__":
    main()
