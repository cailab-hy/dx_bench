"""Websocket policy server (dx_policy env) speaking the Arena / openpi remote protocol.

    ~/miniforge3/envs/dx_policy/bin/python -m dxbench_policy.serve.server --backend lerobot \
        --checkpoint outputs/train/act_bed03/checkpoints/last/pretrained_model --port 8765
    ... --backend hold                                   # "zero" policy: hold the current posture
    ... --backend replay --dataset data/lerobot/dx_bed03_oracle_dx24 --episode 0

Protocol (isaaclab_arena/policy/remote_policy_base.py + openpi_client.msgpack_numpy):
  on connect  server -> metadata {"action_dim", "chunk", "fmt", "state_names", "action_names", "image_keys", "fps", "backend"}
  request     {"observation.state": (D,), "observation.images.<front|right|left>": (H, W, 3) uint8, "task": str}
              or {"reset": True} at episode start (backend state: replay cursor, ACT temporal queue)
  response    {"actions": (H, action_dim) float32}   (reset -> {"actions": (1, action_dim) zeros, "reset": True})
  errors      a text frame with the traceback (the client raises).
Formats: dx24 (= rby1_joint 24-D action, state 24) or rby1 (16-D). See convert/raw_to_lerobot.py.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
import traceback

import numpy as np

from dxbench_policy import rby1_format as R
from dxbench_policy.convert.raw_to_lerobot import DX24_ACTION, DX24_STATE

try:  # openpi-compatible msgpack with numpy support (same wire format as openpi_client.msgpack_numpy)
    from openpi_client import msgpack_numpy as mpn
except ImportError:  # dx_policy has no openpi_client: inline copy of its two hooks
    import functools

    import msgpack

    def _pack(obj):
        if isinstance(obj, np.ndarray):
            if obj.dtype.kind in ("V", "O", "c"):
                raise ValueError(f"Unsupported dtype: {obj.dtype}")
            return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
        if isinstance(obj, np.generic):
            return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
        return obj

    def _unpack(obj):
        if b"__ndarray__" in obj:
            return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
        if b"__npgeneric__" in obj:
            return np.dtype(obj[b"dtype"]).type(obj[b"data"])
        return obj

    class mpn:  # noqa: N801
        packb = staticmethod(functools.partial(msgpack.packb, default=_pack))
        unpackb = staticmethod(functools.partial(msgpack.unpackb, object_hook=_unpack))

IMAGE_KEYS = ["observation.images.front", "observation.images.right", "observation.images.left"]


class Backend:
    """A policy backend. `session()` gives each websocket connection its own state (replay cursor, ...), so several
    simulator clients can share one server (benchmark runs scenes in parallel); models are shared."""
    fmt = "dx24"
    chunk = 1

    @property
    def action_dim(self):
        return 24 if self.fmt == "dx24" else 16

    def session(self):
        import copy

        return copy.copy(self)

    def reset(self):
        pass

    def infer(self, obs: dict) -> np.ndarray:
        raise NotImplementedError


class HoldBackend(Backend):
    """'zero' policy: arm/torso targets = current joint positions, grippers open, base still."""

    def __init__(self, fmt="dx24", chunk=8):
        self.fmt, self.chunk = fmt, chunk

    def infer(self, obs):
        s = np.asarray(obs["observation.state"], dtype=np.float32)
        a = s[: self.action_dim].copy()
        a[14:16] = 1.0
        if self.fmt == "dx24":
            a[22:24] = 0.0
        return np.repeat(a[None], self.chunk, 0)


class ReplayBackend(Backend):
    """Stream the actions of one dataset episode in order (open-loop replay through the server)."""

    def __init__(self, dataset, episode=0, chunk=8):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        ds = LeRobotDataset(repo_id="local/replay", root=dataset)
        lo = int(ds.meta.episodes["dataset_from_index"][episode])
        hi = int(ds.meta.episodes["dataset_to_index"][episode])
        self.actions = np.stack([np.asarray(x) for x in ds.hf_dataset.select(range(lo, hi))["action"]]).astype(np.float32)
        self.fmt = "dx24" if self.actions.shape[1] == 24 else "rby1"
        self.chunk = chunk
        self.t = 0

    def reset(self):
        self.t = 0

    def infer(self, obs):
        idx = np.minimum(np.arange(self.t, self.t + self.chunk), len(self.actions) - 1)
        self.t += self.chunk
        return self.actions[idx]


class LeRobotBackend(Backend):
    """Any LeRobot pretrained policy directory (ACT, SmolVLA, ...) with its saved pre/post processors."""

    def __init__(self, checkpoint, device="cuda", chunk=None):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        self.torch = torch
        cfg = PreTrainedConfig.from_pretrained(checkpoint)
        cfg.device = device
        self.policy = get_policy_class(cfg.type).from_pretrained(checkpoint, config=cfg).to(device).eval()
        self.pre, self.post = make_pre_post_processors(cfg, pretrained_path=checkpoint,
                                                       preprocessor_overrides={"device_processor": {"device": device}})
        out = cfg.output_features["action"].shape[0]
        self.fmt = "dx24" if out == 24 else "rby1"
        self.image_keys = [k for k in cfg.input_features if k.startswith("observation.images.")]
        self.image_shapes = {k: tuple(cfg.input_features[k].shape) for k in self.image_keys}
        n = getattr(cfg, "n_action_steps", None) or getattr(cfg, "chunk_size", 1)
        self.chunk = int(chunk or n)
        self.device = device
        self.type = cfg.type

    def reset(self):
        pass  # predict_action_chunk is stateless (no action queue): nothing per episode

    def infer(self, obs):
        torch = self.torch
        batch = {"observation.state": torch.as_tensor(np.array(obs["observation.state"], np.float32))[None]}
        # all incoming cameras: the saved preprocessor renames them (e.g. SmolVLA front -> camera1, training rename_map);
        # frames are resized to the size the policy was trained on (dx24 datasets may store scaled frames)
        for k in [k for k in obs if k.startswith("observation.images.")]:
            img = torch.as_tensor(np.array(obs[k])).permute(2, 0, 1)[None].float() / 255.0
            shp = self.image_shapes.get(k) or (next(iter(self.image_shapes.values())) if self.image_shapes else None)
            if shp is not None and tuple(img.shape[-2:]) != tuple(shp[-2:]):
                img = torch.nn.functional.interpolate(img, size=tuple(shp[-2:]), mode="bilinear", align_corners=False)
            batch[k] = img
        batch["task"] = [obs.get("task") or ""]
        with torch.no_grad():
            b = self.pre(batch)
            chunk = self.policy.predict_action_chunk(b)  # (1, H, D) normalized
            chunk = self.post(chunk)
        a = chunk[0].float().cpu().numpy()[: self.chunk]
        if a.shape[1] >= 16:
            a[:, 14:16] = (a[:, 14:16] >= 0.5).astype(np.float32)  # binary gripper like the data
        return a


def make_backend(a) -> Backend:
    if a.backend == "hold":
        return HoldBackend(a.fmt, a.chunk or 8)
    if a.backend == "replay":
        return ReplayBackend(a.dataset, a.episode, a.chunk or 8)
    return LeRobotBackend(a.checkpoint, a.device, a.chunk)


async def serve(backend: Backend, host: str, port: int, stats_every: int = 100):
    import websockets

    meta = {"backend": type(backend).__name__, "fmt": backend.fmt, "action_dim": backend.action_dim,
            "chunk": backend.chunk, "fps": R.FPS, "image_keys": IMAGE_KEYS,
            "state_names": DX24_STATE if backend.fmt == "dx24" else R.NAMES,
            "action_names": DX24_ACTION if backend.fmt == "dx24" else R.NAMES}

    async def handler(ws):
        try:
            await _session(ws)
        except websockets.exceptions.ConnectionClosed:
            logging.info("client disconnected")

    async def _session(ws):
        await ws.send(mpn.packb(meta))
        sess = backend.session()
        n, t_sum = 0, 0.0
        async for msg in ws:
            try:
                obs = mpn.unpackb(msg)
                obs = {(k.decode() if isinstance(k, bytes) else k): v for k, v in obs.items()}
                if obs.get("reset"):
                    sess.reset()
                    await ws.send(mpn.packb({"actions": np.zeros((1, backend.action_dim), np.float32), "reset": True}))
                    continue
                t0 = time.perf_counter()
                act = np.ascontiguousarray(await asyncio.to_thread(sess.infer, obs), dtype=np.float32)
                t_sum += time.perf_counter() - t0
                n += 1
                if n % stats_every == 0:
                    logging.info(f"{n} requests, mean infer {1000 * t_sum / n:.1f} ms")
                await ws.send(mpn.packb({"actions": act, "infer_ms": 1000 * t_sum / max(n, 1)}))
            except Exception:  # noqa: BLE001
                await ws.send(traceback.format_exc())

    async with websockets.serve(handler, host, port, compression=None, max_size=None, ping_interval=20,
                                ping_timeout=300):
        print(f"[server] {json.dumps({k: meta[k] for k in ('backend', 'fmt', 'action_dim', 'chunk')})} "
              f"listening on ws://{host}:{port}", flush=True)
        await asyncio.Future()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["hold", "replay", "lerobot"], default="lerobot")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--fmt", choices=["dx24", "rby1"], default="dx24")
    ap.add_argument("--chunk", type=int, default=None, help="actions returned per request")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve(make_backend(a), a.host, a.port))


if __name__ == "__main__":
    main()
