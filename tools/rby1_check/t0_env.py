"""T0: record the software/hardware environment to reports/rby1_check/env.md.

Run inside dx_arena (no Kit needed):
    python tools/rby1_check/t0_env.py
"""
from __future__ import annotations

import importlib.metadata as md
import platform
import subprocess
import sys

from common import ARENA, ISAACLAB, RBY1, REPORTS, out_dir


def sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"(error: {e})"


def ver(pkg: str) -> str:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return "(not installed)"


def git_info(path) -> str:
    sha = sh(["git", "-C", str(path), "rev-parse", "HEAD"])
    desc = sh(["git", "-C", str(path), "describe", "--all", "--always"])
    dirty = sh(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"])
    return f"`{sha[:10]}` ({desc}){' **dirty**' if dirty else ''}"


def main():
    out_dir()
    gpu = sh(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    cuda_drv = sh(["bash", "-c", "nvidia-smi | grep -o 'CUDA Version: [0-9.]*'"])
    try:
        import torch

        torch_cuda = f"{torch.version.cuda} (available={torch.cuda.is_available()})"
    except Exception as e:  # noqa: BLE001
        torch_cuda = f"(torch import failed: {e})"
    rows = [
        ("Host OS", f"{platform.platform()}"),
        ("Python", sys.version.split()[0] + f" ({sys.executable})"),
        ("GPU / driver / VRAM", gpu),
        ("CUDA (driver)", cuda_drv),
        ("CUDA (torch build)", torch_cuda),
        ("isaacsim", ver("isaacsim")),
        ("Isaac Lab", f"Isaac Lab 3.0 (isaaclab {ver('isaaclab')})"),
        ("isaaclab_physx", ver("isaaclab_physx")),
        ("isaaclab_newton", ver("isaaclab_newton")),
        ("isaaclab_arena", ver("isaaclab_arena")),
        ("torch", ver("torch")),
        ("numpy", ver("numpy")),
        ("warp-lang", ver("warp-lang")),
        ("newton", ver("newton")),
        ("lerobot (must be absent in dx_arena)", ver("lerobot")),
    ]
    commits = [
        ("IsaacLab-Arena", git_info(ARENA)),
        ("Isaac Lab (Arena submodule)", git_info(ISAACLAB)),
        ("rby1-sim-isaac", git_info(RBY1)),
    ]
    lines = ["# T0 환경 기록", "", "| 항목 | 값 |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in rows]
    lines += ["", "## 커밋", "", "| 레포 | 커밋 |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in commits]
    lines += ["", "물리 백엔드: PhysX (`isaaclab_physx`) 고정. Newton 비교는 1단계에서 별도로 진행."]
    (REPORTS / "env.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
