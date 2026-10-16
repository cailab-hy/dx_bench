"""Shared paths and helpers for the RB-Y1 USD checks (no Kit imports at module level)."""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

BENCH = Path(os.path.expanduser("~/dx_bench"))
TOOLS = BENCH / "tools" / "rby1_check"
REPORTS = BENCH / "reports" / "rby1_check"
RBY1 = BENCH / "submodules" / "rby1-sim-isaac"
ARENA = BENCH / "submodules" / "IsaacLab-Arena"
ISAACLAB = ARENA / "submodules" / "IsaacLab"
EXT = TOOLS / "_ext"
URDF = EXT / "rby1-sdk" / "models" / "rby1a" / "urdf" / "model_v1.2.urdf"

ASSETS = RBY1 / "assets"
GEN = ASSETS / "generated"
USD_FILES = {
    "model_a": ASSETS / "model_v_1_2_a.usd",
    "rbgripper_d405": GEN / "rby1_v1_2_a_rbgripper_d405.usd",
    "rbgripper_d405_collision": GEN / "rby1_v1_2_a_rbgripper_d405_collision.usd",
    "fullbody_collision": GEN / "rby1_fullbody_collision.usd",
    "gripper_left": ASSETS / "gripper" / "rb_gripper" / "rb_gripper_left.usd",
    "gripper_right": ASSETS / "gripper" / "rb_gripper" / "rb_gripper_right.usd",
    # 24-DOF override layer built by make_override.py (used for T3/T4)
    "override_24dof": TOOLS / "overrides" / "rby1_24dof.usda",
}
ROBOT_USD = USD_FILES["override_24dof"]

DX_DIR = Path(os.path.expanduser("~/P2-DX_project/DX/USD/DX_Hub_Building_add_obj_v1"))
CARE_SCENE = DX_DIR / "DX_Hub_Building_add_obj_v1_CareRoom_scene.usda"
CARE_SPAWN = (-2.6, 17.7, 0.0)

# rby1-sdk C++ order (src/config.py CPP_JOINT_NAMES) + gripper fingers
CPP_JOINT_NAMES = [
    "right_wheel", "left_wheel",
    "torso_0", "torso_1", "torso_2", "torso_3", "torso_4", "torso_5",
    "right_arm_0", "right_arm_1", "right_arm_2", "right_arm_3", "right_arm_4", "right_arm_5", "right_arm_6",
    "left_arm_0", "left_arm_1", "left_arm_2", "left_arm_3", "left_arm_4", "left_arm_5", "left_arm_6",
    "head_0", "head_1",
]

WHEEL_RADIUS = 0.1
WHEEL_TRACK = 0.53


def out_dir(*parts: str) -> Path:
    p = REPORTS.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _clean(o):
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if hasattr(o, "tolist"):
        return _clean(o.tolist())
    if isinstance(o, Path):
        return str(o)
    return o


def save_json(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(data)
    data.setdefault("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
    path.write_text(json.dumps(_clean(data), indent=2, ensure_ascii=False))
    return path


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def kit_log_path() -> Path | None:
    """Return the newest Kit log file (Isaac Sim writes one per process)."""
    root = Path.home() / ".nvidia-omniverse" / "logs" / "Kit"
    cands = list(root.rglob("kit_*.log")) if root.exists() else []
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def collect_log_lines(log: Path | None, since: float, patterns=("[Warning]", "[Error]")) -> list[str]:
    if log is None or not log.exists() or log.stat().st_mtime < since:
        return []
    out = []
    for line in log.read_text(errors="replace").splitlines():
        if any(p in line for p in patterns):
            out.append(line.strip())
    return out
