"""Scene JSON: one episode's initial state, shared by simulation reset and real-robot placement.

{
  "scene_id": "bed03_easy_003", "task": "BED-03", "difficulty": "easy", "seed": 3,
  "robot": {"xy": [x, y], "yaw": rad, "torso": "upright"},
  "objects": [ {"name": "shirt_0", "kind": "tshirt", "variant": "m_blue", "xyz": [..], "yaw": rad,
                "cloth_state": "states/bed03_easy_003_shirt_0.npz" | null}, ...
               {"name": "laundry_basket", "kind": "basket", "xyz": [..], "yaw": rad}, ... ],
  "lighting": {"blinds": "down", "lighting": "day", "intensity": 1.0},  # DX Building variant sets (D-42)
  "language": {"high": "...", "subtasks": [...]}
}
Coordinates: DX world frame (Z-up, m). Cloth states hold absolute world node positions after settling.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from dxbench.tasks.predicates import TEXT

LAYOUT = Path(__file__).resolve().parent / "care_room_layout.json"


# DX Building variant sets: blinds down | half | up, lighting day | day_sunny | night (| none); intensity scales every
# light under the building. The default is what the DX care-room file selects (D-42).
DEFAULT_LIGHTING = {"blinds": "down", "lighting": "day", "intensity": 1.0}


@dataclass
class ObjSpec:
    name: str
    kind: str  # tshirt | towel | sock | basket | doll | book
    xyz: list
    yaw: float = 0.0
    variant: str | None = None
    cloth_state: str | None = None


@dataclass
class SceneSpec:
    scene_id: str
    task: str
    difficulty: str
    seed: int
    robot: dict
    objects: list = field(default_factory=list)
    lighting: dict = field(default_factory=lambda: dict(DEFAULT_LIGHTING))
    language: dict = field(default_factory=dict)

    def to_json(self, path: Path):
        d = asdict(self)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(d, indent=1))

    @staticmethod
    def from_json(path) -> SceneSpec:
        d = json.loads(Path(path).read_text())
        d["objects"] = [ObjSpec(**o) for o in d["objects"]]
        if "room_lights" in d.get("lighting", {}):  # pre-D-42 placeholder, never applied
            d["lighting"] = dict(DEFAULT_LIGHTING)
        return SceneSpec(**d)

    def obj(self, name) -> ObjSpec:
        return next(o for o in self.objects if o.name == name)

    def by_kind(self, *kinds) -> list[ObjSpec]:
        return [o for o in self.objects if o.kind in kinds]


def load_layout() -> dict:
    return json.loads(LAYOUT.read_text())


# ------------------------------------------------------------------------------------------ floor sampling
def free_floor_ok(xy, layout: dict, clearance: float = 0.7, extra: list | None = None) -> bool:
    """>= clearance from walls (room box) and from every furniture AABB (and extra (xy, r) discs)."""
    x0, x1, y0, y1 = layout["room_box_xy"]
    # the room box includes ~0.2 m margin beyond the inner wall faces
    wx0, wx1, wy0, wy1 = x0 + 0.2, x1 - 0.2, y0 + 0.2, y1 - 0.2
    x, y = xy
    if min(x - wx0, wx1 - x, y - wy0, wy1 - y) < clearance:
        return False
    for _name, (lo, hi) in layout["furniture_aabb"].items():
        dx = max(lo[0] - x, 0.0, x - hi[0])
        dy = max(lo[1] - y, 0.0, y - hi[1])
        if math.hypot(dx, dy) < clearance:
            return False
    for (ex, ey), r in extra or []:
        if math.hypot(x - ex, y - ey) < r:
            return False
    return True


def sample_floor(rng: np.random.Generator, layout: dict, n: int, clearance: float = 0.7, sep: float = 0.6,
                 avoid: list | None = None, tries: int = 4000) -> list[tuple[float, float]]:
    x0, x1, y0, y1 = layout["room_box_xy"]
    out: list[tuple[float, float]] = []
    extra = list(avoid or [])
    for _ in range(tries):
        if len(out) == n:
            break
        p = (float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1)))
        if free_floor_ok(p, layout, clearance, extra):
            out.append(p)
            extra.append((p, sep))
    if len(out) < n:
        raise RuntimeError(f"could only place {len(out)}/{n} floor items with clearance {clearance} m")
    return out


def language(task: str) -> dict:
    t = TEXT[task]
    return {"high": t.high[0], "high_alternatives": t.high[1:], "subtasks": t.subtasks}
