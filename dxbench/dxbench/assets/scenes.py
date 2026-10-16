"""Scene (background) assets registered with Arena."""
from __future__ import annotations

from pathlib import Path

from isaaclab_arena.assets.background import Background
from isaaclab_arena.assets.register import register_asset

from dxbench.paths import CARE_ROOM_PRIM, CARE_ROOM_USD

FLAT_USD = Path(__file__).resolve().parent / "scenes" / "flat_ground.usda"


@register_asset
class DxFlatGround(Background):
    """40 x 40 m static ground at z = 0 (regression tests)."""

    name = "dx_flat_ground"
    tags = ["background"]

    def __init__(self, **kwargs):
        super().__init__(name=self.name, tags=self.tags, usd_path=str(FLAT_USD), object_min_z=0.0,
                         prim_path="/World/DxFlatGround", reset_nested_physics=False, **kwargs)


@register_asset
class DxCareRoom(Background):
    """DX Hub 11F care room (bedroom) — `DX_Hub_Building_add_obj_v1_CareRoom.usda` referenced once at a fixed
    world prim (/World/DX_Hub_Building_add_obj), so furniture prim paths match the *_scene.usda stage and the
    WallAnchor world joints of the wardrobe stay valid. Contains its own lights. Nested physics is not reset."""

    name = "dx_care_room"
    tags = ["background"]

    def __init__(self, **kwargs):
        super().__init__(name=self.name, tags=self.tags, usd_path=str(CARE_ROOM_USD), object_min_z=0.0,
                         prim_path=CARE_ROOM_PRIM, reset_nested_physics=False, **kwargs)
