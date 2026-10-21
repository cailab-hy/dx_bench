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


def apply_care_room_lighting(env, env_ids, lighting: dict | None = None):
    """Prestartup event (stage edit only, the DX files are untouched): select the DX Building variant sets
    `blinds` (down | half | up) and `lighting` (day | day_sunny | night | none) and scale every light under the
    building by `intensity` (scene JSON "lighting", D-42)."""
    import isaaclab.sim as sim_utils
    from pxr import Usd, UsdLux

    lighting = lighting or {}
    building = sim_utils.get_current_stage().GetPrimAtPath(f"{CARE_ROOM_PRIM}/Building")
    if not building.IsValid():
        return
    sets = building.GetVariantSets()
    for key in ("blinds", "lighting"):
        if key in lighting and sets.HasVariantSet(key):
            sets.GetVariantSet(key).SetVariantSelection(str(lighting[key]))
    scale = float(lighting.get("intensity", 1.0))
    if scale != 1.0:
        for prim in Usd.PrimRange(building):
            if prim.HasAPI(UsdLux.LightAPI):
                attr = UsdLux.LightAPI(prim).GetIntensityAttr()
                attr.Set(float(attr.Get()) * scale)


CARE_ROOM_FLOOR = "Building/Collision/floor_325"  # DX floor collider: a Cube scaled 29 x 47 x 0.1 m (top at z = 0)


def replace_care_room_floor(env=None, env_ids=None):
    """Prestartup event (stage edit only, the DX files are untouched): switch off the DX floor collider (a thin
    Cube scaled 29 x 47 x 0.1 m) and add an invisible collision Plane at z = 0. On the GPU the scaled thin box gave
    jittery wheel contacts that flung / flipped the robot at random places in the room (stage 3; same failure as the
    flat test ground, D-9). DECISIONS D-22."""
    import isaaclab.sim as sim_utils
    from pxr import UsdGeom, UsdPhysics

    stage = sim_utils.get_current_stage()
    floor = stage.GetPrimAtPath(f"{CARE_ROOM_PRIM}/{CARE_ROOM_FLOOR}")
    if floor.IsValid():
        UsdPhysics.CollisionAPI.Apply(floor).CreateCollisionEnabledAttr(False)
    if not stage.GetPrimAtPath("/World/DxFloorPlane").IsValid():
        plane = UsdGeom.Plane.Define(stage, "/World/DxFloorPlane")
        plane.CreateAxisAttr("Z")
        plane.CreateLengthAttr(60.0)
        plane.CreateWidthAttr(60.0)
        plane.CreatePurposeAttr("guide")
        UsdPhysics.CollisionAPI.Apply(plane.GetPrim())

