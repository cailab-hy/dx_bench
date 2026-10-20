"""Object assets for the bedroom tasks (factories returning Arena objects).

Garments are surface deformables. The DX T-shirt variants already carry the deformable body API in their USD, so
they use `PrebuiltClothObject` (no deformable_props: adding props to a UsdFileCfg creates a second simulation
mesh). Towels / socks are procedural `MeshRectangleCfg` cloths with the DX T-shirt material values.
Cloth parameters follow config/cloth_params.yaml (stage 1).
"""
from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab_physx.sim.schemas.schemas_cfg import PhysxCollisionCfg

from isaaclab_arena.assets.deformable_object import DeformableObject
from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.utils.physics_backend import PhysicsBackend
from isaaclab_arena.utils.pose import Pose

OBJ = Path(__file__).resolve().parent / "objects"
TSHIRTS = ("s_red", "m_blue", "l_gray", "m_white", "m95_blue", "m95_wine", "m95_green")  # m95_*: real product (D-36)


def srgb_to_linear(hexstr: str) -> tuple:
    c = [int(hexstr[i:i + 2], 16) / 255.0 for i in (1, 3, 5)]
    return tuple(v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in c)


# Real towel (D-36): LUNOO 타올, 02 시그니처 컬러 세트, 40 x 80 cm, polyester 85 % / nylon 15 % microfibre, 120 g,
# 375 GSM. Colours: median of a lit patch in the product photos (sRGB).
TOWEL_COLORS = {"01_charcoalgray": "#7a707f", "05_coolgray": "#bbbdc8", "06_white": "#d3d2d8", "07_oatmeal": "#e2c9b9",
                "08_lightpink": "#e3c9d2", "09_ivory": "#e1d9d2", "10_lavender": "#ac94af", "11_applegreen": "#c6d4c1",
                "12_skyblue": "#bccae4", "13_sandbeige": "#c8988d"}
TOWEL_SIZE = (0.80, 0.40)
TOWEL_GSM = 375.0
CLOTH_THICKNESS = 0.0015  # simulation surface thickness (m); mass = density x thickness x area


class PrebuiltClothObject(DeformableObject):
    """Deformable from a USD that already has OmniPhysicsDeformableBodyAPI (PhysX)."""

    @staticmethod
    def _infer_physics_preset(spawner_cfg) -> PhysicsBackend:
        if spawner_cfg.deformable_props is None:
            return PhysicsBackend.PHYSX
        return DeformableObject._infer_physics_preset(spawner_cfg)


def _pose(xyz, yaw=0.0) -> Pose:
    import math

    return Pose(position_xyz=tuple(float(v) for v in xyz), rotation_xyzw=(0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)))


def tshirt(name: str, variant: str = "m_blue", xyz=(0.0, 0.0, 0.0), yaw: float = 0.0) -> PrebuiltClothObject:
    return PrebuiltClothObject(name=name, spawner_cfg=sim_utils.UsdFileCfg(usd_path=str(OBJ / f"tshirt_{variant}.usda")),
                               initial_pose=_pose(xyz, yaw))


def rect_cloth(name: str, size=(0.7, 0.45), xyz=(0.0, 0.0, 0.0), yaw: float = 0.0, rgb=(0.9, 0.85, 0.6),
               spacing_m: float = 0.023, iters: int = 16, self_collision: bool = True,
               youngs_modulus: float = 1.0e6, density: float = 120.0, friction: tuple = (0.72, 0.6)) -> DeformableObject:
    import math

    diag = math.hypot(*size)
    cfg = sim_utils.MeshRectangleCfg(
        size=size, edge_refinement=max(1.0, diag / (1.45 * spacing_m)),
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(solver_position_iteration_count=iters,
                                                               self_collision=self_collision),
        physics_material=sim_utils.PhysxSurfaceDeformableBodyMaterialCfg(
            density=density, static_friction=friction[0], dynamic_friction=friction[1], youngs_modulus=youngs_modulus,
            poissons_ratio=0.25, surface_thickness=CLOTH_THICKNESS, bend_damping=0.1, elasticity_damping=0.1),
        collision_props=[PhysxCollisionCfg(contact_offset=0.008, rest_offset=0.002)],
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb))
    return DeformableObject(name=name, spawner_cfg=cfg, initial_pose=_pose(xyz, yaw))


TOWEL_YOUNGS = 1.0e5  # D-28: at 1e6 a folded towel springs back flat after release (1/6); 1e5 keeps it (3/3), strain < 1 %


def towel(name: str, xyz=(0.0, 0.0, 0.0), yaw: float = 0.0, small: bool = False,
          color: str | None = None) -> DeformableObject:
    """`color` (a TOWEL_COLORS key): the real 40 x 80 cm microfibre towel (D-36) - 375 GSM, so density =
    0.375 kg/m2 / thickness; microfibre grips more than cotton (friction 0.9 / 0.75, inferred). Without `color`:
    the stage-3 procedural towel (0.7 x 0.45, small 0.5 x 0.3), kept for old scenes."""
    if color is not None:
        return rect_cloth(name, TOWEL_SIZE, xyz, yaw, rgb=srgb_to_linear(TOWEL_COLORS[color]),
                          youngs_modulus=TOWEL_YOUNGS, density=TOWEL_GSM / 1000.0 / CLOTH_THICKNESS, friction=(0.9, 0.75))
    return rect_cloth(name, (0.5, 0.3) if small else (0.7, 0.45), xyz, yaw, rgb=(0.95, 0.9, 0.7),
                      youngs_modulus=TOWEL_YOUNGS)


def sock(name: str, xyz=(0.0, 0.0, 0.0), yaw: float = 0.0) -> DeformableObject:
    return rect_cloth(name, (0.24, 0.10), xyz, yaw, rgb=(0.2, 0.2, 0.25), spacing_m=0.012)


def basket(name: str = "laundry_basket", xyz=(0.0, 0.0, 0.0), yaw: float = 0.0, variant: str | None = None) -> Object:
    """variant "gray": the real tapered basket (D-36, 49 x 36 -> 38 x 28 x 40 cm); None: the stage-3 60 x 44 x 32 box."""
    usd = "laundry_basket_gray.usda" if variant == "gray" else "laundry_basket.usda"
    return Object(name=name, usd_path=str(OBJ / usd), object_type=ObjectType.RIGID, initial_pose=_pose(xyz, yaw))


def doll(name: str = "doll", xyz=(0.0, 0.0, 0.0), yaw: float = 0.0) -> Object:
    return Object(name=name, usd_path=str(OBJ / "doll.usda"), object_type=ObjectType.RIGID, initial_pose=_pose(xyz, yaw))


def book(name: str = "book", xyz=(0.0, 0.0, 0.0), yaw: float = 0.0) -> Object:
    return Object(name=name, usd_path=str(OBJ / "book.usda"), object_type=ObjectType.RIGID, initial_pose=_pose(xyz, yaw))


def pillow(name: str = "pillow", xyz=(0.0, 0.0, 0.0), yaw: float = 0.0) -> DeformableObject:
    """Volume-deformable pillow 66 x 42 x 13 cm with the DX PillowPhysics values (E 4 kPa, density 35)."""
    cfg = sim_utils.MeshCuboidCfg(
        size=(0.66, 0.42, 0.13), edge_refinement=4.0,
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(solver_position_iteration_count=16),
        physics_material=sim_utils.DeformableBodyMaterialCfg(youngs_modulus=4000.0, poissons_ratio=0.3, density=35.0,
                                                             static_friction=0.7, dynamic_friction=0.6),
        collision_props=[PhysxCollisionCfg(contact_offset=0.01, rest_offset=0.002)],
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.95, 0.97)))
    return DeformableObject(name=name, spawner_cfg=cfg, initial_pose=_pose(xyz, yaw))
