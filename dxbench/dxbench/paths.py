"""Filesystem locations used across dxbench (no Kit imports)."""
from __future__ import annotations

import os
from pathlib import Path

PKG = Path(__file__).resolve().parent
REPO = PKG.parent.parent  # ~/dx_bench
RBY1_SIM = REPO / "submodules" / "rby1-sim-isaac"
RBY1_DX_USD = PKG / "assets" / "rby1" / "rby1_dx.usda"
DX_USD = Path(os.environ.get("DX_USD_DIR", os.path.expanduser("~/P2-DX_project/DX/USD")))
DX_ADD_OBJ = DX_USD / "DX_Hub_Building_add_obj_v1"
CARE_ROOM_USD = DX_ADD_OBJ / "DX_Hub_Building_add_obj_v1_CareRoom.usda"
DX_ASSETS = DX_ADD_OBJ / "assets"
FURNITURE_CFG = DX_USD / "config" / "add_obj_v1_furniture.json"
# under the (single) env, not a global prim (D-44, 2026-10-22): cloth resting on global colliders rendered wrong (only
# the parts not touching them showed) though its physics was the same; the env origin is (0, 0, 0)
CARE_ROOM_PRIM = "/World/envs/env_0/DX_Hub_Building_add_obj"
CARE_FURNITURE = f"{CARE_ROOM_PRIM}/Furniture/CareRoom"
CARE_SPAWN = (-2.6, 17.7, 0.0)
DATA = REPO / "data"
OUTPUTS = REPO / "outputs"
