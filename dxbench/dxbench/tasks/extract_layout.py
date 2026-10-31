"""Extract a DX room layout (furniture AABBs, part geometry, room box, placements) from the DX USD.

    python dxbench/dxbench/tasks/extract_layout.py                    # care room (dx_arena, Kit pxr)
    python dxbench/dxbench/tasks/extract_layout.py --room Kitchen --parts <prim> ...

Writes dxbench/tasks/<room>_layout.json (CareRoom -> care_room_layout.json, Kitchen -> kitchen_layout.json) — read by
the scene samplers, predicates and oracles without Kit (scene_spec.load_layout(room)).
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--room", default="CareRoom", help="DX room: USD DX_Hub_Building_add_obj_v1_<room>.usda, Furniture/<room>")
ap.add_argument("--parts", nargs="*", default=None,
                help="furniture sub-prims (relative to Furniture/<room>) whose AABBs go to parts_aabb; "
                     "default for CareRoom: bed mattress / frame / pillow / blanket, desk, T-shirt")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

from pathlib import Path  # noqa: E402

from pxr import Usd, UsdGeom  # noqa: E402

from dxbench.paths import DX_ADD_OBJ, FURNITURE_CFG  # noqa: E402

ROOM = args.room
ROOM_USD = DX_ADD_OBJ / f"DX_Hub_Building_add_obj_v1_{ROOM}.usda"
KEY = "".join("_" + ch.lower() if ch.isupper() and i else ch.lower() for i, ch in enumerate(ROOM))  # CareRoom -> care_room
OUT = Path(__file__).resolve().parent / f"{KEY}_layout.json"
PARTS = args.parts if args.parts is not None else (
    ["Bed/Mattress", "Bed/Frame", "Bed/Pillow", "Bed/Blanket", "CareDesk", "TShirt/Shirt"] if ROOM == "CareRoom" else [])


def main():
    st = Usd.Stage.Open(str(ROOM_USD))
    root = st.GetDefaultPrim()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"], useExtentsHint=False)

    def aabb(p):
        r = cache.ComputeWorldBound(p).ComputeAlignedRange()
        return [list(map(float, r.GetMin())), list(map(float, r.GetMax()))]

    fu = root.GetPrimAtPath(f"Furniture/{ROOM}")
    furn = {c.GetName(): aabb(c) for c in fu.GetChildren() if c.IsActive()}
    sub = {}
    for name in PARTS:
        p = fu.GetPrimAtPath(name)
        if p and p.IsValid():
            sub[name] = aabb(p)
    cfg = json.loads(Path(FURNITURE_CFG).read_text())
    room = cfg.get("rooms", {}).get(ROOM) or cfg.get("rooms", {}).get(KEY)
    out = {"source": str(ROOM_USD), "furniture_aabb": furn, "parts_aabb": sub,
           "room_box_xy": json.loads(st.GetRootLayer().customLayerData.get("boxes", "[]"))[0]
           if st.GetRootLayer().customLayerData.get("boxes") else room,
           "placements": cfg["placements"].get(ROOM, {})}
    OUT.write_text(json.dumps(out, indent=1))
    print("[layout]", json.dumps({k: [round(v, 2) for v in b[0][:2] + b[1][:2]] for k, b in furn.items()}))


if __name__ == "__main__":
    main()
    app.close()
