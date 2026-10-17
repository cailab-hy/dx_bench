"""Extract the care-room layout (furniture AABBs, mattress/desk geometry, free floor) from the DX USD.

    python dxbench/dxbench/tasks/extract_layout.py     # (dx_arena, Kit pxr)

Writes dxbench/tasks/care_room_layout.json — used by the scene samplers and predicates without Kit.
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

from pathlib import Path  # noqa: E402

from pxr import Usd, UsdGeom  # noqa: E402

from dxbench.paths import CARE_ROOM_USD, FURNITURE_CFG  # noqa: E402

OUT = Path(__file__).resolve().parent / "care_room_layout.json"


def main():
    st = Usd.Stage.Open(str(CARE_ROOM_USD))
    root = st.GetDefaultPrim()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"], useExtentsHint=False)

    def aabb(p):
        r = cache.ComputeWorldBound(p).ComputeAlignedRange()
        return [list(map(float, r.GetMin())), list(map(float, r.GetMax()))]

    fu = root.GetPrimAtPath("Furniture/CareRoom")
    furn = {c.GetName(): aabb(c) for c in fu.GetChildren() if c.IsActive()}
    sub = {}
    for name in ("Bed/Mattress", "Bed/Frame", "Bed/Pillow", "Bed/Blanket", "CareDesk", "TShirt/Shirt"):
        p = fu.GetPrimAtPath(name)
        if p and p.IsValid():
            sub[name] = aabb(p)
    cfg = json.loads(Path(FURNITURE_CFG).read_text())
    room = cfg.get("rooms", {}).get("CareRoom") or cfg.get("rooms", {}).get("care_room")
    out = {"source": str(CARE_ROOM_USD), "furniture_aabb": furn, "parts_aabb": sub,
           "room_box_xy": json.loads(st.GetRootLayer().customLayerData.get("boxes", "[]"))[0]
           if st.GetRootLayer().customLayerData.get("boxes") else room,
           "placements": cfg["placements"]["CareRoom"]}
    OUT.write_text(json.dumps(out, indent=1))
    print("[layout]", json.dumps({k: [round(v, 2) for v in b[0][:2] + b[1][:2]] for k, b in furn.items()}))


if __name__ == "__main__":
    main()
    app.close()
