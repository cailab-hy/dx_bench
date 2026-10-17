"""Recompress raw episode frames PNG -> JPEG (q95) in place (raw recordings before the recorder switched to JPEG).
Also removes abandoned `pending-*` recordings (killed episodes) when --drop_pending is given.

    ~/miniforge3/envs/dx_arena/bin/python tools/raw_png_to_jpg.py data/raw/bed03_oracle [--drop_pending]

Only `episode-*` directories whose episode.json exists (finished recordings) are converted; frames being written by
a running recorder live in `pending-*` and are not touched unless --drop_pending (do not use while collecting).
"""
import argparse
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image


def convert(png: Path):
    with Image.open(png) as im:
        im.convert("RGB").save(png.with_suffix(".jpg"), quality=95)
    png.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--drop_pending", action="store_true")
    a = ap.parse_args()
    root = Path(a.root)
    n = 0
    for ep in sorted(root.glob("episode-*")):
        if not (ep / "episode.json").exists():
            continue
        pngs = sorted(ep.glob("*/*.png"))
        with ThreadPoolExecutor(8) as ex:
            list(ex.map(convert, pngs))
        n += len(pngs)
        print(f"[png2jpg] {ep.name}: {len(pngs)} frames", flush=True)
    if a.drop_pending:
        for p in sorted(root.glob("pending-*")):
            shutil.rmtree(p)
            print(f"[png2jpg] removed {p.name}", flush=True)
    print(f"[png2jpg] converted {n} frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
