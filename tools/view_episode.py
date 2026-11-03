"""One episode of a dx24 LeRobot dataset as an mp4, without the simulator (dx_policy env): the head / right wrist /
left wrist cameras side by side, with the instruction, time, subtask and the gripper / base commands on top.

    ~/miniforge3/envs/dx_policy/bin/python tools/view_episode.py samples/oracle_d70_dx24 2 outputs/view_bed03.mp4 [start end]

`start end`: only these frames of the episode (default: all).

Subtask names come from meta/dx_episodes/*.json when the dataset has them (samples/), else the stage index is shown.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CAMS = ("observation.images.front", "observation.images.right", "observation.images.left")
LABELS = ("head", "right wrist", "left wrist")
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"  # Korean instructions; falls back to PIL's default


def font(size):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default()


def subtask_names(root: Path, episode: int) -> list[str] | None:
    files = sorted((root / "meta" / "dx_episodes").glob(f"{episode}_*.json"))
    return json.loads(files[0].read_text())["language"]["subtasks"] if files else None


def main():
    root, episode, out = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("dxbench/view", root=root)
    ep_lo = int(ds.meta.episodes["dataset_from_index"][episode])
    ep_hi = int(ds.meta.episodes["dataset_to_index"][episode])
    lo, hi = ep_lo, ep_hi
    if len(sys.argv) > 5:
        lo, hi = ep_lo + int(sys.argv[4]), min(ep_hi, ep_lo + int(sys.argv[5]))
    names = subtask_names(root, episode)
    f_big, f_small = font(18), font(14)
    out.parent.mkdir(parents=True, exist_ok=True)
    box = av.open(str(out), "w")
    stream = None
    for i in range(lo, hi):
        s = ds[i]
        tiles = [Image.fromarray((s[k].permute(1, 2, 0).numpy() * 255).astype(np.uint8)) for k in CAMS]
        w, h = tiles[0].size
        canvas = Image.new("RGB", (3 * w, h + 56), "#1a1a19")
        for j, (t, lab) in enumerate(zip(tiles, LABELS)):
            canvas.paste(t, (j * w, 56))
            ImageDraw.Draw(canvas).text((j * w + 6, 60), lab, font=f_small, fill="#ffffff")
        a = s["action"].numpy()
        stage = int(s["dx.stage_id"].item())
        sub = names[stage] if names and 0 <= stage < len(names) else f"stage {stage}"
        d = ImageDraw.Draw(canvas)
        d.text((8, 4), f"{s['task']}   |   {sub}", font=f_big, fill="#f0efec")
        d.text((8, 30), f"t {(i - ep_lo) / ds.fps:6.1f} s  frame {i - ep_lo}/{ep_hi - ep_lo - 1}   gripper R {a[14]:.0f} L {a[15]:.0f}"
                        f"   base v {a[22]:+.2f} m/s  w {a[23]:+.2f} rad/s", font=f_small, fill="#c3c2b7")
        if stream is None:
            stream = box.add_stream("libx264", rate=int(ds.fps))
            stream.width, stream.height, stream.pix_fmt = canvas.width, canvas.height, "yuv420p"
        for pkt in stream.encode(av.VideoFrame.from_image(canvas)):
            box.mux(pkt)
    for pkt in stream.encode():
        box.mux(pkt)
    box.close()
    print(f"[view] {out}: episode {episode}, {hi - lo} frames")


if __name__ == "__main__":
    main()
