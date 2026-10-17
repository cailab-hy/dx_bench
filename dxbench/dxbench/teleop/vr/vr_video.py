# Ported from cailab-hy/lightwheel_rby1_transfer scripts/vr/vr_video.py @1178454 (same lab, internal use; DECISIONS D-2).
"""Robot camera video for the headset panel: compose, label, JPEG-encode in a background thread.

Panel layout (camera images as recorded, each scaled to the same height):

    +--------------- status strip: line 1 recording / clutch / gripper, line 2 task success ---------+
    | left wrist (fingers at bottom) |          head camera          | right wrist (fingers at bottom) |

Encoding runs in its own thread with a one-frame slot, so the simulation loop only hands over the
arrays it already has (no extra rendering); if encoding falls behind, older frames are dropped.
Uses Pillow (libjpeg-turbo) only.
"""

import threading

import numpy as np
from PIL import Image, ImageDraw, ImageFont

LAYOUT = ("left_hand", "first_person", "right_hand")
STRIP = 84  # status strip height (px): two text lines
COLORS = {"rec": (190, 20, 20), "pause": (150, 110, 0), "ok": (20, 120, 40), "idle": (35, 35, 40)}


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


FONT = _font(26)


def compose(images, text="", tone="idle", height=480):
    """images: camera name -> HxWx3 uint8 (missing cameras are skipped). Returns a PIL image."""
    tiles = []
    for name in LAYOUT:
        img = images.get(name)
        if img is None:
            continue
        im = Image.fromarray(np.ascontiguousarray(img))
        if im.height != height:
            im = im.resize((round(im.width * height / im.height), height), Image.BILINEAR)
        tiles.append(im)
    width = sum(t.width for t in tiles) or 640
    out = Image.new("RGB", (width, height + STRIP), COLORS.get(tone, COLORS["idle"]))
    x = 0
    for t in tiles:
        out.paste(t, (x, STRIP))
        x += t.width
    draw = ImageDraw.Draw(out)
    for sep in np.cumsum([t.width for t in tiles])[:-1]:
        draw.line([(int(sep), STRIP), (int(sep), STRIP + height)], fill=(0, 0, 0), width=3)
    for i, line in enumerate(text.split("\n")[:2]):
        draw.text((12, 21 + 42 * i), line, fill=(255, 255, 255), font=FONT, anchor="lm")
    return out


def encode(image, quality=80):
    import io

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


class VideoStreamer:
    def __init__(self, server, quality=80, height=480):
        self.server, self.quality, self.height = server, quality, height
        self.cv = threading.Condition()
        self.slot = None
        self.running = True
        self.encoded = 0
        self.thread = threading.Thread(target=self._run, name="vr-video-encoder", daemon=True)
        self.thread.start()

    def submit(self, images, text="", tone="idle"):
        """Hand over the newest camera images (the arrays are copied by compose in the thread; do not
        modify them afterwards)."""
        with self.cv:
            self.slot = (images, text, tone)
            self.cv.notify()

    def _run(self):
        while True:
            with self.cv:
                self.cv.wait_for(lambda: self.slot is not None or not self.running)
                if not self.running:
                    return
                images, text, tone = self.slot
                self.slot = None
            try:
                self.server.publish(encode(compose(images, text, tone, self.height), self.quality))
                self.encoded += 1
            except Exception as e:  # never take the collector down for a video frame
                print(f"[vr] video frame skipped: {e!r}", flush=True)

    def close(self):
        with self.cv:
            self.running = False
            self.cv.notify()
        self.thread.join(timeout=2)


def test_images(k):
    """Moving test picture in the RB-Y1 camera sizes (head 480x640, wrists 640x480 portrait)."""
    def card(h, w, label, hue, wrist=True):
        y, x = np.mgrid[0:h, 0:w]
        img = np.zeros((h, w, 3), np.uint8)
        img[..., 0] = (x * 255 // w)
        img[..., 1] = (y * 255 // h)
        img[..., 2] = hue
        c = (k * 7) % w
        img[:, max(c - 6, 0):c + 6] = 255  # moving bar: shows the frame rate and latency
        im = Image.fromarray(img)
        d = ImageDraw.Draw(im)
        d.text((w // 2, h // 2), label, fill=(255, 255, 255), font=_font(40), anchor="mm")
        if wrist:
            d.text((w // 2, h - 30), "fingers at the bottom", fill=(255, 255, 0), font=_font(24), anchor="mm")
        return np.asarray(im)

    return {"left_hand": card(640, 480, "LEFT wrist", 160), "first_person": card(480, 640, "HEAD", 60, wrist=False),
            "right_hand": card(640, 480, "RIGHT wrist", 220)}
