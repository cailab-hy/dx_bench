# Ported from cailab-hy/lightwheel_rby1_transfer scripts/vr/test_vr_server.py @1178454 (same lab, internal use; DECISIONS D-2).
"""VR server / video tests (no Isaac Sim, no headset):  conda run -n lerobot-arena python -m pytest -q scripts/vr"""

import io
import time

from PIL import Image

from dxbench.teleop.vr.vr_server import PAGE, PAGE_VERSION, VRServer, WSClient
from dxbench.teleop.vr.vr_video import STRIP, VideoStreamer, compose
from dxbench.teleop.vr.vr_video import test_images as pattern

PORT = 18191


def wait(cond, timeout=3.0):
    t0 = time.monotonic()
    while not cond() and time.monotonic() - t0 < timeout:
        time.sleep(0.005)
    return cond()


def test_compose_layout():
    im = compose(pattern(0), "IDLE", "idle")
    # wrists 480x640 portrait scaled to height 480 (360 wide) + head 640x480, plus the status strip
    assert im.size == (360 + 640 + 360, 480 + STRIP)


def test_page_is_current():
    assert f"PAGE_VERSION = {PAGE_VERSION};" in PAGE and "__PAGE_VERSION__" not in PAGE


def test_reports_buttons_and_video_roundtrip():
    srv = VRServer(PORT, log=lambda m: None).start()
    streamer = VideoStreamer(srv)
    client = WSClient(PORT, read_video=True)
    try:
        assert wait(lambda: srv.status()["clients"] == 1)
        hand = {"tracked": True, "position": [0, 1, 0], "orientation": [1, 0, 0, 0], "buttons": [0, 0, 0, 0, 1, 0]}
        client.send({"kind": "frame", "v": PAGE_VERSION, "hands": {"right": hand}, "vseq": 0})
        client.send({"kind": "frame", "v": PAGE_VERSION, "hands": {"right": dict(hand, buttons=[0] * 6)}, "vseq": 0})
        assert wait(lambda: srv.status()["reports"] == 2)
        assert srv.take()["hands"]["right"]["buttons"][4] == 1  # short A press latched
        for k in range(5):
            streamer.submit(pattern(k), f"frame {k}")
            time.sleep(0.03)
        assert wait(lambda: client.video_frames >= 1 and client.video_seq == srv.video_seq)
        im = Image.open(io.BytesIO(client.video_jpeg))
        assert im.format == "JPEG" and im.size == (1360, 480 + STRIP)
        # the page reports the frame it drew -> display latency
        client.send({"kind": "frame", "v": PAGE_VERSION, "hands": {}, "vseq": client.video_seq})
        assert wait(lambda: srv.status()["video_latency_ms"] is not None)
        assert 0 <= srv.status()["video_latency_ms"] < 1000
    finally:
        client.close()
        streamer.close()
        srv.close()
