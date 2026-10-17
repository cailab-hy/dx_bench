# Ported from cailab-hy/lightwheel_rby1_transfer scripts/vr/vr_server.py @1178454 (same lab, internal use; DECISIONS D-2).
"""Quest 2 WebXR input + camera-video server for the collector (stdlib only; threads inside Isaac Sim).

The headset opens http://localhost:PORT (reached through `adb reverse tcp:PORT tcp:PORT`; localhost is
a secure context, so WebXR works without certificates). Over one WebSocket:

  headset -> PC  one report per XR frame (~90 Hz), text JSON:
                 {"kind": "frame", "v": PAGE_VERSION, "fps", "hands": {side: {...}}, "head": {...}, "vseq"}
                 (same report format as quest_check.py; see vr_teleop.py). "vseq" = last video frame shown.
  PC -> headset  camera video, binary: 4-byte big-endian frame number + JPEG (see vr_video.py).
                 The page draws it on a panel fixed in VR space in front of the operator.

`take()` returns the newest report, with every button value replaced by its maximum since the previous
take(), so a press shorter than one control step is not lost. `publish(jpeg)` sends a video frame to every
connected page; each connection has its own writer thread that always sends the newest frame only, so a
slow page drops frames instead of building up delay.

  python3 scripts/vr/vr_server.py [port] [--test-video]   # standalone (no Isaac Sim): receive rate,
                                                           # optional moving test picture for the panel
"""

import base64, hashlib, json, os, socket, struct, subprocess, sys, threading, time
from collections import OrderedDict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE_VERSION = 4
WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RB-Y1 VR teleop</title>
<style>body{font:22px sans-serif;margin:24px;background:#111;color:#eee}button{font-size:30px;padding:18px 36px}
.ok{color:#6f6}.bad{color:#f66}canvas{width:100%;max-width:1360px;background:#000;display:block;margin-top:12px}
p.help{font-size:18px;color:#bbb}</style></head><body>
<h2>RB-Y1 VR teleoperation</h2>
<div id="env"></div><p><button id="enter" disabled>Enter VR</button></p><div id="state"></div><div id="video"></div>
<canvas id="preview" width="1360" height="600"></canvas>
<p class="help">In VR the robot camera panel (left wrist | head | right wrist) floats in front of you.
Hold Y for 1 s: recenter (robot forward and the panel). Right stick up/down: panel distance, left/right: panel size.</p>
<script>
const PAGE_VERSION = __PAGE_VERSION__;
const env = document.getElementById('env'), btn = document.getElementById('enter');
const stateEl = document.getElementById('state'), videoEl = document.getElementById('video');
function line(label, ok, extra) { env.innerHTML += `<div>${label}: <b class="${ok ? 'ok' : 'bad'}">${ok ? 'OK' : 'NO'}</b> ${extra || ''}</div>`; }

// ---------------------------------------------------------------- WebSocket (reports out, video in)
let ws = null, latestBuf = null, decoding = false, pending = null, shownSeq = 0;
const stats = {received: 0, shown: 0, errors: 0};
function send(obj) { if (ws && ws.readyState === 1 && ws.bufferedAmount < 65536) ws.send(JSON.stringify(obj)); }
async function decodeLatest() {
  decoding = true;
  while (latestBuf) {
    const buf = latestBuf; latestBuf = null;
    const seq = new DataView(buf).getUint32(0);
    try {
      const bmp = await createImageBitmap(new Blob([new Uint8Array(buf, 4)], {type: 'image/jpeg'}));
      if (pending) pending.bmp.close();
      pending = {bmp, seq};
    } catch (e) { stats.errors++; }
  }
  decoding = false;
}
function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => { stateEl.innerHTML = '<b class="ok">connected to the collector</b>';
    send({kind: 'hello', v: PAGE_VERSION, secure: window.isSecureContext, xr: !!navigator.xr, ua: navigator.userAgent}); };
  ws.onclose = () => { stateEl.innerHTML = '<b class="bad">collector not reachable - retrying</b>'; setTimeout(connect, 1000); };
  ws.onmessage = ev => {
    if (typeof ev.data === 'string') return;
    stats.received++; latestBuf = ev.data;   // keep only the newest frame
    if (!decoding) decodeLatest();
  };
}
connect();
setInterval(() => { videoEl.textContent = `video: ${stats.received} frames received, ${stats.shown} shown, ${stats.errors} decode errors`; }, 1000);

// ---------------------------------------------------------------- WebGL panel
const canvas = document.getElementById('preview');
const gl = canvas.getContext('webgl', {xrCompatible: true, antialias: true});
function shader(type, src) {
  const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}
const prog = gl.createProgram();
gl.attachShader(prog, shader(gl.VERTEX_SHADER,
  'attribute vec2 a; uniform mat4 mvp; varying vec2 uv;' +
  'void main() { uv = vec2(a.x * 0.5 + 0.5, 0.5 - a.y * 0.5); gl_Position = mvp * vec4(a, 0.0, 1.0); }'));
gl.attachShader(prog, shader(gl.FRAGMENT_SHADER,
  'precision mediump float; uniform sampler2D tex; varying vec2 uv; void main() { gl_FragColor = texture2D(tex, uv); }'));
gl.linkProgram(prog);
const aLoc = gl.getAttribLocation(prog, 'a'), mvpLoc = gl.getUniformLocation(prog, 'mvp');
const quad = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, quad);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
const tex = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, tex);
for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR],
                      [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, k, v);
gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([40, 40, 40, 255]));
let aspect = 1360 / 524;
function upload() {
  if (!pending) return;
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, pending.bmp);
  aspect = pending.bmp.width / pending.bmp.height; shownSeq = pending.seq; stats.shown++;
  pending.bmp.close(); pending = null;
}
function drawPanel(mvp) {
  gl.useProgram(prog); gl.bindBuffer(gl.ARRAY_BUFFER, quad); gl.enableVertexAttribArray(aLoc);
  gl.vertexAttribPointer(aLoc, 2, gl.FLOAT, false, 0, 0);
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.uniformMatrix4fv(mvpLoc, false, mvp); gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
}
// column-major 4x4 helpers
function mul(a, b) {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) { let s = 0; for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k]; o[c * 4 + r] = s; }
  return o;
}
function model(p, yaw, w, h) {   // translate * rotate about +y * scale(w/2, h/2)
  const c = Math.cos(yaw), s = Math.sin(yaw);
  return new Float32Array([c * w / 2, 0, -s * w / 2, 0, 0, h / 2, 0, 0, s, 0, c, 0, p[0], p[1], p[2], 1]);
}
function perspective(fovy, asp, n, f) {
  const t = 1 / Math.tan(fovy / 2);
  return new Float32Array([t / asp, 0, 0, 0, 0, t, 0, 0, 0, 0, (f + n) / (n - f), -1, 0, 0, 2 * f * n / (n - f), 0]);
}
// panel placement (VR): in front of the head when entering VR or on a 1 s Y press; stick adjusts it
let panel = {dist: 1.1, width: 1.5};
try { Object.assign(panel, JSON.parse(localStorage.getItem('rby1_panel') || '{}')); } catch (e) {}
let anchor = null, yDown = null, yUsed = false;
function placePanel(tf) {
  const q = tf.orientation, P = tf.position;
  const fx = -2 * (q.x * q.z + q.w * q.y), fz = -(1 - 2 * (q.x * q.x + q.y * q.y));
  anchor = {yaw: Math.atan2(-fx, -fz), base: [P.x, P.y, P.z]};
}
function panelModel() {
  const p = [anchor.base[0] - Math.sin(anchor.yaw) * panel.dist, anchor.base[1] - 0.12, anchor.base[2] - Math.cos(anchor.yaw) * panel.dist];
  return model(p, anchor.yaw, panel.width, panel.width / aspect);
}
function controls(hands, t, viewer, dt) {
  const left = hands.left, right = hands.right;
  const y = left && left.buttons.length > 5 && left.buttons[5] > 0.5;
  if (y && yDown === null) { yDown = t; yUsed = false; }
  if (!y) yDown = null;
  if (y && !yUsed && t - yDown >= 1000 && viewer) { placePanel(viewer.transform); yUsed = true; }
  if (right && right.axes.length >= 4) {
    const ax = right.axes[2], ay = right.axes[3];
    let changed = false;
    if (Math.abs(ay) > 0.25) { panel.dist = Math.min(3.0, Math.max(0.5, panel.dist - ay * 0.6 * dt)); changed = true; }
    if (Math.abs(ax) > 0.25) { panel.width = Math.min(4.0, Math.max(0.5, panel.width * Math.exp(ax * 0.6 * dt))); changed = true; }
    if (changed) try { localStorage.setItem('rby1_panel', JSON.stringify(panel)); } catch (e) {}
  }
}
// 2D preview (before entering VR): same texture and panel shader
let inXR = false;
function preview() {
  if (!inXR) {
    upload();
    gl.bindFramebuffer(gl.FRAMEBUFFER, null); gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0.05, 0.05, 0.05, 1); gl.clear(gl.COLOR_BUFFER_BIT);
    const w = Math.min(2.2, 2 * 0.95 * Math.tan(Math.PI / 6) * aspect);
    drawPanel(mul(perspective(Math.PI / 3, canvas.width / canvas.height, 0.1, 10), model([0, 0, -1], 0, w, w / aspect)));
  }
  requestAnimationFrame(preview);
}
requestAnimationFrame(preview);

// ---------------------------------------------------------------- WebXR session
line('secure context (localhost or https)', window.isSecureContext, location.origin);
line('navigator.xr', !!navigator.xr);
if (navigator.xr) navigator.xr.isSessionSupported('immersive-vr').then(ok => { line('immersive-vr supported', ok); btn.disabled = !ok; });
btn.onclick = async () => {
  const session = await navigator.xr.requestSession('immersive-vr', {optionalFeatures: ['local-floor']});
  await gl.makeXRCompatible();
  session.updateRenderState({baseLayer: new XRWebGLLayer(session, gl)});
  let space; try { space = await session.requestReferenceSpace('local-floor'); } catch (e) { space = await session.requestReferenceSpace('local'); }
  inXR = true; anchor = null;
  send({kind: 'session', v: PAGE_VERSION, started: true});
  session.addEventListener('end', () => { inXR = false; send({kind: 'session', v: PAGE_VERSION, started: false}); });
  let last = 0, frames = 0, fps = 0, lastT = 0;
  session.requestAnimationFrame(function onFrame(t, frame) {
    session.requestAnimationFrame(onFrame); frames++;
    const dt = lastT ? Math.min(0.1, (t - lastT) / 1000) : 0; lastT = t;
    if (t - last >= 1000) { fps = frames * 1000 / (t - last); frames = 0; last = t; }
    upload();
    const hands = {};
    for (const src of session.inputSources) {
      const pose = src.gripSpace && frame.getPose(src.gripSpace, space), gp = src.gamepad, tf = pose && pose.transform;
      hands[src.handedness] = {profile: (src.profiles || [])[0] || '', tracked: !!pose, hand_tracking: !!src.hand,
        position: tf ? [tf.position.x, tf.position.y, tf.position.z] : null,
        orientation: tf ? [tf.orientation.w, tf.orientation.x, tf.orientation.y, tf.orientation.z] : null,
        buttons: gp ? gp.buttons.map(b => b.value) : [], axes: gp ? Array.from(gp.axes) : []};
    }
    const viewer = frame.getViewerPose(space), vt = viewer && viewer.transform;
    if (viewer && !anchor) placePanel(vt);
    controls(hands, t, viewer, dt);
    const layer = session.renderState.baseLayer;
    gl.bindFramebuffer(gl.FRAMEBUFFER, layer.framebuffer);
    gl.clearColor(0.05, 0.08, 0.12, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    if (viewer && anchor) {
      const M = panelModel();
      for (const view of viewer.views) {
        const vp = layer.getViewport(view); gl.viewport(vp.x, vp.y, vp.width, vp.height);
        drawPanel(mul(view.projectionMatrix, mul(view.transform.inverse.matrix, M)));
      }
    }
    send({kind: 'frame', v: PAGE_VERSION, t, fps, hands, vseq: shownSeq,
          head: vt ? {position: [vt.position.x, vt.position.y, vt.position.z], orientation: [vt.orientation.w, vt.orientation.x, vt.orientation.y, vt.orientation.z]} : null});
  });
};
</script></body></html>""".replace("__PAGE_VERSION__", str(PAGE_VERSION))


# ---------------------------------------------------------------- websocket (RFC 6455)
def ws_accept(key):
    return base64.b64encode(hashlib.sha1((key + WS_GUID).encode()).digest()).decode()


def ws_read(rfile):
    """One frame -> (opcode, payload bytes); raises EOFError when the peer is gone."""
    head = rfile.read(2)
    if len(head) < 2:
        raise EOFError
    opcode, n = head[0] & 0x0F, head[1] & 0x7F
    if n == 126:
        n = struct.unpack(">H", rfile.read(2))[0]
    elif n == 127:
        n = struct.unpack(">Q", rfile.read(8))[0]
    mask = rfile.read(4) if head[1] & 0x80 else b"\0\0\0\0"
    data = rfile.read(n)
    if len(data) < n:
        raise EOFError
    if mask != b"\0\0\0\0":
        data = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
    return opcode, data


def ws_frame(opcode, data):
    n = len(data)
    head = bytes([0x80 | opcode]) + (bytes([n]) if n < 126 else bytes([126]) + struct.pack(">H", n) if n < 65536 else bytes([127]) + struct.pack(">Q", n))
    return head + data


# ---------------------------------------------------------------- server
class VRServer:
    def __init__(self, port=8012, record=None, log=print):
        self.port, self.log = port, log
        self.lock = threading.Lock()
        self.latest = None  # newest frame report
        self.latched = {}  # side -> per-button max since the last take()
        self.times = deque(maxlen=400)  # receive times, for the rate
        self.reports = 0
        self.clients = 0
        self.session = False
        self.old_page_warned = False
        self.record = open(record, "a") if record else None
        # video: newest frame + sequence number; writer threads wait on video_cv
        self.video_cv = threading.Condition()
        self.video_frame, self.video_seq = None, 0
        self.video_sent = OrderedDict()  # seq -> publish time (for the display latency)
        self.video_times = deque(maxlen=100)
        self.video_bytes = deque(maxlen=30)
        self.video_shown_seq, self.video_latency = 0, None
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_GET(self):
                if self.path.startswith("/ws") and self.headers.get("Upgrade", "").lower() == "websocket":
                    return server._websocket(self)
                body = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):  # quest_check.py-style pages
                data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                self.send_response(204)
                self.send_header("Content-Length", "0")
                self.end_headers()
                server._receive(data)

        ThreadingHTTPServer.allow_reuse_address = True
        ThreadingHTTPServer.daemon_threads = True
        self.httpd = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, name="vr-server", daemon=True)

    def start(self):
        self.thread.start()
        return self

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        with self.video_cv:
            self.video_cv.notify_all()
        if self.record:
            self.record.close()

    # -------------------------------------------------- per connection
    def _websocket(self, h):
        h.send_response(101, "Switching Protocols")
        h.send_header("Upgrade", "websocket")
        h.send_header("Connection", "Upgrade")
        h.send_header("Sec-WebSocket-Accept", ws_accept(h.headers["Sec-WebSocket-Key"]))
        h.end_headers()
        h.wfile.flush()
        sock = h.connection
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        conn = {"sock": sock, "lock": threading.Lock(), "alive": True}
        writer = threading.Thread(target=self._video_writer, args=(conn,), name="vr-video-writer", daemon=True)
        writer.start()
        with self.lock:
            self.clients += 1
        self.log("[vr] headset page connected")
        try:
            while True:
                opcode, data = ws_read(h.rfile)
                if opcode == 8:  # close
                    break
                if opcode == 9:  # ping
                    with conn["lock"]:
                        sock.sendall(ws_frame(10, data))
                elif opcode == 1:
                    self._receive(json.loads(data))
        except (EOFError, ConnectionError, OSError, ValueError):
            pass
        finally:
            conn["alive"] = False
            with self.video_cv:
                self.video_cv.notify_all()
            with self.lock:
                self.clients -= 1
                if self.clients == 0:
                    self.session = False
            h.close_connection = True
            self.log("[vr] headset page disconnected")

    def _video_writer(self, conn):
        sent = 0
        while conn["alive"]:
            with self.video_cv:
                self.video_cv.wait_for(lambda: not conn["alive"] or self.video_seq != sent, timeout=1.0)
                if not conn["alive"] or self.video_seq == sent:
                    continue
                frame, sent = self.video_frame, self.video_seq
            try:
                with conn["lock"]:
                    conn["sock"].sendall(frame)
            except OSError:
                conn["alive"] = False

    def _receive(self, data):
        now = time.monotonic()
        data["recv_t"] = now
        if self.record:
            with self.lock:
                self.record.write(json.dumps(data) + "\n")
        kind = data.get("kind")
        if data.get("v") != PAGE_VERSION and not self.old_page_warned:
            self.old_page_warned = True
            self.log(f"[vr] the headset runs an old page (v={data.get('v')}): reload http://localhost:{self.port} in the headset")
        if kind == "session":
            self.session = bool(data.get("started"))
            self.log(f"[vr] VR session {'started' if self.session else 'ended'}")
        if kind != "frame":
            return
        with self.lock:
            self.latest = data
            self.reports += 1
            self.times.append(now)
            for side, h in (data.get("hands") or {}).items():
                values = h.get("buttons") or []
                old = self.latched.get(side, [])
                self.latched[side] = [max(v, old[i]) if i < len(old) else v for i, v in enumerate(values)]
            seq = data.get("vseq") or 0
            if seq != self.video_shown_seq:
                self.video_shown_seq = seq
                t = self.video_sent.get(seq)
                if t is not None:  # published -> drawn in the headset -> report back
                    lat = now - t
                    self.video_latency = lat if self.video_latency is None else 0.8 * self.video_latency + 0.2 * lat

    # -------------------------------------------------- API
    def publish(self, jpeg):
        """Send one JPEG video frame to every connected page (newest frame wins)."""
        with self.video_cv:
            self.video_seq = self.video_seq % 0xFFFFFFFF + 1
            self.video_frame = ws_frame(2, struct.pack(">I", self.video_seq) + jpeg)
            now = time.monotonic()
            self.video_sent[self.video_seq] = now
            while len(self.video_sent) > 256:
                self.video_sent.popitem(last=False)
            self.video_times.append(now)
            self.video_bytes.append(len(jpeg))
            self.video_cv.notify_all()

    def take(self):
        """Newest frame report (button values = max since the previous take), or None."""
        with self.lock:
            if self.latest is None:
                return None
            report = dict(self.latest)
            report["hands"] = {side: dict(h, buttons=self.latched.get(side, h.get("buttons") or []))
                               for side, h in (self.latest.get("hands") or {}).items()}
            self.latched = {}
            return report

    def status(self):
        with self.lock:
            now = time.monotonic()
            recent = [t for t in self.times if now - t <= 1.0]
            video = [t for t in self.video_times if now - t <= 1.0]
            return {"clients": self.clients, "session": self.session, "rate_hz": len(recent),
                    "age_s": now - self.times[-1] if self.times else float("inf"), "reports": self.reports,
                    "video_fps": len(video), "video_kb": sum(self.video_bytes) / max(len(self.video_bytes), 1) / 1024,
                    "video_latency_ms": None if self.video_latency is None else 1000 * self.video_latency}


class WSClient:
    """Minimal WebSocket client (tests / scripted operator): sends JSON text frames like the headset page;
    with read_video=True a thread receives video frames (count, newest JPEG, newest frame number)."""

    def __init__(self, port, host="127.0.0.1", read_video=False):
        self.sock = socket.create_connection((host, port), timeout=5)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall((f"GET /ws HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                           f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
        reply = b""
        while b"\r\n\r\n" not in reply:
            chunk = self.sock.recv(1)
            if not chunk:
                raise ConnectionError("no WebSocket handshake reply")
            reply += chunk
        if b" 101 " not in reply.split(b"\r\n")[0] or ws_accept(key).encode() not in reply:
            raise ConnectionError(f"bad WebSocket handshake: {reply[:200]!r}")
        self.video_frames, self.video_jpeg, self.video_seq = 0, None, 0
        self.lock = threading.Lock()
        if read_video:
            self.sock.settimeout(None)
            threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        rfile = self.sock.makefile("rb")
        try:
            while True:
                opcode, data = ws_read(rfile)
                if opcode == 2:
                    self.video_frames += 1
                    self.video_seq, self.video_jpeg = struct.unpack(">I", data[:4])[0], data[4:]
                elif opcode == 8:
                    break
        except (EOFError, OSError):
            pass

    def send(self, obj):
        data, mask = json.dumps(obj).encode(), os.urandom(4)
        frame = ws_frame(1, bytes(b ^ mask[i % 4] for i, b in enumerate(data)))
        i = 2 if len(data) < 126 else 4 if len(data) < 65536 else 10
        with self.lock:
            self.sock.sendall(frame[:1] + bytes([frame[1] | 0x80]) + frame[2:i] + mask + frame[i:])

    def close(self):
        try:
            with self.lock:
                self.sock.sendall(bytes([0x88, 0x80]) + os.urandom(4))
        except OSError:
            pass
        finally:
            self.sock.close()


def open_in_headset(port, log=print):
    """Open a fresh tab on the headset browser (adb VIEW intent); False if adb is unavailable."""
    url = f"http://localhost:{port}/?v={int(time.time())}"
    try:
        subprocess.run(["adb", "reverse", f"tcp:{port}", f"tcp:{port}"], check=True, capture_output=True, timeout=10)
        subprocess.run(["adb", "shell", "am", "start", "-a", "android.intent.action.VIEW", "-d", url, "com.oculus.browser"],
                       check=True, capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as e:
        log(f"[vr] could not open the page over adb ({e.__class__.__name__}); open http://localhost:{port} in the headset")
        return False
    log(f"[vr] opened {url} in the headset browser")
    return True


if __name__ == "__main__":
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    port = int(args[0]) if args else 8012
    srv = VRServer(port).start()
    print(f"VR server on http://localhost:{port}  (Ctrl+C to stop)", flush=True)
    streamer = None
    if "--test-video" in sys.argv:
        from dxbench.teleop.vr.vr_video import VideoStreamer, test_images

        streamer = VideoStreamer(srv)
    if os.environ.get("VR_OPEN", "1") == "1":
        open_in_headset(port)
    try:
        k, last = 0, 0.0
        while True:
            time.sleep(1 / 15)
            k += 1
            if streamer is not None and srv.status()["clients"]:
                streamer.submit(test_images(k), f"TEST PICTURE {k / 15:6.1f} s | hold Y 1 s: recenter panel | right stick: distance / size")
            if time.monotonic() - last >= 1:
                last = time.monotonic()
                st = srv.status()
                r = srv.take()
                hands = (r or {}).get("hands", {})
                lat = f"{st['video_latency_ms']:.0f} ms" if st["video_latency_ms"] is not None else "-"
                print(f"clients={st['clients']} session={st['session']} reports={st['rate_hz']}/s video={st['video_fps']} fps "
                      f"{st['video_kb']:.0f} KB latency={lat}  "
                      + "  ".join(f"{s}:{'ok' if (hands.get(s) or {}).get('tracked') else '--'}" for s in ("left", "right")), flush=True)
    except KeyboardInterrupt:
        if streamer is not None:
            streamer.close()
        srv.close()
