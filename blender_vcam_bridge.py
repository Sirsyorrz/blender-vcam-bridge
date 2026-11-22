# SPDX-License-Identifier: GPL-3.0-or-later
"""
VCam Bridge - drive a Blender camera from an Android phone.

* The phone sends its pose (orientation + position) over TCP at ~60 Hz.
* Blender renders the camera's view into an offscreen buffer and streams it back
  to the phone as JPEG frames (PNG if Pillow is not installed).
* The phone can also change focal length, depth of field, start/stop keyframe
  recording, switch camera, recenter, level the horizon and smooth the motion.

Install:  Edit > Preferences > Add-ons > (v) > Install from Disk... > pick this file.
Use:      3D Viewport > press N > "VCam" tab > Start Server.
"""

bl_info = {
    "name": "VCam Bridge (Phone Virtual Camera)",
    "author": "VCam",
    "version": (1, 1, 0),
    "blender": (3, 6, 0),
    "location": "3D Viewport > Sidebar > VCam",
    "description": "Drive a Blender camera with an Android phone and stream the camera view back live",
    "category": "3D View",
}

import collections
import importlib
import io
import json
import math
import os
import queue
import socket
import struct
import subprocess
import sys
import threading
import time
import traceback
import zlib

import bpy
import gpu
import numpy as np
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty
from bpy_extras.view3d_utils import location_3d_to_region_2d
from mathutils import Matrix, Quaternion, Vector

# --------------------------------------------------------------------------------------
# Protocol
# --------------------------------------------------------------------------------------
# Every message:  [1 byte type][4 bytes big-endian payload length][payload]
#   type 1  JSON (both directions, UTF-8)
#   type 2  pose  (phone -> Blender): 7 big-endian float32: px py pz qw qx qy qz
#                 position in Blender world axes (metres), quaternion = camera orientation
#                 in Blender world axes (camera looks down its local -Z, +Y is up)
#   type 3  frame (Blender -> phone): JPEG or PNG bytes
PROTOCOL_VERSION = 1
DEFAULT_PORT = 9876
MSG_JSON = 1
MSG_POSE = 2
MSG_FRAME = 3
DISCOVER_MAGIC = b"BLENDER_VCAM_DISCOVER"
TICK_SECONDS = 1.0 / 60.0
MAX_INCOMING = 1 << 20

ROT_PATHS = {"QUATERNION": "rotation_quaternion", "AXIS_ANGLE": "rotation_axis_angle"}

# Engines drawn by Blender's own draw manager; GPUOffScreen.draw_view3d renders these fine. Python engines
# (Cycles, Deadlock (game), ...) only draw through their view_draw, so they are captured there instead.
DRAW_MANAGER_ENGINES = {"BLENDER_EEVEE", "BLENDER_EEVEE_NEXT", "BLENDER_WORKBENCH"}

PHONE_DIRECTIONS = [
    ("FORWARD", "Phone forward", ""),
    ("BACK", "Phone back", ""),
    ("RIGHT", "Phone right", ""),
    ("LEFT", "Phone left", ""),
    ("UP", "Phone up", ""),
    ("DOWN", "Phone down", ""),
    ("OFF", "Off", ""),
]


class OneEuroFilter:
    """Casiez et al. 2012: heavy smoothing when nearly still (kills tracking jitter), little when moving
    fast (no lag). min_cutoff in Hz, beta per m/s of speed."""

    D_CUTOFF = 1.0

    def __init__(self):
        self.reset()

    def reset(self):
        self.x = None
        self.dx = Vector((0.0, 0.0, 0.0))
        self.t = 0.0

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x, t, min_cutoff, beta):
        if self.x is None:
            self.x, self.t = x.copy(), t
            self.dx = Vector((0.0, 0.0, 0.0))
            return x.copy()
        dt = max(t - self.t, 1e-3)
        self.t = t
        self.dx = self.dx.lerp((x - self.x) / dt, self._alpha(self.D_CUTOFF, dt))
        cutoff = max(1e-3, min_cutoff + beta * self.dx.length)
        self.x = self.x.lerp(x, self._alpha(cutoff, dt))
        return self.x.copy()


def phone_component(local, direction):
    """local: movement in the phone's heading frame (x right, y forward, z up)."""
    return {
        "FORWARD": local.y,
        "BACK": -local.y,
        "RIGHT": local.x,
        "LEFT": -local.x,
        "UP": local.z,
        "DOWN": -local.z,
    }.get(direction, 0.0)


def pack_msg(msg_type, payload):
    return struct.pack(">BI", msg_type, len(payload)) + payload


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# --------------------------------------------------------------------------------------
# Image encoding (Pillow when available, dependency-free PNG otherwise)
# --------------------------------------------------------------------------------------
Image = None


def _modules_dir():
    try:
        return bpy.utils.user_resource("SCRIPTS", path="modules", create=True)
    except Exception:
        return None


def _ensure_modules_on_path():
    d = _modules_dir()
    if d and d not in sys.path:
        sys.path.append(d)


def _try_import_pil():
    global Image
    try:
        from PIL import Image as _Image

        Image = _Image
        return True
    except Exception:
        Image = None
        return False


def _png_chunk(tag, data):
    body = tag + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def encode_png(arr):
    h, w = arr.shape[:2]
    raw = np.zeros((h, w * 3 + 1), dtype=np.uint8)
    raw[:, 1:] = np.ascontiguousarray(arr[:, :, :3]).reshape(h, w * 3)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(raw.tobytes(), 1))
        + _png_chunk(b"IEND", b"")
    )


def linear_to_display(arr, exposure, gamma):
    """Scene-linear float RGBA -> sRGB uint8. Approximates Blender's view transform as Standard."""
    c = np.clip(arr[:, :, :3] * (2.0 ** exposure), 0.0, 1.0)
    c = np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)
    if abs(gamma - 1.0) > 1e-4:
        c = np.power(c, 1.0 / gamma)
    out = np.empty(arr.shape[:2] + (4,), np.uint8)
    out[:, :, :3] = (c * 255.0 + 0.5).astype(np.uint8)
    out[:, :, 3] = 255
    return out


def encode_frame(arr, quality):
    """arr: uint8 array (h, w, 4), top row first. Returns JPEG (Pillow) or PNG bytes."""
    if Image is not None:
        im = Image.fromarray(arr).convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=int(quality))
        return buf.getvalue()
    return encode_png(arr)


class RawFrame:
    def __init__(self, arr, quality, linear=None):
        self.arr = arr
        self.quality = quality
        self.linear = linear  # (exposure, gamma) when arr is scene-linear float

    def encode(self):
        arr = self.arr if self.linear is None else linear_to_display(self.arr, *self.linear)
        return encode_frame(arr, self.quality)


def buffer_to_numpy(buf, w, h):
    """gpu.types.Buffer (UBYTE RGBA, bottom row first) -> contiguous uint8 (h, w, 4), top row first."""
    slow = False
    try:
        arr = np.frombuffer(buf, dtype=np.uint8)  # buffer protocol: zero-copy view
        if arr.size != w * h * 4:
            raise ValueError("unexpected buffer size")
    except Exception:
        slow = True
        buf.dimensions = w * h * 4
        arr = np.array(buf.to_list(), dtype=np.uint8)
    arr = arr.reshape(h, w, 4)
    return np.ascontiguousarray(arr[::-1]), slow  # the copy also detaches us from the GPU buffer


def float_buffer_to_numpy(buf, w, h):
    try:
        arr = np.frombuffer(buf, dtype=np.float32)
        if arr.size != w * h * 4:
            raise ValueError("unexpected buffer size")
    except Exception:
        buf.dimensions = w * h * 4
        arr = np.array(buf.to_list(), dtype=np.float32)
    return np.ascontiguousarray(arr.reshape(h, w, 4)[::-1])


def camera_frame_rect(scene, cam, region, rv3d):
    """The camera frame's (x0, y0, x1, y1) in region pixels, for a viewport looking through the camera."""
    xs, ys = [], []
    for corner in cam.data.view_frame(scene=scene):
        co = location_3d_to_region_2d(region, rv3d, cam.matrix_world @ corner)
        if co is None:
            return None
        xs.append(co.x)
        ys.append(co.y)
    if max(xs) - min(xs) < 2 or max(ys) - min(ys) < 2:
        return None
    return min(xs), min(ys), max(xs), max(ys)


_patched_engines = {}


def python_engine_classes(idname):
    return [
        cls
        for cls in bpy.types.RenderEngine.__subclasses__()
        if getattr(cls, "bl_idname", None) == idname and callable(getattr(cls, "view_draw", None))
    ]


def _wrap_view_draw(original):
    # Exactly (self, context, depsgraph): Blender sizes the call from co_argcount, so an extra defaulted
    # parameter is passed as NULL and crashes it.
    def view_draw(self, context, depsgraph):
        original(self, context, depsgraph)
        if BRIDGE.running:
            BRIDGE.capture_engine_view(original, self, context, depsgraph)

    return view_draw


def patch_engine(idname):
    """Wrap a Python render engine's view_draw so the rendered viewport can be streamed to the phone."""
    if idname in DRAW_MANAGER_ENGINES:
        return False
    classes = python_engine_classes(idname)
    for cls in classes:
        if cls in _patched_engines:
            continue
        _patched_engines[cls] = cls.view_draw
        cls.view_draw = _wrap_view_draw(cls.view_draw)
    return bool(classes)


def unpatch_engines():
    for cls, original in _patched_engines.items():
        cls.view_draw = original
    _patched_engines.clear()


# --------------------------------------------------------------------------------------
# Math helpers
# --------------------------------------------------------------------------------------
def heading_of(q):
    """Yaw (about world Z) of a camera orientation quaternion."""
    fwd = q @ Vector((0.0, 0.0, -1.0))
    if math.hypot(fwd.x, fwd.y) > 0.3:
        return math.atan2(fwd.y, fwd.x)
    up = q @ Vector((0.0, 1.0, 0.0))
    if fwd.z < 0.0:
        return math.atan2(up.y, up.x)
    return math.atan2(-up.y, -up.x)


def level_quaternion(q):
    """Remove roll so the horizon stays level."""
    fwd = q @ Vector((0.0, 0.0, -1.0))
    right = fwd.cross(Vector((0.0, 0.0, 1.0)))
    if right.length < 1e-4:
        return q
    right.normalize()
    up = right.cross(fwd)
    up.normalize()
    return Matrix((right, up, -fwd)).transposed().to_quaternion()


def set_world_matrix(obj, mat):
    """Write a world matrix to an object using its own rotation mode (keeps keyframing clean)."""
    if obj.parent is None and obj.rotation_mode not in ("AXIS_ANGLE",):
        loc, rot, _scale = mat.decompose()
        obj.location = loc
        if obj.rotation_mode == "QUATERNION":
            obj.rotation_quaternion = rot
        else:
            obj.rotation_euler = rot.to_euler(obj.rotation_mode, obj.rotation_euler)
    else:
        obj.matrix_world = mat


def local_ips():
    ips = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))  # no packet is sent; picks the LAN interface
        ips.append(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    try:
        for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
            if not ip.startswith("127.") and ip not in ips:
                ips.append(ip)
    except Exception:
        pass
    return ips or ["(unknown - check ipconfig / ifconfig)"]


# --------------------------------------------------------------------------------------
# Networking
# --------------------------------------------------------------------------------------
class Client:
    """One connected phone. Two daemon threads: reader and writer. Never touches bpy."""

    def __init__(self, bridge, sock, addr):
        self.bridge = bridge
        self.sock = sock
        self.addr = addr
        self.device = addr[0]
        self.alive = True
        self.last_rx = time.time()
        self._cv = threading.Condition()
        self._msgs = collections.deque()
        self._frame = None

    def start(self):
        threading.Thread(target=self._reader, daemon=True, name="vcam-rx").start()
        threading.Thread(target=self._writer, daemon=True, name="vcam-tx").start()

    def close(self):
        if not self.alive:
            return
        self.alive = False
        for fn in (lambda: self.sock.shutdown(socket.SHUT_RDWR), self.sock.close):
            try:
                fn()
            except OSError:
                pass
        with self._cv:
            self._cv.notify_all()
        self.bridge._client_closed(self)

    def send_json(self, obj):
        data = pack_msg(MSG_JSON, json.dumps(obj, separators=(",", ":")).encode("utf-8"))
        with self._cv:
            if len(self._msgs) > 200:
                self._msgs.popleft()
            self._msgs.append(data)
            self._cv.notify()

    def push_frame(self, frame):
        with self._cv:
            self._frame = frame  # newest wins: slow links drop frames instead of adding lag
            self._cv.notify()

    def frame_pending(self):
        return self._frame is not None

    def _recv_exact(self, n):
        buf = bytearray()
        while len(buf) < n:
            try:
                chunk = self.sock.recv(n - len(buf))
            except socket.timeout:
                if not self.alive or time.time() - self.last_rx > 12.0:
                    raise ConnectionError("phone timed out")
                continue
            if not chunk:
                raise ConnectionError("phone closed the connection")
            buf.extend(chunk)
        return bytes(buf)

    def _reader(self):
        try:
            while self.alive:
                msg_type, length = struct.unpack(">BI", self._recv_exact(5))
                if length > MAX_INCOMING:
                    raise ValueError("message too large")
                payload = self._recv_exact(length) if length else b""
                self.last_rx = time.time()
                if msg_type == MSG_POSE and length >= 28:
                    self.bridge.set_pose(struct.unpack(">7f", payload[:28]))
                elif msg_type == MSG_JSON:
                    try:
                        obj = json.loads(payload.decode("utf-8"))
                    except ValueError:
                        continue
                    if isinstance(obj, dict):
                        self.bridge.commands.put((self, obj))
        except Exception:
            pass
        finally:
            self.close()

    def _writer(self):
        try:
            while self.alive:
                with self._cv:
                    while self.alive and not self._msgs and self._frame is None:
                        self._cv.wait(0.5)
                    if not self.alive:
                        break
                    msgs = list(self._msgs)
                    self._msgs.clear()
                    frame, self._frame = self._frame, None
                for m in msgs:
                    self.sock.sendall(m)
                if frame is not None:
                    data = frame.encode()  # JPEG encoding happens here, off the main thread
                    self.sock.sendall(pack_msg(MSG_FRAME, data))
        except Exception:
            pass
        finally:
            self.close()


class Bridge:
    def __init__(self):
        self.running = False
        self.port = DEFAULT_PORT
        self.lock = threading.Lock()
        self.client = None
        self.pose = None
        self.commands = queue.Queue()
        self.srv = None
        self.udp = None
        self.ips = []
        self.hostname = ""
        self.blender_version = ""
        self.draw_handle = None
        self.offscreen = None
        self.off_size = None
        self.cap_offscreen = None
        self.cap_size = None
        self.engine_captures = False
        self.last_capture_t = 0.0
        self.capture_hint = ""
        self.last_frame_t = 0.0
        self.last_status_t = 0.0
        self.last_redraw_t = 0.0
        self.last_error = ""
        self.slow_readback = False
        self.reset_tracking()
        self.recording = False
        self.rec_start_frame = 1
        self.rec_start_time = 0.0
        self.rec_last_frame = None
        self.rec_count = 0

    def reset_tracking(self):
        self.recenter_pending = False
        self.origin_set = False
        self.yaw_q = Quaternion((1.0, 0.0, 0.0, 0.0))
        self.pos = Vector((0.0, 0.0, 0.0))
        self.prev_pp = None
        self.raw_prev = None
        self.pos_filter = OneEuroFilter()
        self.jumps = 0
        self.sm_q = None

    def reset_smoothing(self):
        self.sm_q = None
        self.pos_filter.reset()
        self.prev_pp = None

    # ---- lifecycle ---------------------------------------------------------------
    def start(self, port):
        if self.running:
            return
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            if os.name != "nt":
                srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            udp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind(("0.0.0.0", port))
            srv.listen(2)
            srv.settimeout(0.5)
            udp.bind(("0.0.0.0", port))
            udp.settimeout(0.5)
        except OSError:
            srv.close()
            udp.close()
            raise
        self.srv, self.udp, self.port = srv, udp, port
        self.ips = local_ips()
        self.hostname = socket.gethostname()
        self.blender_version = bpy.app.version_string
        self.running = True
        self.last_error = ""
        self.reset_tracking()
        threading.Thread(target=self._accept_loop, daemon=True, name="vcam-accept").start()
        threading.Thread(target=self._discovery_loop, daemon=True, name="vcam-discovery").start()
        self.draw_handle = bpy.types.SpaceView3D.draw_handler_add(_draw_callback, (), "WINDOW", "POST_PIXEL")
        if not bpy.app.timers.is_registered(_tick):
            bpy.app.timers.register(_tick, first_interval=0.0, persistent=True)

    def stop(self):
        was_running = self.running
        self.running = False
        with self.lock:
            client, self.client = self.client, None
        if client is not None:
            client.close()
        for s in (self.srv, self.udp):
            try:
                if s is not None:
                    s.close()
            except OSError:
                pass
        self.srv = self.udp = None
        if self.draw_handle is not None:
            try:
                bpy.types.SpaceView3D.draw_handler_remove(self.draw_handle, "WINDOW")
            except Exception:
                pass
            self.draw_handle = None
        unpatch_engines()
        for off in (self.offscreen, self.cap_offscreen):
            try:
                if off is not None:
                    off.free()
            except Exception:
                pass
        self.offscreen = self.cap_offscreen = None
        self.off_size = self.cap_size = None
        self.engine_captures = False
        if bpy.app.timers.is_registered(_tick):
            bpy.app.timers.unregister(_tick)
        self.recording = False
        self.pose = None
        while not self.commands.empty():
            try:
                self.commands.get_nowait()
            except queue.Empty:
                break
        if was_running:
            self._tag_redraw_all()

    # ---- threads -----------------------------------------------------------------
    def _accept_loop(self):
        while self.running:
            try:
                conn, addr = self.srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn.settimeout(5.0)
            client = Client(self, conn, addr)
            with self.lock:
                old, self.client = self.client, client
            if old is not None:
                old.close()
            client.start()

    def _discovery_loop(self):
        while self.running:
            try:
                data, addr = self.udp.recvfrom(1024)
            except socket.timeout:
                continue
            except OSError:
                break
            if data.startswith(DISCOVER_MAGIC):
                reply = json.dumps(
                    {
                        "app": "blender-vcam",
                        "protocol": PROTOCOL_VERSION,
                        "name": self.hostname,
                        "blender": self.blender_version,
                        "port": self.port,
                    }
                ).encode("utf-8")
                try:
                    self.udp.sendto(reply, addr)
                except OSError:
                    pass

    def set_pose(self, values):
        with self.lock:
            self.pose = (values, time.perf_counter())

    def _client_closed(self, client):
        with self.lock:
            if self.client is client:
                self.client = None
        self.commands.put((client, {"cmd": "_closed"}))

    # ---- helpers (main thread) ---------------------------------------------------
    @staticmethod
    def props():
        return bpy.context.window_manager.vcam

    def target_camera(self):
        cam = None
        try:
            cam = self.props().camera
            if cam is not None and cam.type != "CAMERA":
                cam = None
        except Exception:
            cam = None
        if cam is None:
            cam = bpy.context.scene.camera
        return cam

    def _tag_redraw_all(self):
        try:
            for window in bpy.context.window_manager.windows:
                for area in window.screen.areas:
                    if area.type == "VIEW_3D":
                        area.tag_redraw()
        except Exception:
            pass

    def toast(self, text):
        c = self.client
        if c is not None:
            c.send_json({"type": "toast", "text": text})

    # ---- main-thread tick --------------------------------------------------------
    def tick(self):
        if not self.running:
            return None
        try:
            self.engine_captures = patch_engine(bpy.context.scene.render.engine)
            self._drain_commands()
            self._apply_pose()
            self._record_tick()
            now = time.perf_counter()
            c = self.client
            if c is not None and now - self.last_status_t > 0.25:
                self.last_status_t = now
                c.send_json(self._status())
                if self.engine_captures and self.capture_hint and now - self.last_frame_t > 3.0:
                    self.toast(self.capture_hint)
                    self.last_frame_t = now  # repeat the hint every few seconds, not every status
            self._redraw(now)
        except Exception:
            self.last_error = traceback.format_exc(limit=4)
            traceback.print_exc()
        return TICK_SECONDS

    def _redraw(self, now):
        streaming = self.client is not None and self.props().stream_enabled
        interval = 1.0 / max(5, self.props().stream_fps) if streaming else 0.5
        if now - self.last_redraw_t < interval:
            return
        self.last_redraw_t = now
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == "VIEW_3D":
                    area.tag_redraw()
                    if not self.engine_captures:
                        break  # one 3D view per window is enough to drive the offscreen render

    def _status(self):
        scene = bpy.context.scene
        p = self.props()
        cam = self.target_camera()
        d = {
            "type": "status",
            "camera": cam.name if cam else "",
            "recording": self.recording,
            "frame": scene.frame_current,
            "fps": scene.render.fps / scene.render.fps_base,
            "res": [scene.render.resolution_x, scene.render.resolution_y],
            "level": bool(p.level_horizon),
            "smoothing": p.smoothing,
            "pillow": Image is not None,
            "stream": {"width": p.stream_width, "fps": p.stream_fps, "quality": p.jpeg_quality},
        }
        if cam is not None and cam.data is not None:
            d["focal"] = cam.data.lens
            d["dof"] = {
                "on": bool(cam.data.dof.use_dof),
                "distance": cam.data.dof.focus_distance,
                "fstop": cam.data.dof.aperture_fstop,
            }
        return d

    def _send_cameras(self):
        c = self.client
        if c is None:
            return
        cam = self.target_camera()
        names = [o.name for o in bpy.context.scene.objects if o.type == "CAMERA"]
        c.send_json({"type": "cameras", "items": names, "active": cam.name if cam else ""})

    # ---- commands ----------------------------------------------------------------
    def _drain_commands(self):
        for _ in range(64):
            try:
                client, msg = self.commands.get_nowait()
            except queue.Empty:
                return
            try:
                self._handle_command(client, msg)
            except Exception:
                traceback.print_exc()

    def _handle_command(self, client, msg):
        cmd = msg.get("cmd")
        p = self.props()
        if cmd == "_closed":
            if self.client is None:
                if self.recording:
                    self.recording = False
                self.origin_set = False
                self.reset_smoothing()
            return
        if client is not self.client:
            return
        if cmd == "hello":
            client.device = str(msg.get("device") or client.device)[:40]
            self._apply_stream(msg.get("stream") or {}, p)
            self.origin_set = False
            self.reset_smoothing()
            with self.lock:
                self.pose = None
            self._send_cameras()
            client.send_json(self._status())
        elif cmd == "stream":
            self._apply_stream(msg, p)
        elif cmd == "ping":
            client.send_json({"type": "pong", "t": msg.get("t")})
        elif cmd == "recenter":
            self.recenter_pending = True
        elif cmd == "level":
            p.level_horizon = bool(msg.get("on"))
        elif cmd == "smoothing":
            p.smoothing = clamp(float(msg.get("value", 0.0)), 0.0, 0.95)
        elif cmd == "record":
            if msg.get("on"):
                self.start_recording()
            else:
                self.stop_recording()
        elif cmd == "list_cameras":
            self._send_cameras()
        elif cmd == "set_camera":
            obj = bpy.data.objects.get(str(msg.get("name", "")))
            if obj is not None and obj.type == "CAMERA":
                p.camera = obj
                bpy.context.scene.camera = obj
                self.origin_set = False
                self.reset_smoothing()
                self._send_cameras()
        elif cmd == "set":
            cam = self.target_camera()
            if cam is None or cam.data is None:
                return
            d = cam.data
            if "focal" in msg:
                d.lens = clamp(float(msg["focal"]), 1.0, 1000.0)
            if "dof" in msg:
                d.dof.use_dof = bool(msg["dof"])
            if "focus_distance" in msg:
                d.dof.focus_distance = clamp(float(msg["focus_distance"]), 0.001, 100000.0)
            if "fstop" in msg:
                d.dof.aperture_fstop = clamp(float(msg["fstop"]), 0.1, 128.0)

    @staticmethod
    def _apply_stream(s, p):
        if "width" in s:
            p.stream_width = int(clamp(int(s["width"]), 160, 1920))
        if "fps" in s:
            p.stream_fps = int(clamp(int(s["fps"]), 5, 60))
        if "quality" in s:
            p.jpeg_quality = int(clamp(int(s["quality"]), 20, 95))
        if "enabled" in s:
            p.stream_enabled = bool(s["enabled"])

    # ---- pose --------------------------------------------------------------------
    def _apply_pose(self):
        with self.lock:
            pose, self.pose = self.pose, None
        if pose is None:
            return
        cam = self.target_camera()
        if cam is None:
            return
        p = self.props()
        (px, py, pz, qw, qx, qy, qz), t = pose
        qp = Quaternion((qw, qx, qy, qz))
        if qp.magnitude < 1e-6:
            return
        qp.normalize()
        raw = Vector((px, py, pz))

        # ARCore relocalising after lost tracking moves the pose in one step: continue from where we were.
        if self.raw_prev is not None and (raw - self.raw_prev).length > p.jump_limit:
            self.jumps += 1
            self.pos_filter.reset()
            self.prev_pp = None
        self.raw_prev = raw.copy()
        # Filtered in phone metres, before move_scale, so the settings don't depend on the scene's scale.
        if p.filter_position:
            pp = self.pos_filter.filter(raw, t, p.filter_min_cutoff, p.filter_beta)
        else:
            pp = raw
        if self.prev_pp is None:
            self.prev_pp = pp.copy()

        loc0, rot0, scale0 = cam.matrix_world.decompose()
        if self.recenter_pending or not self.origin_set:
            if self.recenter_pending or p.auto_recenter:
                # Make the phone's current pose coincide with the camera's current pose (yaw + position only).
                yaw = heading_of(rot0) - heading_of(qp)
                self.yaw_q = Quaternion((0.0, 0.0, 1.0), yaw)
                self.pos = loc0.copy()
            else:
                self.yaw_q = Quaternion((1.0, 0.0, 0.0, 0.0))
                self.pos = pp.copy()
            self.prev_pp = pp.copy()
            self.origin_set = True
            self.recenter_pending = False
            self.sm_q = None

        q = self.yaw_q @ qp
        phone_heading = heading_of(q)
        if not p.use_rotation:
            q = rot0
        if p.level_horizon:
            q = level_quaternion(q)

        # Position is integrated from deltas so each phone direction can be remapped, relative to where the
        # phone faces. With the default mapping this equals the absolute phone position.
        delta = self.yaw_q @ (pp - self.prev_pp)
        self.prev_pp = pp.copy()
        # heading_of is measured from +X; the local frame has forward on +Y.
        local = Quaternion((0.0, 0.0, 1.0), math.pi / 2 - phone_heading) @ delta
        moved = Vector(
            (
                phone_component(local, p.move_right),
                phone_component(local, p.move_forward),
                phone_component(local, p.move_up),
            )
        ) * p.move_scale
        if p.move_space == "CAMERA":
            moved = Quaternion((0.0, 0.0, 1.0), heading_of(q) - math.pi / 2) @ moved
        if p.use_position:
            self.pos += moved
        else:
            self.pos = loc0.copy()
        pos = self.pos.copy()

        smooth = p.smoothing
        if self.sm_q is None or smooth <= 0.001:
            self.sm_q = q.copy()
        else:
            a = max(0.03, 1.0 - smooth)
            if self.sm_q.dot(q) < 0.0:
                q.negate()
            self.sm_q = self.sm_q.slerp(q, a)
        set_world_matrix(cam, Matrix.LocRotScale(pos, self.sm_q, scale0))

    # ---- recording ---------------------------------------------------------------
    def start_recording(self):
        scene = bpy.context.scene
        self.rec_start_frame = scene.frame_current
        self.rec_start_time = time.perf_counter()
        self.rec_last_frame = None
        self.rec_count = 0
        self.recording = True

    def stop_recording(self):
        if self.recording:
            self.recording = False
            self.toast("Recorded %d frames" % self.rec_count)

    def _record_tick(self):
        if not self.recording:
            return
        cam = self.target_camera()
        if cam is None:
            return
        scene = bpy.context.scene
        fps = scene.render.fps / scene.render.fps_base
        frame = self.rec_start_frame + int((time.perf_counter() - self.rec_start_time) * fps)
        if frame == self.rec_last_frame:
            return
        self.rec_last_frame = frame
        self.rec_count += 1
        scene.frame_current = frame
        cam.keyframe_insert(data_path="location", frame=frame)
        cam.keyframe_insert(data_path=ROT_PATHS.get(cam.rotation_mode, "rotation_euler"), frame=frame)
        if cam.data is not None:
            cam.data.keyframe_insert(data_path="lens", frame=frame)
            if cam.data.dof.use_dof:
                cam.data.keyframe_insert(data_path="dof.focus_distance", frame=frame)

    # ---- preview rendering (called from the 3D view draw handler) ----------------
    def stream_dims(self, scene, p):
        r = scene.render
        aspect = (max(1, r.resolution_x) * r.pixel_aspect_x) / (max(1, r.resolution_y) * r.pixel_aspect_y)
        width = int(p.stream_width)
        if Image is None:
            width = min(width, 480)  # PNG fallback is bandwidth hungry
        width = max(64, width) & ~1
        height = max(2, int(round(width / aspect))) & ~1
        return width, height

    def render_preview(self, context):
        client = self.client
        p = self.props()
        if client is None or not p.stream_enabled or client.frame_pending():
            return
        space = context.space_data
        if space is None or space.type != "VIEW_3D":
            return
        now = time.perf_counter()
        if self.engine_captures and (space.shading.type == "RENDERED" or now - self.last_capture_t < 1.0):
            return  # draw_view3d can't run a Python engine; capture_engine_view streams that view
        if now - self.last_frame_t < 0.9 / max(5, p.stream_fps):
            return
        cam = self.target_camera()
        if cam is None:
            return
        scene = context.scene
        w, h = self.stream_dims(scene, p)
        if self.offscreen is None or self.off_size != (w, h):
            if self.offscreen is not None:
                self.offscreen.free()
            self.offscreen = gpu.types.GPUOffScreen(w, h)
            self.off_size = (w, h)
        depsgraph = context.evaluated_depsgraph_get()
        view_matrix = cam.matrix_world.normalized().inverted()
        projection = cam.calc_matrix_camera(depsgraph, x=w, y=h, scale_x=1.0, scale_y=1.0)
        self.offscreen.draw_view3d(
            scene, context.view_layer, space, context.region, view_matrix, projection, do_color_management=True
        )
        with self.offscreen.bind():
            fb = gpu.state.active_framebuffer_get()
            buf = fb.read_color(0, 0, w, h, 4, 0, "UBYTE")
        arr, slow = buffer_to_numpy(buf, w, h)
        self.slow_readback = slow
        self.last_frame_t = now
        client.push_frame(RawFrame(arr, p.jpeg_quality))

    def capture_engine_view(self, draw, engine, context, depsgraph):
        """Runs inside a Python engine's view_draw. Draws the engine a second time into a small offscreen whose
        viewport is placed so the camera frame fills it: no full-resolution readback."""
        try:
            client = self.client
            p = self.props()
            if client is None or not p.stream_enabled or client.frame_pending():
                return
            region, rv3d = context.region, context.region_data
            scene = context.scene
            cam = self.target_camera()
            if rv3d is None or cam is None:
                return
            if rv3d.view_perspective != "CAMERA" or scene.camera != cam:
                self.capture_hint = "Look through the phone camera (Numpad 0) in a Rendered viewport"
                return
            now = time.perf_counter()
            if now - self.last_frame_t < 0.9 / max(5, p.stream_fps):
                return
            rect = camera_frame_rect(scene, cam, region, rv3d)
            if rect is None:
                return
            x0, y0, x1, y1 = rect
            w, h = self.stream_dims(scene, p)
            if self.cap_offscreen is None or self.cap_size != (w, h):
                if self.cap_offscreen is not None:
                    self.cap_offscreen.free()
                self.cap_offscreen = gpu.types.GPUOffScreen(w, h, format="RGBA16F")
                self.cap_size = (w, h)
            sx, sy = w / (x1 - x0), h / (y1 - y0)
            saved_viewport = gpu.state.viewport_get()
            with self.cap_offscreen.bind():
                fb = gpu.state.active_framebuffer_get()
                fb.clear(color=(0.0, 0.0, 0.0, 1.0), depth=1.0)
                gpu.state.viewport_set(
                    int(round(-x0 * sx)), int(round(-y0 * sy)),
                    int(round(region.width * sx)), int(round(region.height * sy)),
                )
                draw(engine, context, depsgraph)
                buf = fb.read_color(0, 0, w, h, 4, 0, "FLOAT")
            gpu.state.viewport_set(*saved_viewport)
            vs = scene.view_settings
            self.capture_hint = ""
            self.last_frame_t = self.last_capture_t = now
            client.push_frame(RawFrame(float_buffer_to_numpy(buf, w, h), p.jpeg_quality, (vs.exposure, vs.gamma)))
        except Exception:
            err = traceback.format_exc(limit=3)
            if err != self.last_error:
                self.last_error = err
                print("[VCam] rendered preview error:\n" + err)


BRIDGE = Bridge()


def _tick():
    return BRIDGE.tick()


def _draw_callback():
    b = BRIDGE
    if not b.running or b.client is None:
        return
    try:
        b.render_preview(bpy.context)
    except Exception:
        err = traceback.format_exc(limit=3)
        if err != b.last_error:
            b.last_error = err
            print("[VCam] preview error:\n" + err)


# --------------------------------------------------------------------------------------
# Blender UI
# --------------------------------------------------------------------------------------
class VCamProps(bpy.types.PropertyGroup):
    port: IntProperty(name="Port", default=DEFAULT_PORT, min=1024, max=65535)
    camera: PointerProperty(
        name="Camera",
        description="Camera driven by the phone (empty = the scene's active camera)",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == "CAMERA",
    )
    stream_enabled: BoolProperty(name="Stream preview to phone", default=True)
    stream_width: IntProperty(name="Preview width", default=640, min=160, max=1920, subtype="PIXEL")
    stream_fps: IntProperty(name="Preview FPS", default=30, min=5, max=60)
    jpeg_quality: IntProperty(name="JPEG quality", default=70, min=20, max=95)
    smoothing: FloatProperty(
        name="Rotation smoothing", default=0.25, min=0.0, max=0.95, subtype="FACTOR",
        description="Also set by the phone's Smoothing slider",
    )
    filter_position: BoolProperty(
        name="Filter position jitter", default=True,
        description="One Euro filter on the phone's position: steady when still, responsive when moving",
    )
    filter_min_cutoff: FloatProperty(
        name="Steadiness", default=1.0, min=0.01, soft_max=10.0, unit="NONE",
        description="Cutoff in Hz while still. Lower = steadier but more lag on slow moves",
    )
    filter_beta: FloatProperty(
        name="Speed response", default=3.0, min=0.0, soft_max=30.0,
        description="How fast the filter opens up with speed. Higher = less lag on quick moves, more jitter",
    )
    jump_limit: FloatProperty(
        name="Jump limit", default=0.5, min=0.01, soft_max=5.0, subtype="DISTANCE",
        description="Phone moves bigger than this in one update (tracking recovered) are ignored",
    )
    level_horizon: BoolProperty(name="Level horizon", description="Remove roll from the phone's rotation", default=False)
    auto_recenter: BoolProperty(
        name="Auto recenter on connect", description="Start from the camera's current pose", default=True
    )
    use_position: BoolProperty(name="Position", default=True)
    use_rotation: BoolProperty(name="Rotation", default=True)
    move_forward: EnumProperty(
        name="Forward", items=PHONE_DIRECTIONS, default="FORWARD",
        description="Phone movement that moves the camera forward",
    )
    move_right: EnumProperty(
        name="Right", items=PHONE_DIRECTIONS, default="RIGHT",
        description="Phone movement that moves the camera right",
    )
    move_up: EnumProperty(
        name="Up", items=PHONE_DIRECTIONS, default="UP",
        description="Phone movement that moves the camera up",
    )
    move_space: EnumProperty(
        name="Relative to",
        items=[
            ("CAMERA", "Camera", "Forward is where the camera faces (flattened to the ground)"),
            ("WORLD", "World", "Forward is world +Y, right +X, up +Z"),
        ],
        default="CAMERA",
    )
    move_scale: FloatProperty(
        name="Move scale", default=1.0, min=0.0, soft_max=100.0,
        description="Blender metres per metre the phone moves (joysticks included)",
    )


class VCAM_OT_start(bpy.types.Operator):
    bl_idname = "vcam.start"
    bl_label = "Start Server"
    bl_description = "Listen for the VCam phone app on the local network"

    def execute(self, context):
        p = context.window_manager.vcam
        try:
            BRIDGE.start(p.port)
        except OSError as e:
            self.report({"ERROR"}, "Could not open port %d: %s" % (p.port, e))
            return {"CANCELLED"}
        return {"FINISHED"}


class VCAM_OT_stop(bpy.types.Operator):
    bl_idname = "vcam.stop"
    bl_label = "Stop Server"

    def execute(self, context):
        BRIDGE.stop()
        return {"FINISHED"}


class VCAM_OT_recenter(bpy.types.Operator):
    bl_idname = "vcam.recenter"
    bl_label = "Recenter"
    bl_description = "Make the phone's current pose match the camera's current pose"

    def execute(self, context):
        BRIDGE.recenter_pending = True
        return {"FINISHED"}


class VCAM_OT_record(bpy.types.Operator):
    bl_idname = "vcam.record"
    bl_label = "Record / Stop"
    bl_description = "Insert camera keyframes on every frame while the timeline advances in real time"

    def execute(self, context):
        if BRIDGE.recording:
            BRIDGE.stop_recording()
        else:
            BRIDGE.start_recording()
        return {"FINISHED"}


class VCAM_OT_add_camera(bpy.types.Operator):
    bl_idname = "vcam.add_camera"
    bl_label = "Add Phone Camera"

    def execute(self, context):
        data = bpy.data.cameras.new("PhoneCam")
        obj = bpy.data.objects.new("PhoneCam", data)
        context.collection.objects.link(obj)
        obj.location = context.scene.cursor.location
        obj.rotation_euler = (math.radians(90.0), 0.0, 0.0)
        context.window_manager.vcam.camera = obj
        context.scene.camera = obj
        return {"FINISHED"}


class VCAM_OT_install_pillow(bpy.types.Operator):
    bl_idname = "vcam.install_pillow"
    bl_label = "Install Pillow (faster preview)"
    bl_description = "Installs the Pillow imaging library into Blender's user modules folder (needs internet)"

    def execute(self, context):
        target = _modules_dir()
        if not target:
            self.report({"ERROR"}, "Could not locate Blender's user modules folder")
            return {"CANCELLED"}
        py = sys.executable
        try:
            subprocess.call([py, "-m", "ensurepip", "--upgrade"])
            subprocess.check_call([py, "-m", "pip", "install", "--upgrade", "--target", target, "pillow"])
        except Exception as e:
            self.report({"ERROR"}, "Pillow install failed (see system console): %s" % e)
            return {"CANCELLED"}
        _ensure_modules_on_path()
        importlib.invalidate_caches()
        if _try_import_pil():
            self.report({"INFO"}, "Pillow installed - preview now uses JPEG")
            return {"FINISHED"}
        self.report({"WARNING"}, "Pillow installed but could not be imported; restart Blender")
        return {"FINISHED"}


class VCAM_PT_main(bpy.types.Panel):
    bl_label = "Phone Virtual Camera"
    bl_idname = "VCAM_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "VCam"

    def draw(self, context):
        layout = self.layout
        p = context.window_manager.vcam
        b = BRIDGE

        if not b.running:
            layout.prop(p, "port")
            layout.operator("vcam.start", icon="PLAY")
        else:
            layout.operator("vcam.stop", icon="PAUSE")
            box = layout.box()
            box.label(text="Enter this address in the phone app:")
            for ip in b.ips:
                box.label(text="%s : %d" % (ip, b.port))
            c = b.client
            box.label(text=("Connected: " + c.device) if c else "Waiting for phone...")
            if b.last_error:
                box.label(text="Error - see system console", icon="ERROR")

        col = layout.column(align=True)
        col.prop(p, "camera")
        if p.camera is None and context.scene.camera is None:
            col.operator("vcam.add_camera", icon="PLUS")

        box = layout.box()
        box.label(text="Motion")
        box.prop(p, "smoothing", slider=True)
        box.prop(p, "level_horizon")
        box.prop(p, "auto_recenter")
        row = box.row(align=True)
        row.prop(p, "use_position", toggle=True)
        row.prop(p, "use_rotation", toggle=True)
        box.operator("vcam.recenter", icon="FILE_REFRESH")

        box = layout.box()
        box.label(text="Movement")
        col = box.column(align=True)
        col.prop(p, "move_forward")
        col.prop(p, "move_right")
        col.prop(p, "move_up")
        box.prop(p, "move_space", expand=True)
        box.prop(p, "move_scale")
        box.prop(p, "filter_position")
        col = box.column(align=True)
        col.active = p.filter_position
        col.prop(p, "filter_min_cutoff")
        col.prop(p, "filter_beta")
        box.prop(p, "jump_limit")
        if b.jumps:
            box.label(text="Tracking jumps ignored: %d" % b.jumps)

        box = layout.box()
        box.label(text="Preview")
        box.prop(p, "stream_enabled")
        col = box.column(align=True)
        col.prop(p, "stream_width")
        col.prop(p, "stream_fps")
        col.prop(p, "jpeg_quality")
        if Image is None:
            box.label(text="Pillow missing: PNG fallback, max 480 px", icon="ERROR")
            box.operator("vcam.install_pillow")
        if b.slow_readback:
            box.label(text="Slow GPU readback path in use", icon="ERROR")
        if b.running:
            box.label(text="Keep a 3D viewport visible while streaming")
            if b.engine_captures and b.capture_hint and time.perf_counter() - b.last_capture_t > 1.0:
                col = box.column(align=True)
                col.label(text="Rendered preview: look through", icon="INFO")
                col.label(text="the camera (Numpad 0) with")
                col.label(text="Rendered viewport shading")

        box = layout.box()
        box.label(text="Recording")
        box.operator("vcam.record", text="Stop" if b.recording else "Record keyframes", icon="REC" if not b.recording else "PAUSE")


classes = (
    VCamProps,
    VCAM_OT_start,
    VCAM_OT_stop,
    VCAM_OT_recenter,
    VCAM_OT_record,
    VCAM_OT_add_camera,
    VCAM_OT_install_pillow,
    VCAM_PT_main,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.WindowManager.vcam = PointerProperty(type=VCamProps)
    _ensure_modules_on_path()
    _try_import_pil()


def unregister():
    BRIDGE.stop()
    del bpy.types.WindowManager.vcam
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
