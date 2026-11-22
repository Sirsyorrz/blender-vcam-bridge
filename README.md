# VCam Bridge — Phone Virtual Camera for Blender

Drive a Blender camera from your phone: the phone streams its pose (orientation + position) over
TCP at ~60 Hz, Blender moves the camera live and renders the camera's view back to the phone as a
preview stream. Record the result as regular keyframes on the timeline.

Works with the **[Blender VCam Android app](https://github.com/toti-s/blender-vcam-android)**.

## Features

- Handheld camera control — rotate and walk with the phone, Blender's camera follows (gyro, or
  6-DoF ARCore tracking on supported phones)
- Live viewfinder streamed from Blender to the phone (JPEG with Pillow, PNG fallback without)
- Rendered preview for Python render engines (Cycles, Deadlock (game), ...): look through the phone camera
  (Numpad 0) in a viewport with Rendered shading and the phone shows exactly that render
- Movement mapping: pick which phone direction drives the camera's forward, right and up (or turn one off),
  relative to the camera's heading or the world axes, with a move scale
- Position jitter filter (One Euro: steady when still, responsive when moving), tuned in phone metres so it
  behaves the same at any move scale; rotation smoothing stays separate
- Tracking jumps (ARCore relocalising after lost tracking) are ignored instead of teleporting the camera
- Change focal length, depth of field (focus distance / f-stop) from the phone
- Start/stop keyframe recording from the phone — inserts location, rotation and lens keyframes
- Switch between cameras, recenter, level the horizon, motion smoothing
- Automatic LAN discovery — the phone finds Blender, or type the IP shown in the panel
- Virtual joysticks for dolly/strafe/lift/yaw without walking around

## Requirements

- Blender 3.6 or newer (tested up to 5.2)
- The [Android app](https://github.com/toti-s/blender-vcam-android) on a phone (Android 8+)
- Phone and computer on the same Wi-Fi network (allow Blender through the firewall if asked)
- Optional: [Pillow](https://pypi.org/project/Pillow/) for JPEG preview — without it the preview
  falls back to PNG capped at 480 px. The panel has a one-click *Install Pillow* button.

## Install

1. Download `blender_vcam_bridge.py` from the [latest release](https://github.com/toti-s/blender-vcam-bridge/releases/latest)
   (or grab it straight from this repo).
2. In Blender: **Edit > Preferences > Add-ons > ⌄ > Install from Disk…** and pick the file.
3. Enable **VCam Bridge (Phone Virtual Camera)**.

## Use

1. In a 3D Viewport press **N** and open the **VCam** tab.
2. Click **Start Server** — the panel shows the addresses to enter on the phone
   (e.g. `192.168.1.23 : 9876`).
3. On the phone tap **Search** (or type the address) and **Connect**.
4. Keep a 3D viewport visible while streaming — the preview is rendered from it.

Recording (**Record** button in the panel or on the phone) inserts a keyframe per scene frame while
the timeline advances in real time, including focal length and focus distance when DoF is on.

## Protocol

Everything is plain TCP/UDP on one port (default **9876**), so other clients can implement it too:

- Every message: `[1 byte type][4 bytes big-endian length][payload]`
  - type `1` JSON (UTF-8, both directions)
  - type `2` pose (phone → Blender): 7 big-endian float32 `px py pz qw qx qy qz` — position in
    Blender world axes (metres) and camera orientation (camera looks down its local −Z, +Y up)
  - type `3` frame (Blender → phone): JPEG or PNG bytes
- Discovery: UDP broadcast of `BLENDER_VCAM_DISCOVER` → JSON reply
  `{"app": "blender-vcam", "protocol": 1, "name": …, "blender": …, "port": …}`
- JSON commands: `hello`, `stream`, `ping`, `recenter`, `level`, `smoothing`, `record`,
  `set_camera`, `set` (focal / dof / focus_distance / fstop)
- JSON messages: `status`, `cameras`, `pong`, `toast`

## License

[GPL-3.0-or-later](LICENSE)
