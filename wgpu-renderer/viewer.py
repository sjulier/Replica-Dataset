"""Interactive Replica viewer in a window.

    python viewer.py <replica_v1>/room_0

Controls: drag with the left mouse button to look around; W/A/S/D move,
Q/E down/up, hold Shift to move faster; +/- change the exposure;
M toggles mirrors and glass;
R resets the view; Esc quits.
The title bar shows the frame rate.
"""

from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np
import wgpu
from rendercanvas.glfw import RenderCanvas, loop

from replica_wgpu import Camera, Renderer
from replica_wgpu.scene import read_ply

LOOK_SPEED = 0.005  # radians per pixel dragged
MOVE_SPEED = 1.0  # metres per second (x4 with Shift)


class Viewer:
    def __init__(self, scene_dir: str, size: tuple[int, int], vfov_deg: float, atlas: str,
                 exposure: float, max_seconds: float | None):
        self.vfov = math.radians(vfov_deg)
        w, h = size
        # The Renderer's own camera is only for its off-screen render(); the window
        # gets a camera matching its current size in draw().
        self.r = Renderer(scene_dir, self._camera(w, h), atlas_format=atlas, mirrors=True)
        self.r.exposure = exposure

        pos, _ = read_ply(os.path.join(scene_dir, "mesh.ply"))
        self.home = (pos.min(axis=0) + pos.max(axis=0)) / 2
        self.reset()

        self.canvas = RenderCanvas(size=size, title="Replica", update_mode="continuous",
                                   max_fps=120, vsync=True)
        self.context = self.canvas.get_wgpu_context()
        # The shader applies Replica's gamma itself: use a non-sRGB surface format so
        # the image is not gamma-encoded a second time.
        fmt = self.context.get_preferred_format(self.r.adapter).removesuffix("-srgb")
        self.context.configure(device=self.r.device, format=fmt)
        self.targets_size = None

        self.keys: set[str] = set()
        self.drag_from: tuple[float, float] | None = None
        self.last_t = time.perf_counter()
        self.fps_t, self.fps_n = self.last_t, 0
        self.t_end = None if max_seconds is None else self.last_t + max_seconds

        self.canvas.add_event_handler(self.on_event, "key_down", "key_up", "pointer_down",
                                      "pointer_up", "pointer_move")
        self.canvas.request_draw(self.draw)

    def _camera(self, w: int, h: int) -> Camera:
        f = (h / 2) / math.tan(self.vfov / 2)
        return Camera(w, h, f, f, (w - 1) / 2, (h - 1) / 2)

    def reset(self) -> None:
        self.pos = self.home.copy()
        self.yaw, self.pitch = 0.0, 0.0

    def pose(self) -> np.ndarray:
        """Camera-to-world (x right, y down, z forward) from yaw/pitch; world is z-up."""
        cp = math.cos(self.pitch)
        fwd = np.array([cp * math.cos(self.yaw), cp * math.sin(self.yaw), math.sin(self.pitch)])
        right = np.cross(fwd, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        T = np.eye(4)
        T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = right, down, fwd, self.pos
        return T

    # ---- input --------------------------------------------------------------

    def on_event(self, ev: dict) -> None:
        t = ev["event_type"]
        if t in ("key_down", "key_up"):
            key = ev["key"].lower()
            if t == "key_down":
                if key == "escape":
                    self.canvas.close()
                elif key == "r":
                    self.reset()
                elif key == "m":
                    self.r.draw_mirrors = not self.r.draw_mirrors
                elif key in ("+", "=", "-", "_"):
                    self.r.exposure *= 1.25 if key in ("+", "=") else 0.8
                    print(f"exposure {self.r.exposure:.4f}", flush=True)
                self.keys.add(key)
            else:
                self.keys.discard(key)
        elif t == "pointer_down" and ev["button"] == 1:
            self.drag_from = (ev["x"], ev["y"])
        elif t == "pointer_up" and ev["button"] == 1:
            self.drag_from = None
        elif t == "pointer_move" and self.drag_from is not None:
            x0, y0 = self.drag_from
            self.yaw -= (ev["x"] - x0) * LOOK_SPEED
            self.pitch = max(-1.5, min(1.5, self.pitch - (ev["y"] - y0) * LOOK_SPEED))
            self.drag_from = (ev["x"], ev["y"])

    def move(self, dt: float) -> None:
        speed = MOVE_SPEED * (4 if "shift" in self.keys else 1) * dt
        flat = np.array([math.cos(self.yaw), math.sin(self.yaw), 0.0])
        right = np.array([math.sin(self.yaw), -math.cos(self.yaw), 0.0])
        up = np.array([0.0, 0.0, 1.0])
        for key, d in (("w", flat), ("s", -flat), ("d", right), ("a", -right), ("e", up), ("q", -up)):
            if key in self.keys:
                self.pos += speed * d

    # ---- drawing --------------------------------------------------------------

    def draw(self) -> None:
        now = time.perf_counter()
        self.move(now - self.last_t)
        self.last_t = now
        if self.t_end is not None and now > self.t_end:
            self.canvas.close()
            return

        tex = self.context.get_current_texture()
        w, h = tex.size[0], tex.size[1]
        if self.targets_size != (w, h):
            d = self.r.device
            self.depth_tex = d.create_texture(size=(w, h, 1), format=wgpu.TextureFormat.r32float,
                                              usage=wgpu.TextureUsage.RENDER_ATTACHMENT)
            self.zbuf = d.create_texture(size=(w, h, 1), format=wgpu.TextureFormat.depth32float,
                                         usage=wgpu.TextureUsage.RENDER_ATTACHMENT)
            self.targets_size = (w, h)
        self.r.draw(self.pose(), self._camera(w, h), tex, self.depth_tex, self.zbuf)

        self.fps_n += 1
        if now - self.fps_t > 1.0:
            fps = self.fps_n / (now - self.fps_t)
            self.canvas.set_title(f"Replica - {fps:.0f} fps, {w}x{h}, exposure {self.r.exposure:.4f}, "
                                  f"mirrors {'on' if self.r.draw_mirrors else 'off'}")
            print(f"{fps:.0f} fps at {w}x{h}", flush=True)
            self.fps_t, self.fps_n = now, 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_dir")
    ap.add_argument("--size", type=int, nargs=2, default=(1200, 680), metavar=("W", "H"))
    ap.add_argument("--vfov", type=float, default=59.1,
                    help="vertical field of view in degrees (59.1 = pyslam's Replica camera)")
    ap.add_argument("--atlas", choices=["rgba16float", "rgb9e5"], default="rgba16float")
    ap.add_argument("--exposure", type=float, default=Renderer.exposure)
    ap.add_argument("--seconds", type=float, help="close after this many seconds (for testing)")
    args = ap.parse_args()
    Viewer(args.scene_dir, tuple(args.size), args.vfov, args.atlas, args.exposure, args.seconds)
    loop.run()


if __name__ == "__main__":
    main()
