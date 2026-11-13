"""Off-screen Replica renderer on WebGPU (wgpu): Metal, D3D12 or Vulkan.

    r = Renderer("replica_v1/room_0", Camera.replica_pyslam())
    rgb, depth = r.render(T_world_camera)   # (H,W,3) uint8, (H,W) float32 metres

Poses use the camera convention of ReplicaSDK and OpenCV: x right, y down,
z forward. T_world_camera maps camera coordinates to world coordinates.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
import wgpu

from .scene import Scene, load_hdr_atlas, load_hdr_atlas_rgb9e5, load_scene

# Atlas formats on the GPU: (wgpu format, loader, bytes per texel)
ATLAS_FORMATS = {
    "rgba16float": (wgpu.TextureFormat.rgba16float, load_hdr_atlas, 8),
    "rgb9e5": (wgpu.TextureFormat.rgb9e5ufloat, load_hdr_atlas_rgb9e5, 4),
}

_SHADER = os.path.join(os.path.dirname(__file__), "ptex.wgsl")


@dataclass
class Camera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float  # pixel-centre convention: the centre of the image is (width-1)/2
    cy: float
    near: float = 0.1
    far: float = 100.0

    @staticmethod
    def replica_pyslam() -> "Camera":
        """The camera of pyslam's settings/REPLICA.yaml (and the NICE-SLAM sequences)."""
        return Camera(1200, 680, 600.0, 600.0, 599.5, 339.5)

    @staticmethod
    def replica_sdk() -> "Camera":
        """The camera of ReplicaSDK/src/render.cpp."""
        return Camera(1280, 960, 640.0, 640.0, 639.5, 479.5)

    def projection(self) -> np.ndarray:
        """Camera (x right, y down, z forward) to WebGPU clip space (y up, z in [0,1])."""
        w, h, n, f = self.width, self.height, self.near, self.far
        p = np.zeros((4, 4))
        p[0, 0] = 2 * self.fx / w
        p[0, 2] = 2 * (self.cx + 0.5) / w - 1
        p[1, 1] = -2 * self.fy / h
        p[1, 2] = 1 - 2 * (self.cy + 0.5) / h
        p[2, 2] = f / (f - n)
        p[2, 3] = -f * n / (f - n)
        p[3, 2] = 1
        return p


@dataclass
class _GpuSubMesh:
    bind_group: object
    num_vertices: int


class Renderer:
    # HDR tone mapping defaults of PTexMesh
    exposure = 0.025
    gamma = 1.6969
    saturation = 1.5

    def __init__(self, scene_dir: str, camera: Camera, cull: str = "back",
                 atlas_format: str = "rgba16float", verbose: bool = True):
        self.camera = camera
        self.atlas_format = atlas_format
        self.verbose = verbose
        t0 = time.perf_counter()
        scene = load_scene(scene_dir)
        self._log(f"loaded mesh: {len(scene.submeshes)} sub-meshes, "
                  f"{sum(s.quads.shape[0] for s in scene.submeshes)} quads ({time.perf_counter() - t0:.1f} s)")
        self.tile_size = scene.tile_size

        self.adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        info = self.adapter.info
        self._log(f"GPU: {info.get('device', '?')} ({info.get('backend_type', '?')})")
        lim = self.adapter.limits
        self.device = self.adapter.request_device_sync(
            required_limits={
                "max-texture-dimension-2d": lim["max-texture-dimension-2d"],
                "max-storage-buffer-binding-size": lim["max-storage-buffer-binding-size"],
                "max-buffer-size": lim["max-buffer-size"],
            }
        )
        self._make_pipeline(cull)
        t0 = time.perf_counter()
        self.submeshes, self.gpu_bytes = self._upload(scene)
        self._log(f"uploaded atlases: {self.gpu_bytes / 1e9:.2f} GB of textures "
                  f"({time.perf_counter() - t0:.1f} s)")
        self._make_targets()

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    # ---- setup --------------------------------------------------------------

    def _make_pipeline(self, cull: str) -> None:
        d = self.device
        with open(_SHADER) as f:
            module = d.create_shader_module(code=f.read())
        self.frame_layout = d.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
             "buffer": {"type": wgpu.BufferBindingType.uniform}},
        ])
        storage = {"type": wgpu.BufferBindingType.read_only_storage}
        self.mesh_layout = d.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.FRAGMENT,
             "buffer": {"type": wgpu.BufferBindingType.uniform}},
            {"binding": 1, "visibility": wgpu.ShaderStage.VERTEX, "buffer": storage},
            {"binding": 2, "visibility": wgpu.ShaderStage.VERTEX, "buffer": storage},
            {"binding": 3, "visibility": wgpu.ShaderStage.FRAGMENT, "buffer": storage},
            {"binding": 4, "visibility": wgpu.ShaderStage.FRAGMENT,
             "texture": {"sample_type": wgpu.TextureSampleType.unfilterable_float}},
        ])
        self.pipeline = d.create_render_pipeline(
            layout=d.create_pipeline_layout(bind_group_layouts=[self.frame_layout, self.mesh_layout]),
            vertex={"module": module, "entry_point": "vs_main", "buffers": []},
            # Replica quads are counter-clockwise in OpenGL's y-up window space. The
            # projection flips y (WebGPU framebuffers are y-down), so they arrive clockwise.
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list,
                       "front_face": wgpu.FrontFace.cw, "cull_mode": cull},
            depth_stencil={"format": wgpu.TextureFormat.depth32float,
                           "depth_write_enabled": True,
                           "depth_compare": wgpu.CompareFunction.less},
            multisample=None,
            fragment={"module": module, "entry_point": "fs_main", "targets": [
                {"format": wgpu.TextureFormat.rgba8unorm},
                {"format": wgpu.TextureFormat.r32float},
            ]},
        )
        self.frame_buf = d.create_buffer(size=144, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self.frame_group = d.create_bind_group(layout=self.frame_layout, entries=[
            {"binding": 0, "resource": {"buffer": self.frame_buf}}])

    def _upload(self, scene: Scene) -> tuple[list[_GpuSubMesh], int]:
        d = self.device
        out, tex_bytes = [], 0

        def buffer(arr: np.ndarray, usage) -> object:
            return d.create_buffer_with_data(data=np.ascontiguousarray(arr).tobytes(), usage=usage)

        fmt, loader, bpt = ATLAS_FORMATS[self.atlas_format]
        for sm in scene.submeshes:
            texels, dim = loader(sm.atlas_path)
            tex = d.create_texture(size=(dim, dim, 1), format=fmt,
                                   usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
            d.queue.write_texture({"texture": tex}, texels, {"bytes_per_row": dim * bpt}, (dim, dim, 1))
            tex_bytes += texels.nbytes
            del texels
            info = np.array([dim // scene.tile_size, 0, 0, 0], dtype=np.int32)
            group = d.create_bind_group(layout=self.mesh_layout, entries=[
                {"binding": 0, "resource": {"buffer": buffer(info, wgpu.BufferUsage.UNIFORM)}},
                {"binding": 1, "resource": {"buffer": buffer(sm.positions.astype(np.float32), wgpu.BufferUsage.STORAGE)}},
                {"binding": 2, "resource": {"buffer": buffer(sm.quads, wgpu.BufferUsage.STORAGE)}},
                {"binding": 3, "resource": {"buffer": buffer(sm.adjacency, wgpu.BufferUsage.STORAGE)}},
                {"binding": 4, "resource": tex.create_view()},
            ])
            out.append(_GpuSubMesh(group, 6 * sm.quads.shape[0]))
        return out, tex_bytes

    def _make_targets(self) -> None:
        d, c = self.device, self.camera
        size = (c.width, c.height, 1)
        rt = wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC
        self.color_tex = d.create_texture(size=size, format=wgpu.TextureFormat.rgba8unorm, usage=rt)
        self.depth_out = d.create_texture(size=size, format=wgpu.TextureFormat.r32float, usage=rt)
        self.zbuffer = d.create_texture(size=size, format=wgpu.TextureFormat.depth32float,
                                        usage=wgpu.TextureUsage.RENDER_ATTACHMENT)

    # ---- rendering ------------------------------------------------------------

    def render(self, T_world_camera: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Render colour (H,W,3 uint8) and depth along the optical axis (H,W float32, metres; 0 = nothing)."""
        c = self.camera
        mv = np.linalg.inv(np.asarray(T_world_camera, dtype=np.float64))
        mvp = c.projection() @ mv
        frame = np.zeros(36, dtype=np.float32)
        frame[0:16] = mvp.T.reshape(-1)  # WGSL matrices are column-major
        frame[16:32] = mv.T.reshape(-1)
        frame[32:35] = (self.exposure, 1.0 / self.gamma, self.saturation)
        frame[35:36] = np.array([self.tile_size], dtype=np.int32).view(np.float32)
        d = self.device
        d.queue.write_buffer(self.frame_buf, 0, frame.tobytes())

        enc = d.create_command_encoder()
        rp = enc.begin_render_pass(
            color_attachments=[
                {"view": self.color_tex.create_view(), "load_op": wgpu.LoadOp.clear,
                 "store_op": wgpu.StoreOp.store, "clear_value": (0, 0, 0, 1)},
                {"view": self.depth_out.create_view(), "load_op": wgpu.LoadOp.clear,
                 "store_op": wgpu.StoreOp.store, "clear_value": (0, 0, 0, 0)},
            ],
            depth_stencil_attachment={"view": self.zbuffer.create_view(), "depth_clear_value": 1.0,
                                      "depth_load_op": wgpu.LoadOp.clear,
                                      "depth_store_op": wgpu.StoreOp.discard},
        )
        rp.set_pipeline(self.pipeline)
        rp.set_bind_group(0, self.frame_group)
        for sm in self.submeshes:
            rp.set_bind_group(1, sm.bind_group)
            rp.draw(sm.num_vertices)
        rp.end()
        d.queue.submit([enc.finish()])

        size = (c.width, c.height, 1)
        rgba = np.frombuffer(d.queue.read_texture({"texture": self.color_tex}, {"bytes_per_row": c.width * 4}, size),
                             dtype=np.uint8).reshape(c.height, c.width, 4)
        depth = np.frombuffer(d.queue.read_texture({"texture": self.depth_out}, {"bytes_per_row": c.width * 4}, size),
                              dtype=np.float32).reshape(c.height, c.width)
        return rgba[..., :3].copy(), depth.copy()
