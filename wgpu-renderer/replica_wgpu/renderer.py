"""Off-screen Replica renderer on WebGPU (wgpu): Metal, D3D12 or Vulkan.

    r = Renderer("replica_v1/room_0", Camera.replica_pyslam())
    rgb, depth = r.render(T_world_camera)   # (H,W,3) uint8, (H,W) float32 metres

Poses use the camera convention of ReplicaSDK and OpenCV: x right, y down,
z forward. T_world_camera maps camera coordinates to world coordinates.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass

import numpy as np
import wgpu

from .mirrors import MASK_SIZE, MirrorSurface, load_mirrors
from .scene import Scene, load_hdr_atlas, load_hdr_atlas_rgb9e5, load_scene

# Atlas formats on the GPU: (wgpu format, loader, bytes per texel)
ATLAS_FORMATS = {
    "rgba16float": (wgpu.TextureFormat.rgba16float, load_hdr_atlas, 8),
    "rgb9e5": (wgpu.TextureFormat.rgb9e5ufloat, load_hdr_atlas_rgb9e5, 4),
}

# What the depth image shows inside a mirror's polygon (glass always keeps its surface):
#   surface: the mirror surface itself, as ReplicaRenderer (the default)
#   sensor:  the reflected scene, as a depth sensor sees it (consistent with the colour)
#   invalid: 0, as the dropouts of real sensors
MIRROR_DEPTH_MODES = ("surface", "sensor", "invalid")

_DIR = os.path.dirname(__file__)
_FRAME_BYTES = 160
_MIRROR_BYTES = 224
_NO_CLIP = np.array([0.0, 0.0, 0.0, 1.0])


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


@dataclass
class _GpuMirror:
    surface: MirrorSurface
    quad: np.ndarray  # (4, 3) world rectangle as drawn
    frame_buf: object  # Frame uniform of the reflection pass
    frame_group: object
    mirror_buf: object  # Mirror uniform of mirror.wgsl
    mask_view: object


@dataclass
class _ReflectionTargets:
    color: object
    depth: object
    zbuffer: object
    groups: list  # mirror.wgsl bind group per mirror


def _norm(s: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def match_adapter(spec: str, infos: list[dict]) -> int:
    """Index of the adapter that `spec` names, among adapter-info dicts.

    `spec` is matched without regard to case or punctuation: first as a whole
    adapter type ("discrete-gpu", "integrated-gpu", "cpu") or backend ("vulkan",
    "metal", "d3d12", "opengl"), otherwise as part of the device name, type or
    backend ("5090", "llvmpipe", "radv"). When several match, the first that is
    not OpenGL wins: a GPU is often listed a second time as an OpenGL adapter,
    which lacks features the renderer needs.
    """
    want = _norm(spec)
    fields = [[_norm(i.get(k, "")) for k in ("adapter_type", "backend_type", "device")] for i in infos]
    hits = [n for n, f in enumerate(fields) if want and want in f[:2]]
    if not hits:
        hits = [n for n, f in enumerate(fields) if want and any(want in x for x in f)]
    if not hits:
        listing = "\n".join(f"  {describe_adapter(i)}" for i in infos) or "  (none)"
        raise ValueError(f"no GPU adapter matches {spec!r}. Available:\n{listing}")
    hits.sort(key=lambda n: fields[n][1] == "opengl")  # stable: keeps the listed order otherwise
    return hits[0]


def describe_adapter(info: dict) -> str:
    return f"{info.get('device', '?')} ({info.get('backend_type', '?')}, {info.get('adapter_type', '?')})"


def select_adapter(spec: str | None):
    """The adapter `spec` names (see match_adapter), or the high-performance default."""
    if not spec:
        return wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    adapters = wgpu.gpu.enumerate_adapters_sync()
    return adapters[match_adapter(spec, [a.info for a in adapters])]


class Renderer:
    # HDR tone mapping. Gamma and saturation are PTexMesh's HDR defaults. Exposure is
    # ReplicaViewer's 0.01 rather than PTexMesh's 0.025 (which ReplicaRenderer uses):
    # the NICE-SLAM Replica sequences that pyslam uses were rendered at 0.01 (PSNR
    # 32.6 dB against them at 0.01, 11.2 dB at 0.025).
    exposure = 0.01
    gamma = 1.6969
    saturation = 1.5

    def __init__(self, scene_dir: str, camera: Camera, cull: str = "back",
                 atlas_format: str = "rgba16float", mirrors: bool = False,
                 mirror_depth: str = "surface", verbose: bool = True,
                 adapter: str | None = None):
        """adapter: which GPU to use, e.g. "integrated-gpu", "cpu" (software) or part of
        a device name (see match_adapter); by default the REPLICA_ADAPTER environment
        variable, or else the high-performance adapter. mirrors: draw the mirrors and glass of the scene's glass.sur (off by default:
        the NICE-SLAM sequences were rendered without). mirror_depth: one of
        MIRROR_DEPTH_MODES, used when mirrors are drawn."""
        self.camera = camera
        self.atlas_format = atlas_format
        self.draw_mirrors = mirrors
        self.mirror_depth = mirror_depth
        self.verbose = verbose
        t0 = time.perf_counter()
        scene = load_scene(scene_dir)
        self._log(f"loaded mesh: {len(scene.submeshes)} sub-meshes, "
                  f"{sum(s.quads.shape[0] for s in scene.submeshes)} quads ({time.perf_counter() - t0:.1f} s)")
        self.tile_size = scene.tile_size
        sur = os.path.join(scene_dir, "glass.sur")
        surfaces = load_mirrors(sur) if os.path.exists(sur) else []

        self.adapter = select_adapter(adapter or os.environ.get("REPLICA_ADAPTER"))
        self._log(f"GPU: {describe_adapter(self.adapter.info)}")
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
        self.mirrors = self._upload_mirrors(surfaces)
        if surfaces:
            n = sum(s.is_mirror for s in surfaces)
            self._log(f"glass.sur: {n} mirror(s), {len(surfaces) - n} glass surface(s)")
        self._reflection_targets: dict[tuple[int, int], _ReflectionTargets] = {}
        self._make_targets()

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    # ---- setup --------------------------------------------------------------

    def _make_pipeline(self, cull: str) -> None:
        d = self.device
        with open(os.path.join(_DIR, "ptex.wgsl")) as f:
            self._module = d.create_shader_module(code=f.read())
        with open(os.path.join(_DIR, "mirror.wgsl")) as f:
            self._mirror_module = d.create_shader_module(code=f.read())
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
        frag = wgpu.ShaderStage.FRAGMENT
        self.mirror_layout = d.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.VERTEX | frag,
             "buffer": {"type": wgpu.BufferBindingType.uniform}},
            {"binding": 1, "visibility": frag, "texture": {"sample_type": wgpu.TextureSampleType.float}},
            {"binding": 2, "visibility": frag,
             "texture": {"sample_type": wgpu.TextureSampleType.unfilterable_float}},
            {"binding": 3, "visibility": frag, "texture": {"sample_type": wgpu.TextureSampleType.float}},
            {"binding": 4, "visibility": frag, "sampler": {"type": wgpu.SamplerBindingType.filtering}},
        ])
        self._mask_sampler = d.create_sampler(mag_filter="linear", min_filter="linear")  # clamp to edge
        self._cull, self._pipelines = cull, {}
        self.pipeline = self.pipeline_for(wgpu.TextureFormat.rgba8unorm)
        self.frame_buf = d.create_buffer(size=_FRAME_BYTES, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self.frame_group = d.create_bind_group(layout=self.frame_layout, entries=[
            {"binding": 0, "resource": {"buffer": self.frame_buf}}])

    def _mesh_pipeline(self, color_format: str, reflected: bool) -> object:
        d, module = self.device, self._module
        # Replica quads are counter-clockwise in OpenGL's y-up window space. The
        # projection flips y (WebGPU framebuffers are y-down), so they arrive clockwise;
        # a reflection flips them back.
        front = wgpu.FrontFace.ccw if reflected else wgpu.FrontFace.cw
        return d.create_render_pipeline(
            layout=d.create_pipeline_layout(bind_group_layouts=[self.frame_layout, self.mesh_layout]),
            vertex={"module": module, "entry_point": "vs_main", "buffers": []},
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list,
                       "front_face": front, "cull_mode": "none" if reflected else self._cull},
            depth_stencil={"format": wgpu.TextureFormat.depth32float,
                           "depth_write_enabled": True,
                           "depth_compare": wgpu.CompareFunction.less},
            multisample=None,
            fragment={"module": module, "entry_point": "fs_clipped" if reflected else "fs_main",
                      "targets": [{"format": color_format}, {"format": wgpu.TextureFormat.r32float}]},
        )

    def _mirror_pipeline(self, color_format: str, depth_pass: bool) -> object:
        """The mirror quad: blends the reflection into the colour target or, for
        depth_pass, replaces the depth target inside the polygon."""
        d, module = self.device, self._mirror_module
        if depth_pass:
            targets = [{"format": color_format, "write_mask": 0},
                       {"format": wgpu.TextureFormat.r32float}]
        else:
            blend = {
                "color": {"src_factor": wgpu.BlendFactor.src_alpha,
                          "dst_factor": wgpu.BlendFactor.one_minus_src_alpha,
                          "operation": wgpu.BlendOperation.add},
                "alpha": {"src_factor": wgpu.BlendFactor.zero, "dst_factor": wgpu.BlendFactor.one,
                          "operation": wgpu.BlendOperation.add},
            }
            targets = [{"format": color_format, "blend": blend},
                       {"format": wgpu.TextureFormat.r32float, "write_mask": 0}]
        return d.create_render_pipeline(
            layout=d.create_pipeline_layout(bind_group_layouts=[self.mirror_layout]),
            vertex={"module": module, "entry_point": "vs_mirror", "buffers": []},
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list, "cull_mode": wgpu.CullMode.none},
            # The depth pass draws the same quad again after the colour pass wrote its z.
            depth_stencil={"format": wgpu.TextureFormat.depth32float,
                           "depth_write_enabled": not depth_pass,
                           "depth_compare": wgpu.CompareFunction.less_equal if depth_pass
                           else wgpu.CompareFunction.less},
            multisample=None,
            fragment={"module": module, "entry_point": "fs_mirror_depth" if depth_pass else "fs_mirror",
                      "targets": targets},
        )

    def _cached(self, key: tuple, make) -> object:
        if key not in self._pipelines:
            self._pipelines[key] = make()
        return self._pipelines[key]

    def pipeline_for(self, color_format: str) -> object:
        """The render pipeline for a colour target format (e.g. a window's), built once."""
        return self._cached(("mesh", color_format), lambda: self._mesh_pipeline(color_format, False))

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

    def _upload_mirrors(self, surfaces: list[MirrorSurface]) -> list[_GpuMirror]:
        d = self.device
        uniform = wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        out = []
        for s in surfaces:
            mask = d.create_texture(size=(MASK_SIZE, MASK_SIZE, 1), format=wgpu.TextureFormat.r8unorm,
                                    usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
            d.queue.write_texture({"texture": mask}, s.mask(), {"bytes_per_row": MASK_SIZE},
                                  (MASK_SIZE, MASK_SIZE, 1))
            frame_buf = d.create_buffer(size=_FRAME_BYTES, usage=uniform)
            out.append(_GpuMirror(
                surface=s, quad=s.quad(), frame_buf=frame_buf,
                frame_group=d.create_bind_group(layout=self.frame_layout, entries=[
                    {"binding": 0, "resource": {"buffer": frame_buf}}]),
                mirror_buf=d.create_buffer(size=_MIRROR_BYTES, usage=uniform),
                mask_view=mask.create_view()))
        return out

    def _make_targets(self) -> None:
        d, c = self.device, self.camera
        size = (c.width, c.height, 1)
        rt = wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC
        self.color_tex = d.create_texture(size=size, format=wgpu.TextureFormat.rgba8unorm, usage=rt)
        self.depth_out = d.create_texture(size=size, format=wgpu.TextureFormat.r32float, usage=rt)
        self.zbuffer = d.create_texture(size=size, format=wgpu.TextureFormat.depth32float,
                                        usage=wgpu.TextureUsage.RENDER_ATTACHMENT)

    def _reflection_targets_for(self, w: int, h: int) -> _ReflectionTargets:
        """Off-screen targets for the reflection passes (shared by all mirrors), per size."""
        if (w, h) not in self._reflection_targets:
            d, size = self.device, (w, h, 1)
            rt = wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING
            color = d.create_texture(size=size, format=wgpu.TextureFormat.rgba8unorm, usage=rt)
            depth = d.create_texture(size=size, format=wgpu.TextureFormat.r32float, usage=rt)
            zbuffer = d.create_texture(size=size, format=wgpu.TextureFormat.depth32float,
                                       usage=wgpu.TextureUsage.RENDER_ATTACHMENT)
            groups = [d.create_bind_group(layout=self.mirror_layout, entries=[
                {"binding": 0, "resource": {"buffer": m.mirror_buf}},
                {"binding": 1, "resource": color.create_view()},
                {"binding": 2, "resource": depth.create_view()},
                {"binding": 3, "resource": m.mask_view},
                {"binding": 4, "resource": self._mask_sampler},
            ]) for m in self.mirrors]
            self._reflection_targets[(w, h)] = _ReflectionTargets(color, depth, zbuffer, groups)
        return self._reflection_targets[(w, h)]

    # ---- rendering ------------------------------------------------------------

    def _frame_uniform(self, mvp: np.ndarray, mv: np.ndarray, clip_plane: np.ndarray) -> bytes:
        frame = np.zeros(_FRAME_BYTES // 4, dtype=np.float32)
        frame[0:16] = mvp.T.reshape(-1)  # WGSL matrices are column-major
        frame[16:32] = mv.T.reshape(-1)
        frame[32:35] = (self.exposure, 1.0 / self.gamma, self.saturation)
        frame[35:36] = np.array([self.tile_size], dtype=np.int32).view(np.float32)
        frame[36:40] = clip_plane
        return frame.tobytes()

    def _mirror_uniform(self, m: _GpuMirror, mvp: np.ndarray, mv: np.ndarray) -> bytes:
        u = np.zeros(_MIRROR_BYTES // 4, dtype=np.float32)
        u[0:16] = mvp.T.reshape(-1)
        u[16:32] = mv.T.reshape(-1)
        u[32:48] = np.c_[m.quad, np.ones(4)].reshape(-1)
        u[48:51] = m.surface.plane[:3]
        u[52] = m.surface.reflectivity
        u[53:54] = np.array([0 if self.mirror_depth == "sensor" else 1], dtype=np.uint32).view(np.float32)
        return u.tobytes()

    @staticmethod
    def _screen_rect(quad: np.ndarray, mvp: np.ndarray, w: int, h: int) -> tuple[int, int, int, int] | None:
        """The pixel rectangle a quad can cover, or None if it is outside the view frustum."""
        x, y, z, cw = mvp @ np.c_[quad, np.ones(4)].T  # clip space, one column per corner
        if (np.all(x > cw) or np.all(x < -cw) or np.all(y > cw) or np.all(y < -cw)
                or np.all(z < 0) or np.all(z > cw)):
            return None
        if np.any(cw <= 1e-6):  # crosses the camera plane: no bound on screen
            return 0, 0, w, h
        px = (x / cw + 1) / 2 * w
        py = (1 - y / cw) / 2 * h
        x0, x1 = max(int(np.floor(px.min())), 0), min(int(np.ceil(px.max())) + 1, w)
        y0, y1 = max(int(np.floor(py.min())), 0), min(int(np.ceil(py.max())) + 1, h)
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1 - x0, y1 - y0

    def draw(self, T_world_camera: np.ndarray, camera: Camera, color_tex, depth_tex, zbuffer_tex) -> None:
        """Render into the given targets (all camera-sized): colour in any format with a
        pipeline from pipeline_for(), r32float depth, depth32float z-buffer."""
        if self.mirror_depth not in MIRROR_DEPTH_MODES:
            raise ValueError(f"mirror_depth must be one of {MIRROR_DEPTH_MODES}")
        T = np.asarray(T_world_camera, dtype=np.float64)
        mv = np.linalg.inv(T)
        P = camera.projection()
        mvp = P @ mv
        d = self.device
        d.queue.write_buffer(self.frame_buf, 0, self._frame_uniform(mvp, mv, _NO_CLIP))
        w, h = camera.width, camera.height
        visible = []
        if self.draw_mirrors:
            for k, m in enumerate(self.mirrors):
                rect = self._screen_rect(m.quad, mvp, w, h)
                if rect is not None:
                    visible.append((k, rect))

        enc = d.create_command_encoder()
        rp = enc.begin_render_pass(
            color_attachments=[
                {"view": color_tex.create_view(), "load_op": wgpu.LoadOp.clear,
                 "store_op": wgpu.StoreOp.store, "clear_value": (0, 0, 0, 1)},
                {"view": depth_tex.create_view(), "load_op": wgpu.LoadOp.clear,
                 "store_op": wgpu.StoreOp.store, "clear_value": (0, 0, 0, 0)},
            ],
            depth_stencil_attachment={"view": zbuffer_tex.create_view(), "depth_clear_value": 1.0,
                                      "depth_load_op": wgpu.LoadOp.clear,
                                      "depth_store_op": wgpu.StoreOp.store if visible else wgpu.StoreOp.discard},
        )
        rp.set_pipeline(self.pipeline_for(color_tex.format))
        rp.set_bind_group(0, self.frame_group)
        for sm in self.submeshes:
            rp.set_bind_group(1, sm.bind_group)
            rp.draw(sm.num_vertices)
        rp.end()

        if visible:
            self._draw_mirrors(enc, visible, T, mv, P, mvp, color_tex, depth_tex, zbuffer_tex, w, h)
        d.queue.submit([enc.finish()])

    def _draw_mirrors(self, enc, visible, T, mv, P, mvp, color_tex, depth_tex, zbuffer_tex, w, h) -> None:
        """For each mirror in view, as ReplicaSDK's MirrorRenderer: render the scene
        reflected about its plane, then draw its rectangle over the image."""
        d = self.device
        tg = self._reflection_targets_for(w, h)
        rgba8 = wgpu.TextureFormat.rgba8unorm
        refl_pipeline = self._cached(("reflected", rgba8), lambda: self._mesh_pipeline(rgba8, True))
        fmt = color_tex.format
        color_pipeline = self._cached(("mirror", fmt), lambda: self._mirror_pipeline(fmt, False))
        depth_pipeline = self._cached(("mirror-depth", fmt), lambda: self._mirror_pipeline(fmt, True))
        for k, (x, y, rw, rh) in visible:
            m = self.mirrors[k]
            mv_r = mv @ m.surface.reflection_matrix()
            d.queue.write_buffer(m.frame_buf, 0,
                                 self._frame_uniform(P @ mv_r, mv_r, m.surface.clip_plane(T[:3, 3])))
            d.queue.write_buffer(m.mirror_buf, 0, self._mirror_uniform(m, mvp, mv))

            rp = enc.begin_render_pass(
                color_attachments=[
                    {"view": tg.color.create_view(), "load_op": wgpu.LoadOp.clear,
                     "store_op": wgpu.StoreOp.store, "clear_value": (0, 0, 0, 1)},
                    {"view": tg.depth.create_view(), "load_op": wgpu.LoadOp.clear,
                     "store_op": wgpu.StoreOp.store, "clear_value": (0, 0, 0, 0)},
                ],
                depth_stencil_attachment={"view": tg.zbuffer.create_view(), "depth_clear_value": 1.0,
                                          "depth_load_op": wgpu.LoadOp.clear,
                                          "depth_store_op": wgpu.StoreOp.discard},
            )
            rp.set_scissor_rect(x, y, rw, rh)
            rp.set_pipeline(refl_pipeline)
            rp.set_bind_group(0, m.frame_group)
            for sm in self.submeshes:
                rp.set_bind_group(1, sm.bind_group)
                rp.draw(sm.num_vertices)
            rp.end()

            rp = enc.begin_render_pass(
                color_attachments=[
                    {"view": color_tex.create_view(), "load_op": wgpu.LoadOp.load, "store_op": wgpu.StoreOp.store},
                    {"view": depth_tex.create_view(), "load_op": wgpu.LoadOp.load, "store_op": wgpu.StoreOp.store},
                ],
                depth_stencil_attachment={"view": zbuffer_tex.create_view(),
                                          "depth_load_op": wgpu.LoadOp.load,
                                          "depth_store_op": wgpu.StoreOp.store},
            )
            rp.set_scissor_rect(x, y, rw, rh)
            rp.set_bind_group(0, tg.groups[k])
            rp.set_pipeline(color_pipeline)
            rp.draw(6)
            if self.mirror_depth != "surface" and m.surface.is_mirror:
                rp.set_pipeline(depth_pipeline)
                rp.draw(6)
            rp.end()

    def render(self, T_world_camera: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Render colour (H,W,3 uint8) and depth along the optical axis (H,W float32, metres; 0 = nothing)."""
        c, d = self.camera, self.device
        self.draw(T_world_camera, c, self.color_tex, self.depth_out, self.zbuffer)
        size = (c.width, c.height, 1)
        rgba = np.frombuffer(d.queue.read_texture({"texture": self.color_tex}, {"bytes_per_row": c.width * 4}, size),
                             dtype=np.uint8).reshape(c.height, c.width, 4)
        depth = np.frombuffer(d.queue.read_texture({"texture": self.depth_out}, {"bytes_per_row": c.width * 4}, size),
                              dtype=np.float32).reshape(c.height, c.width)
        return rgba[..., :3].copy(), depth.copy()
