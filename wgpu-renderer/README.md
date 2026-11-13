# Replica renderer on WebGPU

A native renderer for the Replica scenes in Python on [wgpu](https://github.com/pygfx/wgpu-py)
(WebGPU over Metal, Direct3D 12 or Vulkan). It replaces the OpenGL 4.3 `ReplicaRenderer`, which
cannot run on macOS, and needs nothing compiled: the environment is Python, NumPy, Pillow and
wgpu's prebuilt wheels (about 180 MB).

```bash
cd wgpu-renderer
pixi run view /path/to/replica_v1/room_0      # interactive window
pixi run render /path/to/replica_v1/room_0 out/room0 --frames 100
pixi run python -m pytest tests          # synthetic tests, no dataset needed
REPLICA_SCENE=/path/to/replica_v1/room_0 pixi run python -m pytest tests   # plus real-data checks
# plus a comparison with the original renderer's output (NICE-SLAM's Replica/room0 folder):
REPLICA_SCENE=.../room_0 NICE_SLAM_ROOM0=.../Replica/room0 pixi run python -m pytest tests
pixi run python benchmark.py /path/to/replica_v1/room_0
```

`render` writes the layout pyslam's `ReplicaDataset` reads (the NICE-SLAM one):
`results/frame%06d.jpg`, `results/depth%06d.png` (16-bit, metres × 6553.5) and `traj.txt`
(camera-to-world, 16 numbers per line, row-major), with pyslam's `settings/REPLICA.yaml` camera
(1200×680, fx = fy = 600) by default. From Python:

```python
from replica_wgpu import Camera, Renderer
r = Renderer("replica_v1/room_0", Camera.replica_pyslam())
rgb, depth = r.render(T_world_camera)  # (680,1200,3) uint8, (680,1200) float32 metres
```

Poses are camera-to-world with the camera axes of ReplicaSDK and OpenCV (x right, y down,
z forward), the convention of NICE-SLAM's `traj.txt` (verified, see below).

## Choosing the GPU

By default the renderer takes the fastest adapter. `--adapter` (on `view`, `render` and
`benchmark.py`), `Renderer(..., adapter=...)` or the `REPLICA_ADAPTER` environment variable
picks another: an adapter type (`discrete-gpu`, `integrated-gpu`, `cpu`), a backend (`vulkan`,
`metal`, `d3d12`) or any part of a device name (`5090`, `radv`, `llvmpipe`), ignoring case.
`cpu` is a software rasteriser where one is installed (Mesa's llvmpipe on Linux), for machines
without a usable GPU. A name that matches nothing fails with the list of adapters present. The
startup line `GPU: ...` shows which one is in use.

Off-screen rendering (`render`, `benchmark.py`, the Python API, the tests) works on any adapter.
The viewer window also needs the adapter to be able to draw to the screen: on the Linux test
machine `view --adapter integrated-gpu` stops with wgpu's `Invalid surface` error (most
likely because the monitor is on the other card), while `--adapter cpu` and the default work.

```bash
pixi run render /path/to/replica_v1/room_0 out/room0 --adapter integrated-gpu
REPLICA_ADAPTER=cpu pixi run python -m pytest tests     # the tests on software Vulkan
```

## Viewer

`pixi run view <scene>` opens a window: drag with the left mouse button or use the arrow keys
to look around, W/A/S/D to move, Q/E down/up, Shift to move or turn faster, +/- to change the
exposure, M to toggle mirrors and glass (on in the viewer), R to reset, Esc to quit. The title
bar shows the frame rate; it is capped at the display's refresh rate (vsync), so a 75 Hz
monitor shows about 75 fps. It uses [rendercanvas](https://github.com/pygfx/rendercanvas) with GLFW, from prebuilt wheels.

## Data needed per scene

`mesh.ply`, `textures/` and **`habitat/sorted_faces.bin`**. The last is not in the main Replica
archive but in `additional_habitat_configs.zip` (which `download.sh` also fetches). It is
required: atlas tile *i* holds face *i* of its sub-mesh in the order produced by ReplicaSDK's
unstable `std::sort`, which cannot be reproduced (see below).

## How it works

- **No geometry shader** (Metal and MoltenVK have none). Quads and positions live in storage
  buffers; the vertex shader pulls a quad's corners and emits two triangles in the order and with
  the UVs of the original `mesh-ptex.geom`, plus a flat face id.
- **Ptex lookup ported from `atlas.glsl`**: per-face adjacency and rotation, bilinear filtering
  across face borders with `textureLoad`.
- **Colour and metric depth in one pass** (two render targets); the OpenGL version draws twice.
- **HDR atlases** as `rgba16float` (wgpu has no RGB16F) or, with `--atlas rgb9e5`, as
  `rgb9e5ufloat` at half the memory.
- **Tone mapping** as PTexMesh (exposure, saturation, gamma), with exposure **0.01**:
  ReplicaViewer's default, and what the NICE-SLAM sequences were rendered with.
  `ReplicaRenderer` uses PTexMesh's 0.025, which is much brighter (`--exposure 0.025`).

## Mirrors and glass

`--mirrors` (or `Renderer(..., mirrors=True)`) draws the surfaces in the scene's `glass.sur`,
as ReplicaSDK's `MirrorRenderer`: for each surface in view, the scene is rendered reflected
about its plane (clipped to the camera's side) and blended over the surface's polygon with
alpha = Schlick Fresnel term on its reflectivity (1 for mirrors, room_0's frosted glass 0.3).
Mirrors are off by default because the NICE-SLAM sequences were rendered without them (their
frames match ours without mirrors, including at both of room_0's surfaces).

Depth inside a mirror is a modelling choice, `--mirror-depth`:

| mode | depth inside a mirror | models |
|---|---|---|
| `surface` (default) | the mirror surface | `ReplicaRenderer`, whose depth pass ignores mirrors; geometric ground truth |
| `sensor` | the reflected scene (virtual geometry behind the mirror) | an ideal depth sensor, which sees the reflection; consistent with the colour image |
| `invalid` | 0 | the dropouts real sensors often give at mirrors |

Glass keeps its surface depth in every mode: room_0's is frosted, and what lies behind clear
glass is not in the scans anyway. For the literature on depth sensors and mirrors, see e.g.
Mirror3D (Tan et al., CVPR 2021), which found raw depth at mirrors missing or wrong in NYUv2,
ScanNet and Matterport3D, and Whelan et al. (SIGGRAPH 2018), where Replica's surfaces come from.

Checked on a synthetic mirror (tests) and on room_0: `sensor` depth reflected back about the
mirror lands on real geometry inside the room. There is no pixel reference: ReplicaRenderer's
own output with mirrors would need the OpenGL renderer (the Docker image). Like the original,
a surface's polygon covers its whole outline, so the grid frame of room_0's mirror is drawn
as mirror too. Cost on an M1: about +1 ms per frame close to a mirror and +3 ms with both of
room_0's surfaces in view (each reflection pass is scissored to the surface's screen area).

## Validation against the original renderer

NICE-SLAM's `Replica/room0` sequence (the one pyslam downloads) was rendered from `room_0` by
the original OpenGL renderer. Rendering its poses (`traj.txt`, frames 0 and 500):

| | |
|---|---|
| depth, median difference | 1.6 mm (1.8 mm); 98.6% (99.6%) of pixels within 1 cm |
| depth with the poses read as OpenGL camera axes instead | over 1 m: so `traj.txt` is OpenCV axes |
| colour at exposure 0.01 | PSNR 32.6 dB against the reference JPEGs, visually indistinguishable |
| colour at exposure 0.025 | PSNR 11.2 dB, 48% of pixels saturated (reference: 2%) |

The reference frames are not in the repository. `python fetch_reference.py <dest>` pulls single
frames and `traj.txt` out of NICE-SLAM's 12 GB `Replica.zip` with HTTP range requests (about
10 MB for two frames).

## Known scan artefacts (not renderer bugs)

Some oddities are in the Replica scans themselves: the original OpenGL renderer draws them
too. Before reporting a rendering bug, compare with a NICE-SLAM frame of the same place
(`fetch_reference.py`, then render that frame's pose from `traj.txt`).

In `room_0`, in the corner to the right of the sideboard (NICE-SLAM `room0` frame 80 shows
all three, identically to this renderer):

- two spoked, wheel-like marks on the carpet;
- a faint spoked dome shape on the wall above the skirting board;
- small fragments floating above the vase on the sideboard.

The likely cause is objects too thin to reconstruct (something made of wire in the corner,
twigs in the vase): their appearance ended up in the textures of the surfaces behind them,
and the twigs left only a few stray pieces of mesh.

## Performance (room_0, Apple M1, Metal, 2026-11-13)

| | |
|---|---|
| mesh load (954k vertices, 954k quads, 6 sub-meshes) | 0.8 s |
| startup, `rgba16float` / `rgb9e5` | 4.4 s / 13.9 s (the RGB9E5 packing runs in NumPy) |
| GPU textures, `rgba16float` / `rgb9e5` | 1.96 GB / 0.98 GB |
| frame, 1200×680 (render + copy both images back) | 17.8 ms median, 56 fps |
| frame, 1280×960 | 22.4 ms median, 45 fps |
| `rgb9e5` against `rgba16float` | PSNR 63.8 dB, max difference 3/255, depth identical |
| peak process memory | 1.9 GB |

### Linux (room_0, 1200×680, Ubuntu 26.04, Vulkan, 2026-11-13)

One desktop (Ryzen 9 9950X, 32 threads) with three adapters, chosen with `--adapter`.
`benchmark.py`, median over 30 frames of render + copy both images back, `rgba16float`:

| adapter | startup | frame | with `--mirrors` | peak process memory |
|---|---|---|---|---|
| NVIDIA RTX 5090 (driver 595.99) | 2.0 s | 3.4 ms (292 fps) | 3.6 ms (278 fps) | 1.1 GB |
| AMD integrated, RADV (Mesa 26.0.8) | 2.0 s | 15.3 ms (65 fps) | 15.8 ms (63 fps) | 1.1 GB |
| llvmpipe, software (LLVM 21.1.8) | 2.2 s | 108 ms (9.2 fps) | 131 ms (7.6 fps) | 4.3 GB |
| llvmpipe, 4 threads (`LP_NUM_THREADS=4`) | 2.2 s | 112 ms (8.9 fps) | 132 ms (7.6 fps) | 4.3 GB |
| llvmpipe, 2 threads | 2.2 s | 135 ms (7.4 fps) | 160 ms (6.2 fps) | 4.3 GB |

- Software rendering works and passes every test, at 7 to 9 frames per second here: enough to
  pre-render sequences, slow for the viewer. It hardly depends on the thread count, but these
  are fast cores; a laptop CPU will be slower per thread, so treat the figures as a best case.
- Software rendering keeps the textures in ordinary memory: 4.3 GB at peak, or 2.5 GB with
  `--atlas rgb9e5` (same frame time, 10.5 s startup for the NumPy packing).
- On the integrated GPU `--atlas rgb9e5` is faster as well as smaller: 11.1 ms (90 fps).
- All three adapters pass the full test suite, including the comparison with NICE-SLAM's
  frames (`REPLICA_ADAPTER=... pytest`).

## Status and limitations

- Tested on an Apple M1 (Metal) and one Linux desktop (Vulkan: NVIDIA, AMD integrated and
  llvmpipe). Windows (D3D12), WSL2 and a real low-spec laptop are untested. On WSLg, pyslam
  found wgpu's default surface path crashes and uses software Vulkan for Rerun; off-screen
  rendering (no surface) may avoid that, but the viewer window would not.
- Validated against one scene (room_0) through NICE-SLAM's frames; other scenes, and the
  remaining 1.6 mm median depth difference (pixel-centre convention or quantisation), not yet
  investigated.
- Only HDR atlases (`*.hdr`), which is what Replica v1 ships.
- Not yet run through pyslam itself.

## Next steps

1. Run on Windows/WSL2 machines and a low-spec laptop (Linux desktop done).
2. Docker image with the OpenGL renderer, for reference frames on other scenes and settings.
3. Ship pre-packed scenes (RGB9E5 or BC6H-compressed atlases, plus `sorted_faces.bin`) so
   students download one small scene with no conversion; check the Replica licence for
   redistribution first.
4. Package for pixi alongside pyslam; a live data source for pyslam.
