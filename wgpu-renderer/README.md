# Replica renderer on WebGPU (spike)

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
z forward).

## Viewer

`pixi run view <scene>` opens a window: drag with the left mouse button to look around,
W/A/S/D to move, Q/E down/up, Shift to move faster, R to reset, Esc to quit. The title bar
shows the frame rate (about 75 fps at 1200×680 on an Apple M1). It uses
[rendercanvas](https://github.com/pygfx/rendercanvas) with GLFW, from prebuilt wheels.

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

## Results (room_0, Apple M1, Metal, 2026-11-13)

| | |
|---|---|
| mesh load (954k vertices, 954k quads, 6 sub-meshes) | 0.8 s |
| startup, `rgba16float` / `rgb9e5` | 4.4 s / 13.9 s (the RGB9E5 packing runs in NumPy) |
| GPU textures, `rgba16float` / `rgb9e5` | 1.96 GB / 0.98 GB |
| frame, 1200×680 (render + copy both images back) | 17.8 ms median, 56 fps |
| frame, 1280×960 | 22.4 ms median, 45 fps |
| `rgb9e5` against `rgba16float` | PSNR 63.8 dB, max difference 3/255, depth identical |
| peak process memory | 1.9 GB |

## Status and limitations

- Tested only on an Apple M1 (Metal). Linux (Vulkan, and Mesa's lavapipe without a GPU),
  Windows (D3D12) and WSL2 are untested. On WSLg, pyslam found wgpu's default surface path crashes
  and uses software Vulkan for Rerun; this renderer is off-screen (no surface), which may avoid that.
- Not yet compared pixel-by-pixel against the OpenGL `ReplicaRenderer`. The planned Docker image
  is for producing those reference frames. Sub-pixel conventions (pixel centres, near/far) are the
  main thing to check.
- Mirrors and glass (`glass.sur`) are not rendered yet; the OpenGL renderer has a reflection pass.
- Only HDR atlases (`*.hdr`), which is what Replica v1 ships.
- The camera rotation convention of NICE-SLAM's `traj.txt` is unverified; positions, which
  trajectory error uses, are unaffected.
- Not yet run through pyslam itself.

## Next steps

1. Run on Linux/Windows/WSL2 machines, ideally including a low-spec laptop.
2. Docker image with the OpenGL renderer for reference frames; compare.
3. Ship pre-packed scenes (RGB9E5 or BC6H-compressed atlases, plus `sorted_faces.bin`) so
   students download one small scene with no conversion; check the Replica licence for
   redistribution first.
4. Mirrors; package for pixi alongside pyslam; a live data source for pyslam.
