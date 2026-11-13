# CLAUDE.md

Working notes for this fork of `facebookresearch/Replica-Dataset`. Keep this file
current; it is how context moves between machines.

## Purpose

Teaching use: students on Windows (via WSL2), macOS (including Apple Silicon) and
Linux use the Replica SDK to

1. generate pre-rendered views of the Replica scenes (first), and later
2. run the renderer in a "real-time" mode driven by separate, student-written code.

Every change should be judged by whether it works across all of those platforms.
Upstream is effectively unmaintained, so fixes land here rather than waiting on it.

Design principles:

- **Native and small for students.** Many students have resource-constrained
  machines, so what they run must work natively, without Docker or a VM, and must be
  small. **Docker is for development and testing only** (Linux verification, CI,
  reference images), never the student route.
- **Students build as little as possible.** Follow the companion project
  [sjulier/pyslam](https://github.com/sjulier/pyslam): pixi environments from a lock
  file, with any native C++ shipped as prebuilt per-platform bundles (GitHub release
  assets, falling back to a source build). See pyslam's `docs/TEACHING_SETUP.md` and
  `scripts/native_bundle.py`.
- **Interoperate with pyslam.** pyslam's `ReplicaDataset` (`pyslam/io/dataset.py`)
  reads the NICE-SLAM layout: `<scene>/results/frame%06d.jpg`,
  `<scene>/results/depth%06d.png` (16-bit, scale 6553.5) and `<scene>/traj.txt`, with
  the camera in pyslam's `settings/REPLICA.yaml` (1200x680, fx=fy=600). Output from
  this repo should drop into that loader without conversion, and the real-time mode
  should eventually be usable as a pyslam data source.

## Git workflow

- The fork is `sjulier/Replica-Dataset`; `main` is its default branch.
- Make each change on its own branch off `main` (e.g. `build/...`, `docker/...`,
  `renderer/...`) and open a PR into `sjulier/Replica-Dataset:main`. Don't commit
  directly to `main`.
- Push regularly, including work in progress, so the work survives machine switches.
- **PRs target the fork, never upstream.** In a fork, `gh pr create` defaults to the
  parent repo. On every fresh clone run once:
  `gh repo set-default sjulier/Replica-Dataset`
  and/or pass `-R sjulier/Replica-Dataset --base main` explicitly.
- Keep upstream authorship when pulling in upstream PRs (merge or cherry-pick their
  commits rather than re-typing them). Merge PRs with a merge commit, not a squash,
  for the same reason; delete the branch after merging.

### Release model (same as pyslam)

- `main` is development.
- Each course run gets a long-lived branch named after it, matching pyslam
  (e.g. `comp0227-2026`). It is what students clone, and it **only receives bug
  fixes**, cherry-picked or merged from `main` by PR. Create it when there is first
  something to ship to students.
- Tag the course branch whenever something ships, e.g. `comp0227-2026-lab03`, so a
  lab's instructions can pin an exact version.
- Prebuilt native bundles, if any, are release assets on a dedicated tag (pyslam
  uses `native-bundles`).

## Building

```bash
git submodule update --init   # not --recursive: Pangolin's own submodules aren't needed
./build.sh                    # JOBS=<n>, BUILD_TYPE=<type> to override
```

- `build.sh` installs the pinned submodules `3rdparty/eigen` and `3rdparty/Pangolin`
  into `3rdparty/install/` (build trees in `3rdparty/build/`), then builds the SDK
  into `build/` (`build/bin`, `build/lib`). The top-level CMake searches
  `3rdparty/install` before system paths.
- Pinned versions: Pangolin v0.9.6, Eigen 5.0.1. To bump: check out the new tag in the
  submodule, rebuild on macOS and Linux, and commit the submodule pointer.
- `ReplicaRenderer` (EGL/X11) is Linux-only (`REPLICA_BUILD_RENDERER`, default ON on
  Linux). `ptex` and `ReplicaViewer` are platform-neutral at build level.
- There are no automated tests. Minimum verification is a clean `./build.sh` on macOS
  arm64 plus Linux (x86_64 and arm64).

## Platform constraints (learned the hard way)

- **OpenGL 4.3 is required** (`#version 430` shaders, shader storage buffers, plus
  legacy calls such as `glPushAttrib`/`glBegin` needing a compatibility context).
  macOS's OpenGL tops out at 4.1, so the current renderer cannot run natively on a
  Mac at all. This is the main reason for the renderer port (roadmap item 2); a
  Linux container is a developer workaround, not a student route.
- **No x86-only flags** (`-msse*`, etc.): Apple Silicon and ARM Linux VMs/containers
  are first-class targets.
- **Case-insensitive file systems** (macOS, Windows, WSL2 under `/mnt/c`): headers on
  the include path must not share a name with a system header. `Assert.h` was renamed
  `ReplicaAssert.h` because it shadowed `<assert.h>`.
- Use only standard C++17 (`std::filesystem`, not `<experimental/filesystem>`/`stdc++fs`)
  and guard Linux-only APIs (e.g. `MAP_POPULATE`).
- Pangolin builds with an unconditional `-Werror`. Optional Pangolin drivers are
  switched off in `build.sh` because whatever happens to be installed can break it
  (Homebrew's librealsense did). If a newer compiler trips `-Werror`, patch via
  `build.sh` rather than editing the submodule.
- An *installed* Pangolin only exports its GLEW include path to build-tree consumers,
  so Replica links `GLEW::GLEW` itself.
- EGL context selection (`ReplicaSDK/ptex/EGL.cpp`): NVIDIA CUDA device, then an X
  display if one opens, then the first EGL device (Mesa GPU or llvmpipe).
- **Atlas tiles depend on an unstable sort.** Tile *i* of atlas *k* is face *i* of
  sub-mesh *k* in the order `PTexMesh::SplitMesh`'s `std::sort` left it when the atlases
  were made (with gcc). Faces with equal Morton codes can come out in another order with
  another standard library (e.g. libc++ on macOS), putting textures on the wrong faces.
  `habitat/sorted_faces.bin` (from `additional_habitat_configs.zip`) records the real
  order; `wgpu-renderer` uses it, and the C++ code should too.

## wgpu renderer

`wgpu-renderer/` is a pixi project: a Python renderer on wgpu that runs natively on macOS
(Metal), Windows (D3D12) and Linux (Vulkan) with nothing compiled, plus an interactive viewer
window. See its README for usage, validation, performance and limitations. Run its tests with
`pixi run python -m pytest tests` from that folder.

- Validated against NICE-SLAM's `room0` frames (rendered by the original renderer): depth
  within 1.6 mm median, colour 32.6 dB PSNR. That comparison also showed NICE-SLAM's
  `traj.txt` is camera-to-world in OpenCV axes, and that it was rendered at exposure 0.01
  (ReplicaViewer's default), not ReplicaRenderer's 0.025. Keep the default at 0.01.
The test data used during development was `room_0`, extracted to a `replica_v1/` folder next
to the repo (outside git).

## Upstream PR status (as of 2026-11)

- #114 (christian-rauch, Pangolin 0.9 API / CMake 3.22): merged into this fork, then
  extended.
- #94: subset of #114 without the submodule bump; not taken.
- #112: changes the Windows download drive to `D:`; not taken.

## Roadmap

1. ~~Toolchain modernisation (CMake 4, latest Pangolin/Eigen, portability fixes).~~
   Done in PR #1.
2. Renderer spike: a native renderer for every student platform. Leading candidate
   is **wgpu-py** (WebGPU: Metal / D3D12 / Vulkan; prebuilt wheels for every target
   including Linux and Windows on ARM, so students compile nothing; pyslam's Rerun
   viewer already runs on the same wgpu stack). Alternative: VulkanSceneGraph (C++,
   macOS via MoltenVK, would need native bundles). Neither Metal nor MoltenVK has
   geometry shaders, so quads are split on the CPU at load time with a per-vertex
   face id. Validate against the existing OpenGL renderer's output.
3. Docker image (Ubuntu + Mesa llvmpipe, x86_64 + arm64), for development and
   testing only: verifies Linux builds and produces reference images from the
   OpenGL renderer.
4. Output compatible with pyslam's Replica loader (configurable intrinsics and
   resolution, `traj.txt`).
5. Real-time mode: a Python API that student code (and pyslam) drives directly,
   e.g. `render(pose) -> rgb, depth`.
6. Packaging aligned with pyslam (pixi; native bundles only if native code remains).
