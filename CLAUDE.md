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
- Mirrors and glass (`glass.sur`) are **off by default**: NICE-SLAM's frames were rendered
  without them. Depth inside mirrors has three modes (`surface`, the ReplicaRenderer default;
  `sensor`, the reflected scene; `invalid`, 0); glass keeps its surface depth. No pixel
  reference exists for mirrors yet (would need the OpenGL renderer).
The test data used during development was `room_0`, extracted to a `replica_v1/` folder next
to the repo (outside git).

## Upstream PR status (as of 2026-11)

- #114 (christian-rauch, Pangolin 0.9 API / CMake 3.22): merged into this fork, then
  extended.
- #94: subset of #114 without the submodule bump; not taken.
- #112: changes the Windows download drive to `D:`; not taken.

## Roadmap

Done:

- Toolchain modernisation: CMake 4, Pangolin 0.9.6, Eigen 5.0.1, portability fixes
  (PR #1).
- Native renderer and viewer on wgpu-py in `wgpu-renderer/` (PR #3): no geometry shader
  (vertex pulling), validated against NICE-SLAM's `room0` frames, writes pyslam's
  Replica layout, and has a Python API (`Renderer.render(pose) -> rgb, depth`) that the
  real-time mode can build on.

Next, roughly in order:

1. **Mirrors and glass** (`glass.sur`): the OpenGL renderer's reflection pass, ported.
2. **Cross-platform testing**: Linux (Vulkan, and lavapipe without a GPU), Windows
   (D3D12), WSL2, and a low-spec laptop. The biggest open question for students.
3. **Scene distribution for students**: per-scene downloads rather than the 34 GB
   archive, with atlases pre-packed (RGB9E5 or BC6H) and `sorted_faces.bin` included.
   Check the Replica licence for redistribution first.
4. **pyslam integration**: the renderer as a live pyslam data source; pixi packaging and
   a course branch, as in pyslam.
5. **Docker, for development and testing only**, now lower priority: NICE-SLAM's frames
   already provide reference output for the scenes it covers (room0-2, office0-4). Still
   useful for OpenGL reference frames of the other scenes and for Linux CI.
