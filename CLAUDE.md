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
  commits rather than re-typing them).

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
  macOS's OpenGL tops out at 4.1, so nothing can render natively on a Mac: Mac
  students need a Linux VM/container (Mesa llvmpipe) until the renderer is ported.
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

## Upstream PR status (as of 2026-11)

- #114 (christian-rauch, Pangolin 0.9 API / CMake 3.22): merged into this fork, then
  extended.
- #94: subset of #114 without the submodule bump; not taken.
- #112: changes the Windows download drive to `D:`; not taken.

## Roadmap

1. Toolchain modernisation (CMake 4, latest Pangolin/Eigen, portability fixes).
2. Docker image (Ubuntu + Mesa llvmpipe, x86_64 + arm64) as the common environment
   for headless rendering on every platform.
3. Real-time mode: renderer as a server that student code drives over a socket.
4. Investigate porting the renderer off legacy OpenGL (e.g. VulkanSceneGraph, which
   would reach macOS via MoltenVK).
