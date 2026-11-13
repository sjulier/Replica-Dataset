#!/usr/bin/env bash
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# Builds the pinned third-party dependencies (Eigen, Pangolin) into
# 3rdparty/install, then builds the Replica SDK into build/.
#
# Re-running is incremental. Environment overrides:
#   JOBS=<n>            parallel build jobs (default: number of CPUs)
#   BUILD_TYPE=<type>   CMake build type (default: Release)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="$ROOT/3rdparty/install"
DEPS_BUILD="$ROOT/3rdparty/build"
BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"

if [ ! -f "$ROOT/3rdparty/Pangolin/CMakeLists.txt" ] || [ ! -f "$ROOT/3rdparty/eigen/CMakeLists.txt" ]; then
  echo "Submodules missing; running: git submodule update --init" >&2
  git -C "$ROOT" submodule update --init
fi

echo "==> Eigen"
cmake -S "$ROOT/3rdparty/eigen" -B "$DEPS_BUILD/eigen" \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DBUILD_TESTING=OFF \
  -DEIGEN_BUILD_TESTING=OFF \
  -DEIGEN_BUILD_DOC=OFF \
  -DEIGEN_BUILD_DEMOS=OFF \
  -DEIGEN_BUILD_BLAS=OFF \
  -DEIGEN_BUILD_LAPACK=OFF \
  -DEIGEN_BUILD_PKGCONFIG=OFF \
  -DEIGEN_BUILD_CMAKE_PACKAGE=ON
cmake --build "$DEPS_BUILD/eigen" --target install

# Replica only uses Pangolin for windowing, GL helpers and JPEG/PNG output.
# Optional camera/video drivers are switched off so that whatever happens to
# be installed on a machine cannot break the build (e.g. Homebrew's
# librealsense headers fail Pangolin's -Werror).
echo "==> Pangolin"
cmake -S "$ROOT/3rdparty/Pangolin" -B "$DEPS_BUILD/pangolin" \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCMAKE_PREFIX_PATH="$PREFIX" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_TOOLS=OFF \
  -DBUILD_TESTS=OFF \
  -DBUILD_PANGOLIN_PYTHON=OFF \
  -DBUILD_PANGOLIN_LIBJPEG=ON \
  -DBUILD_PANGOLIN_LIBPNG=ON \
  -DBUILD_PANGOLIN_DEPTHSENSE=OFF \
  -DBUILD_PANGOLIN_FFMPEG=OFF \
  -DBUILD_PANGOLIN_LIBDC1394=OFF \
  -DBUILD_PANGOLIN_LIBOPENEXR=OFF \
  -DBUILD_PANGOLIN_LIBRAW=OFF \
  -DBUILD_PANGOLIN_LIBTIFF=OFF \
  -DBUILD_PANGOLIN_LIBUVC=OFF \
  -DBUILD_PANGOLIN_LZ4=OFF \
  -DBUILD_PANGOLIN_OPENNI=OFF \
  -DBUILD_PANGOLIN_OPENNI2=OFF \
  -DBUILD_PANGOLIN_PLEORA=OFF \
  -DBUILD_PANGOLIN_REALSENSE=OFF \
  -DBUILD_PANGOLIN_REALSENSE2=OFF \
  -DBUILD_PANGOLIN_TELICAM=OFF \
  -DBUILD_PANGOLIN_UVC_MEDIAFOUNDATION=OFF \
  -DBUILD_PANGOLIN_V4L=OFF \
  -DBUILD_PANGOLIN_ZSTD=OFF
cmake --build "$DEPS_BUILD/pangolin" -j "$JOBS" --target install

echo "==> Replica SDK"
cmake -S "$ROOT" -B "$ROOT/build" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
cmake --build "$ROOT/build" -j "$JOBS"

echo "Done. Binaries are in $ROOT/build/bin/"
