"""Time the renderer's phases on a real scene.

    python benchmark.py <replica_v1>/room_0 [--frames 30]
"""

import argparse
import sys
import time

import numpy as np

from render_sequence import spin_trajectory
from replica_wgpu import Camera, Renderer


def peak_memory_gb() -> float | None:
    """Peak resident memory of this process (not available on Windows)."""
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1e9 if sys.platform == "darwin" else 1e6)  # bytes on macOS, kB on Linux


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--camera", choices=["pyslam", "sdk"], default="pyslam")
    ap.add_argument("--atlas", choices=["rgba16float", "rgb9e5"], default="rgba16float",
                    help="GPU texture format: rgb9e5 halves texture memory")
    ap.add_argument("--exposure", type=float, default=Renderer.exposure,
                    help="HDR exposure (default %(default)s, as the NICE-SLAM sequences)")
    ap.add_argument("--mirrors", action="store_true", help="draw the mirrors and glass of glass.sur")
    ap.add_argument("--adapter", help="GPU to use: discrete-gpu, integrated-gpu, cpu (software), a backend, "
                                      "or part of a device name (default: $REPLICA_ADAPTER, else the fastest)")
    args = ap.parse_args()

    cam = Camera.replica_pyslam() if args.camera == "pyslam" else Camera.replica_sdk()
    t0 = time.perf_counter()
    r = Renderer(args.scene_dir, cam, atlas_format=args.atlas, mirrors=args.mirrors, adapter=args.adapter)
    r.exposure = args.exposure
    print(f"startup: {time.perf_counter() - t0:.1f} s")
    poses = spin_trajectory(args.scene_dir, args.frames)
    r.render(poses[0])  # warm-up (pipeline compilation)

    times = []
    for T in poses:
        t0 = time.perf_counter()
        r.render(T)
        times.append(time.perf_counter() - t0)
    ms = 1000 * np.array(times)
    print(f"{cam.width}x{cam.height}: render + read back {np.median(ms):.1f} ms median "
          f"({1000 / np.median(ms):.1f} fps), min {ms.min():.1f}, max {ms.max():.1f}")
    peak = peak_memory_gb()
    if peak is not None:
        print(f"peak process memory: {peak:.1f} GB")


if __name__ == "__main__":
    main()
