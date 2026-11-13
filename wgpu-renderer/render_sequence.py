"""Render a Replica scene along a trajectory, in the layout pyslam reads.

    python render_sequence.py <replica_v1>/room_0 out/room0 [--frames 100] [--traj traj.txt]

Writes out/room0/results/frame%06d.jpg, results/depth%06d.png (16-bit,
metres * 6553.5) and traj.txt (one camera-to-world 4x4 per line, row-major),
i.e. the NICE-SLAM layout of pyslam's ReplicaDataset. Without --traj the
camera turns on the spot at the centre of the scene.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
from PIL import Image

from replica_wgpu import Camera, Renderer
from replica_wgpu.renderer import MIRROR_DEPTH_MODES
from replica_wgpu.scene import read_ply

DEPTH_SCALE = 6553.5  # ReplicaSDK's 65535 * 0.1, and pyslam's DepthMapFactor


def spin_trajectory(scene_dir: str, frames: int) -> np.ndarray:
    """Turn a full circle at the centre of the bounding box, looking horizontally.

    Replica scenes are z-up; the camera's y axis (down) is world -z.
    """
    pos, _ = read_ply(os.path.join(scene_dir, "mesh.ply"))
    centre = (pos.min(axis=0) + pos.max(axis=0)) / 2
    poses = []
    for a in np.linspace(0, 2 * np.pi, frames, endpoint=False):
        fwd = np.array([np.cos(a), np.sin(a), 0.0])
        down = np.array([0.0, 0.0, -1.0])
        right = np.cross(down, fwd)
        T = np.eye(4)
        T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = right, down, fwd, centre
        poses.append(T)
    return np.stack(poses)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--traj", help="camera-to-world poses, 16 numbers per line (NICE-SLAM traj.txt)")
    ap.add_argument("--camera", choices=["pyslam", "sdk"], default="pyslam")
    ap.add_argument("--atlas", choices=["rgba16float", "rgb9e5"], default="rgba16float",
                    help="GPU texture format: rgb9e5 halves texture memory")
    ap.add_argument("--exposure", type=float, default=Renderer.exposure,
                    help="HDR exposure (default %(default)s, as the NICE-SLAM sequences)")
    ap.add_argument("--cull", choices=["back", "front", "none"], default="back")
    ap.add_argument("--mirrors", action="store_true",
                    help="draw the mirrors and glass of glass.sur (off: as the NICE-SLAM sequences)")
    ap.add_argument("--mirror-depth", choices=MIRROR_DEPTH_MODES, default="surface",
                    help="depth inside mirrors: the mirror surface (as ReplicaRenderer), the "
                         "reflected scene (as a depth sensor), or 0 (a sensor dropout)")
    args = ap.parse_args()

    cam = Camera.replica_pyslam() if args.camera == "pyslam" else Camera.replica_sdk()
    r = Renderer(args.scene_dir, cam, cull=args.cull, atlas_format=args.atlas,
                 mirrors=args.mirrors, mirror_depth=args.mirror_depth)
    r.exposure = args.exposure
    poses = (np.loadtxt(args.traj).reshape(-1, 4, 4) if args.traj
             else spin_trajectory(args.scene_dir, args.frames))[: args.frames]

    res = os.path.join(args.out_dir, "results")
    os.makedirs(res, exist_ok=True)
    t_render = 0.0
    for i, T in enumerate(poses):
        t0 = time.perf_counter()
        rgb, depth = r.render(T)
        t_render += time.perf_counter() - t0
        Image.fromarray(rgb).save(os.path.join(res, f"frame{i:06d}.jpg"), quality=95)
        d16 = np.clip(depth * DEPTH_SCALE + 0.5, 0, 65535).astype(np.uint16)
        Image.fromarray(d16).save(os.path.join(res, f"depth{i:06d}.png"))
        print(f"\rframe {i + 1}/{len(poses)}", end="", flush=True)
    np.savetxt(os.path.join(args.out_dir, "traj.txt"), poses.reshape(len(poses), 16), fmt="%.9f")
    print(f"\n{len(poses)} frames, {1000 * t_render / max(len(poses), 1):.1f} ms per frame "
          f"(render + read back), {cam.width}x{cam.height}")


if __name__ == "__main__":
    main()
