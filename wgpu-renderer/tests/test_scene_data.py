"""Tests on a real Replica scene; skipped unless REPLICA_SCENE points at one.

    REPLICA_SCENE=/path/to/replica_v1/room_0 pixi run python -m pytest tests
"""

import os

import numpy as np
import pytest

from replica_wgpu.scene import FACE_MASK, ROTATION_SHIFT, compute_adjacency, load_scene

SCENE = os.environ.get("REPLICA_SCENE")
pytestmark = pytest.mark.skipif(not SCENE, reason="set REPLICA_SCENE to a Replica scene folder")


def reference_adjacency(quads):
    """A literal transcription of PTexMesh::CalculateAdjacency (ReplicaSDK/ptex/PTexLib.cpp)."""
    edge_map, keys = {}, {}
    for f in range(len(quads)):
        for e in range(4):
            i0, i1 = int(quads[f, e]), int(quads[f, (e + 1) % 4])
            key = (min(i0, i1) << 32) | max(i0, i1)
            edge_map.setdefault(key, []).append((f, e))
            keys[(f, e)] = key
    out = np.zeros((len(quads), 4), dtype=np.uint32)
    for f in range(len(quads)):
        for e in range(4):
            adj = edge_map[keys[(f, e)]]
            adj_face = -1
            for af, _ in adj:
                if af != f:
                    adj_face = af
            rot = 0
            if len(adj) == 2:
                e0 = e1 = 0
                if adj[0][1] == e:
                    e0, e1 = adj[0][1], adj[1][1]
                elif adj[1][1] == e:
                    e0, e1 = adj[1][1], adj[0][1]
                rot = (e0 - e1 + 2) & 3
            out[f, e] = (rot << ROTATION_SHIFT) | (adj_face & FACE_MASK)
    return out


@pytest.fixture(scope="module")
def scene():
    return load_scene(SCENE)


def test_atlases_match_submeshes(scene):
    for sm in scene.submeshes:
        dim = int(round(np.sqrt(os.path.getsize(sm.atlas_path) / 6)))
        tiles = (dim // scene.tile_size) ** 2
        assert tiles >= sm.quads.shape[0], f"{sm.atlas_path}: {tiles} tiles < {sm.quads.shape[0]} faces"


def test_adjacency_matches_cpp(scene):
    quads = scene.submeshes[0].quads[:30000]  # a slice keeps the pure-Python reference quick
    np.testing.assert_array_equal(compute_adjacency(quads), reference_adjacency(quads))


# NICE-SLAM's room0 sequence was rendered from Replica's room_0 by the original
# OpenGL renderer: a reference for pose convention, geometry and tone mapping.
NICE_ROOM0 = os.environ.get("NICE_SLAM_ROOM0")  # .../Replica/room0 from NICE-SLAM's Replica.zip


@pytest.mark.skipif(not NICE_ROOM0, reason="set NICE_SLAM_ROOM0 (with REPLICA_SCENE = room_0)")
def test_matches_nice_slam_room0():
    from PIL import Image

    from replica_wgpu import Camera, Renderer

    poses = np.loadtxt(os.path.join(NICE_ROOM0, "traj.txt")).reshape(-1, 4, 4)
    r = Renderer(SCENE, Camera.replica_pyslam(), verbose=False)
    rgb, depth = r.render(poses[0])
    ref_rgb = np.asarray(Image.open(os.path.join(NICE_ROOM0, "results", "frame000000.jpg")).convert("RGB"))
    ref_depth = np.asarray(Image.open(os.path.join(NICE_ROOM0, "results", "depth000000.png"))) / 6553.5

    valid = (depth > 0) & (ref_depth > 0)
    err = np.abs(depth[valid] - ref_depth[valid])
    assert np.median(err) < 0.005 and np.mean(err < 0.01) > 0.95, "geometry or pose convention"
    mse = np.mean((rgb.astype(float) - ref_rgb.astype(float)) ** 2)
    assert 10 * np.log10(255**2 / mse) > 30, "tone mapping (exposure/gamma/saturation)"
