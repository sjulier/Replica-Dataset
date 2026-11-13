"""Smoke tests on a synthetic scene: no dataset needed.

The scene is the inside of a 2 m cube centred on the origin, one quad per wall,
each wall's atlas tile split into four coloured quadrants. A camera at the
centre with a 90 degree field of view sees exactly one wall.
"""

import json
import os

import numpy as np
import pytest

from replica_wgpu import Camera, Renderer
from replica_wgpu.scene import FACE_MASK, compute_adjacency, load_scene

TILE = 8
CORNERS = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float32)


def vid(x, y, z):
    return int(np.flatnonzero((CORNERS == [x, y, z]).all(axis=1))[0])


# Quads as (v0, v1, v2, v3): uv (0,0), (1,0), (1,1), (0,1). For the +x wall, seen
# from inside looking along +x with world +z up, v0 is top-left, v1 top-right.
QUADS = np.array([
    [vid(1, 1, 1), vid(1, -1, 1), vid(1, -1, -1), vid(1, 1, -1)],      # +x
    [vid(-1, -1, 1), vid(-1, 1, 1), vid(-1, 1, -1), vid(-1, -1, -1)],  # -x
    [vid(-1, 1, 1), vid(1, 1, 1), vid(1, 1, -1), vid(-1, 1, -1)],      # +y
    [vid(1, -1, 1), vid(-1, -1, 1), vid(-1, -1, -1), vid(1, -1, -1)],  # -y
    [vid(-1, 1, 1), vid(-1, -1, 1), vid(1, -1, 1), vid(1, 1, 1)],      # +z
    [vid(-1, -1, -1), vid(-1, 1, -1), vid(1, 1, -1), vid(1, -1, -1)],  # -z
], dtype=np.uint32)

# +x wall quadrants (image top-left, top-right, bottom-left, bottom-right)
QUADRANTS = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=np.float32)


def write_scene(path):
    os.makedirs(os.path.join(path, "textures"))
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(CORNERS)}\nproperty float x\nproperty float y\nproperty float z\n"
        f"element face {len(QUADS)}\nproperty list uchar int vertex_indices\nend_header\n"
    )
    faces = np.zeros(len(QUADS), dtype=[("n", "u1"), ("v", "<i4", (4,))])
    faces["n"], faces["v"] = 4, QUADS
    with open(os.path.join(path, "mesh.ply"), "wb") as f:
        f.write(header.encode())
        f.write(CORNERS.astype("<f4").tobytes())
        f.write(faces.tobytes())
    with open(os.path.join(path, "textures", "parameters.json"), "w") as f:
        json.dump({"splitSize": 0.0, "tileSize": TILE}, f)

    dim = 3 * TILE  # 3x3 tiles for 6 faces
    atlas = np.full((dim, dim, 3), 0.5, dtype=np.float16)
    h = TILE // 2
    for (r, c), colour in zip([(0, 0), (0, 1), (1, 0), (1, 1)], QUADRANTS):
        atlas[r * h:(r + 1) * h, c * h:(c + 1) * h] = colour  # tile 0 = face 0 (+x)
    for face in range(1, 6):
        ty, tx = divmod(face, 3)
        atlas[ty * TILE:(ty + 1) * TILE, tx * TILE:(tx + 1) * TILE] = 0.2 * face / 6
    atlas.tofile(os.path.join(path, "textures", "0-color-ptex.hdr"))


@pytest.fixture(scope="module")
def renderer(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("cube"))
    write_scene(path)
    r = Renderer(path, Camera(64, 64, 32.0, 32.0, 31.5, 31.5), cull="none", verbose=False)
    r.exposure, r.gamma, r.saturation = 1.0, 1.0, 1.0  # identity tone mapping
    return r


def look_along_x():
    """At the origin looking along +x, world +z up (camera y = down = -z)."""
    fwd, down = np.array([1.0, 0, 0]), np.array([0, 0, -1.0])
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2] = np.cross(down, fwd), down, fwd
    return T


def test_adjacency_of_closed_cube():
    adj = compute_adjacency(QUADS)
    assert np.all((adj & FACE_MASK) != FACE_MASK), "every cube edge is shared by two faces"


def test_load_scene(tmp_path):
    write_scene(str(tmp_path))
    s = load_scene(str(tmp_path))
    assert len(s.submeshes) == 1 and s.submeshes[0].quads.shape == (6, 4)


def test_orientation(renderer):
    rgb, _ = renderer.render(look_along_x())
    q = 64 // 4
    for (r, c), colour in zip([(q, q), (q, 3 * q), (3 * q, q), (3 * q, 3 * q)], QUADRANTS):
        np.testing.assert_allclose(rgb[r, c] / 255.0, colour, atol=0.02,
                                   err_msg=f"quadrant at pixel ({r},{c})")


def test_depth_is_along_optical_axis(renderer):
    _, depth = renderer.render(look_along_x())
    # The wall x = 1 is perpendicular to the optical axis: z = 1 everywhere.
    np.testing.assert_allclose(depth, 1.0, atol=1e-4)
