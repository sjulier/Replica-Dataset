"""Load a Replica scene for rendering: mesh, sub-mesh split, face adjacency, atlases.

This mirrors ReplicaSDK/ptex/PTexLib.cpp, with one deliberate difference: the
faces of each sub-mesh are taken from habitat/sorted_faces.bin instead of
being re-sorted. Tile i of atlas k holds the texels of face i of sub-mesh k,
and the order of faces with equal Morton codes depends on the std::sort that
generated the atlases (it is not stable, and gcc and clang differ), so it
cannot be reproduced; sorted_faces.bin records it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

ROTATION_SHIFT = 30
FACE_MASK = 0x3FFFFFFF

_PLY_TYPES = {
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "uchar": "u1",
    "uint8": "u1",
    "char": "i1",
    "int8": "i1",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "short": "<i2",
    "ushort": "<u2",
}


@dataclass
class SubMesh:
    positions: np.ndarray  # (V, 3) float32, vertices referenced by this sub-mesh
    quads: np.ndarray  # (F, 4) uint32, indices into positions; face i <-> atlas tile i
    adjacency: np.ndarray  # (F, 4) uint32, per edge: (rot << 30) | adjacent face
    atlas_path: str


@dataclass
class Scene:
    submeshes: list[SubMesh]
    tile_size: int
    is_hdr: bool


def read_ply(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read a binary little-endian quad PLY: returns (positions (N,3) f4, quads (M,4) u4)."""
    with open(path, "rb") as f:
        if f.readline().strip() not in (b"ply", b"PLY"):
            raise ValueError(f"{path}: not a PLY file")
        elements: list[tuple[str, int, list]] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: no end_header")
            tok = line.decode("ascii").split()
            if not tok or tok[0] in ("comment", "obj_info"):
                continue
            if tok[0] == "format":
                if tok[1] != "binary_little_endian":
                    raise ValueError(f"{path}: only binary_little_endian PLY is supported")
            elif tok[0] == "element":
                elements.append((tok[1], int(tok[2]), []))
            elif tok[0] == "property":
                elements[-1][2].append(tok[1:])
            elif tok[0] == "end_header":
                break
        offset = f.tell()

    data = np.memmap(path, dtype=np.uint8, mode="r", offset=offset)
    pos = 0
    positions = quads = None
    for name, count, props in elements:
        if name == "vertex":
            dtype = np.dtype([(p[1], _PLY_TYPES[p[0]]) for p in props])
            verts = np.frombuffer(data, dtype=dtype, count=count, offset=pos)
            pos += count * dtype.itemsize
            positions = np.stack([verts["x"], verts["y"], verts["z"]], axis=1).astype(np.float32)
        elif name == "face":
            if len(props) != 1 or props[0][0] != "list":
                raise ValueError(f"{path}: unexpected face properties {props}")
            count_t, index_t = _PLY_TYPES[props[0][1]], _PLY_TYPES[props[0][2]]
            # Replica meshes are all quads, so faces have a fixed size.
            dtype = np.dtype([("n", count_t), ("v", index_t, (4,))])
            faces = np.frombuffer(data, dtype=dtype, count=count, offset=pos)
            pos += count * dtype.itemsize
            if not np.all(faces["n"] == 4):
                raise ValueError(f"{path}: not a pure quad mesh")
            quads = faces["v"].astype(np.uint32)
        else:
            raise ValueError(f"{path}: unexpected element {name}")
    if positions is None or quads is None:
        raise ValueError(f"{path}: missing vertex or face element")
    return positions, quads


def read_sorted_faces(path: str) -> list[np.ndarray]:
    """habitat/sorted_faces.bin: u64 #submeshes, then per sub-mesh u64 #faces + u32 face ids."""
    raw = np.fromfile(path, dtype=np.uint8)
    n = int(raw[:8].view("<u8")[0])
    pos = 8
    out = []
    for _ in range(n):
        m = int(raw[pos : pos + 8].view("<u8")[0])
        pos += 8
        out.append(raw[pos : pos + 4 * m].view("<u4").copy())
        pos += 4 * m
    if pos != raw.size:
        raise ValueError(f"{path}: {raw.size - pos} trailing bytes")
    return out


def compute_adjacency(quads: np.ndarray) -> np.ndarray:
    """Per-edge adjacent face and rotation, as PTexMesh::CalculateAdjacency.

    Edge e of face f runs from vertex e to vertex (e+1)%4. Its entry packs the
    other face sharing that edge (FACE_MASK if none) and the number of 90-degree
    rotations between the two faces' tile frames.
    """
    nf = quads.shape[0]
    i0 = quads.astype(np.uint64)
    i1 = np.roll(quads, -1, axis=1).astype(np.uint64)
    keys = (np.minimum(i0, i1) << np.uint64(32)) | np.maximum(i0, i1)
    keys = keys.reshape(-1)  # entry f*4+e, i.e. the C++ insertion order
    face = np.repeat(np.arange(nf, dtype=np.int64), 4)
    edge = np.tile(np.arange(4, dtype=np.int64), nf)

    order = np.argsort(keys, kind="stable")  # stable keeps insertion order within an edge
    sk = keys[order]
    starts = np.flatnonzero(np.r_[True, sk[1:] != sk[:-1]])
    sizes = np.diff(np.r_[starts, sk.size])

    adj_face = np.full(nf * 4, -1, dtype=np.int64)
    rot = np.zeros(nf * 4, dtype=np.int64)

    # The common case: an edge shared by exactly two entries.
    two = starts[sizes == 2]
    a, b = order[two], order[two + 1]
    for s, o in ((a, b), (b, a)):
        adj_face[s] = np.where(face[o] != face[s], face[o], -1)
        rot[s] = (edge[s] - edge[o] + 2) & 3

    # Non-manifold edges (> 2 faces): the last other face in insertion order, rot 0.
    for st, sz in zip(starts[sizes > 2], sizes[sizes > 2]):
        members = order[st : st + sz]
        for s in members:
            others = [o for o in members if face[o] != face[s]]
            if others:
                adj_face[s] = face[others[-1]]

    packed = (rot.astype(np.uint32) << ROTATION_SHIFT) | (adj_face & FACE_MASK).astype(np.uint32)
    return packed.reshape(nf, 4)


def load_scene(scene_dir: str) -> Scene:
    """Load <scene_dir>/mesh.ply, textures/ and habitat/sorted_faces.bin."""
    tex_dir = os.path.join(scene_dir, "textures")
    with open(os.path.join(tex_dir, "parameters.json")) as f:
        params = json.load(f)
    tile_size = int(params["tileSize"])

    positions, quads = read_ply(os.path.join(scene_dir, "mesh.ply"))

    if float(params["splitSize"]) > 0:
        groups = read_sorted_faces(os.path.join(scene_dir, "habitat", "sorted_faces.bin"))
    else:
        groups = [np.arange(quads.shape[0], dtype=np.uint32)]
    if sum(g.size for g in groups) != quads.shape[0]:
        raise ValueError("sorted_faces.bin does not cover the mesh")

    submeshes = []
    is_hdr = False
    for i, faces in enumerate(groups):
        global_quads = quads[faces]
        # Re-index vertices locally (first-use order, as the C++ code does).
        uniq, first = np.unique(global_quads.reshape(-1), return_index=True)
        uniq = uniq[np.argsort(first)]
        remap = np.empty(positions.shape[0], dtype=np.uint32)
        remap[uniq] = np.arange(uniq.size, dtype=np.uint32)
        local_quads = remap[global_quads]

        hdr = os.path.join(tex_dir, f"{i}-color-ptex.hdr")
        if not os.path.exists(hdr):
            raise FileNotFoundError(f"{hdr} (only HDR atlases are supported in the spike)")
        is_hdr = True
        submeshes.append(
            SubMesh(
                positions=np.ascontiguousarray(positions[uniq]),
                quads=local_quads,
                adjacency=compute_adjacency(local_quads),
                atlas_path=hdr,
            )
        )
    return Scene(submeshes=submeshes, tile_size=tile_size, is_hdr=is_hdr)


def _read_hdr(path: str) -> tuple[np.ndarray, int]:
    raw = np.fromfile(path, dtype="<f2")
    dim = int(round(np.sqrt(raw.size / 3)))
    if dim * dim * 3 != raw.size:
        raise ValueError(f"{path}: not a square RGB16F image")
    return raw.reshape(dim, dim, 3), dim


def load_hdr_atlas(path: str) -> tuple[np.ndarray, int]:
    """A raw square half-float RGB atlas, expanded to RGBA16F (wgpu has no RGB16F): 8 bytes/texel."""
    rgb, dim = _read_hdr(path)
    rgba = np.empty((dim, dim, 4), dtype=np.float16)
    rgba[..., :3] = rgb
    rgba[..., 3] = 1.0
    return rgba, dim


def load_hdr_atlas_rgb9e5(path: str) -> tuple[np.ndarray, int]:
    """The atlas packed as RGB9E5 (shared 5-bit exponent, 9-bit mantissas): 4 bytes/texel.

    Follows the EXT_texture_shared_exponent / D3D encoding; inputs are clamped
    to [0, 65408]. Processed in row blocks to bound the temporary memory.
    """
    rgb_all, dim = _read_hdr(path)
    out = np.empty((dim, dim), dtype=np.uint32)
    N, B, MAX = 9, 15, 65408.0
    for r0 in range(0, dim, 512):
        c = np.clip(rgb_all[r0:r0 + 512].astype(np.float32), 0.0, MAX)
        maxc = c.max(axis=-1)
        with np.errstate(divide="ignore"):
            e = np.maximum(-B - 1, np.floor(np.log2(np.maximum(maxc, 1e-30)))).astype(np.int32) + 1 + B
        scale = np.exp2((e - B - N).astype(np.float32))
        maxm = np.floor(maxc / scale + 0.5)
        bump = maxm >= (1 << N)
        e = np.where(bump, e + 1, e)
        scale = np.where(bump, scale * 2, scale)
        m = np.minimum(np.floor(c / scale[..., None] + 0.5), (1 << N) - 1).astype(np.uint32)
        out[r0:r0 + 512] = m[..., 0] | (m[..., 1] << 9) | (m[..., 2] << 18) | (e.astype(np.uint32) << 27)
    return out, dim
