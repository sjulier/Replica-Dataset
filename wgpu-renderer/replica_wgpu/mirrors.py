"""Mirrors and glass from a Replica scene's glass.sur (a port of MirrorSurface).

Each surface is a plane, a polygon boundary in 2D plane ("manifold") coordinates,
a bounding rectangle in world coordinates and a reflectivity (1 for a mirror,
lower for glass). It is rendered as in ReplicaSDK's MirrorRenderer: the scene
reflected about the plane, drawn over the bounding rectangle with alpha =
Fresnel term x a polygon mask.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

MASK_SIZE = 256  # as MirrorRenderer
SURFACE_OFFSET = 0.0025  # metres; MirrorRenderer::surfaceOffset
# Surfaces at least this reflective count as mirrors for the depth modes; the rest
# are glass. Replica uses 1 for mirrors (room_0's glass is 0.3).
MIRROR_REFLECTIVITY = 0.9


@dataclass
class MirrorSurface:
    plane: np.ndarray  # (4,) n.p + d = 0, world coordinates
    rect: np.ndarray  # (4, 3) bounding_rect_w
    boundary_mani: np.ndarray  # (N, 2)
    T_mani_plane: np.ndarray  # (2, 4)
    T_plane_mani: np.ndarray  # (3, 3)
    reflectivity: float

    @property
    def is_mirror(self) -> bool:
        return self.reflectivity >= MIRROR_REFLECTIVITY

    def reflection_matrix(self) -> np.ndarray:
        """World-to-world reflection about the plane (MirrorRenderer::GetReflectionMatrix)."""
        n, d = self.plane[:3], self.plane[3]
        m = np.eye(4)
        m[:3, :3] -= 2 * np.outer(n, n)
        m[:3, 3] = -2 * n * d
        return m

    def clip_plane(self, camera_pos: np.ndarray) -> np.ndarray:
        """The plane that keeps the camera's side of the mirror (as CaptureReflection)."""
        sign = 1.0 if self.plane @ np.append(camera_pos, 1.0) > 0 else -1.0
        plane = self.plane.copy()
        plane[3] -= SURFACE_OFFSET
        return sign * plane

    def quad(self) -> np.ndarray:
        """The rectangle drawn on screen, (4, 3) in uv order (0,0), (1,0), (1,1), (0,1),
        offset along the normal as MirrorRenderer::Draw."""
        n = self.plane[:3] / np.linalg.norm(self.plane[:3])
        return self.rect[[0, 1, 3, 2]] + SURFACE_OFFSET * n

    def mask(self, w: int = MASK_SIZE, h: int = MASK_SIZE) -> np.ndarray:
        """(h, w) uint8 polygon mask over the bounding rectangle (MirrorSurface::GenerateMask)."""
        r = (self.T_mani_plane @ np.c_[self.rect, np.ones(4)].T).T  # (4, 2)
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        u = (1.0 - (x + 0.5) / w)[..., None]
        v = (1.0 - (y + 0.5) / h)[..., None]
        p0 = u * r[0] + (1 - u) * r[1]
        p1 = u * r[2] + (1 - u) * r[3]
        p_mani = v * p0 + (1 - v) * p1  # (h, w, 2)
        # to the world and back, as the original
        p_w = np.einsum("ij,hwj->hwi", self.T_plane_mani, np.concatenate([p_mani, np.ones((h, w, 1))], -1))
        q = np.einsum("ij,hwj->hwi", self.T_mani_plane, np.concatenate([p_w, np.ones((h, w, 1))], -1))
        return (255 * _in_polygon(q[..., 0], q[..., 1], self.boundary_mani)).astype(np.uint8)


def _in_polygon(px: np.ndarray, py: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Even-odd crossing test, as MirrorSurface::InBoundary."""
    inside = np.zeros(px.shape, dtype=bool)
    n = len(poly)
    for v in range(n):
        u = (v - 1) % n
        xv, yv = poly[v]
        xu, yu = poly[u]
        if yu == yv:
            continue  # never straddles: (yv > py) == (yu > py)
        straddles = (yv > py) != (yu > py)
        inside ^= straddles & (px < (xu - xv) * (py - yv) / (yu - yv) + xv)
    return inside


def load_mirrors(path: str) -> list[MirrorSurface]:
    with open(path) as f:
        surfaces = json.load(f)
    return [
        MirrorSurface(
            plane=np.asarray(s["plane_w"], dtype=np.float64),
            rect=np.asarray(s["bounding_rect_w"], dtype=np.float64),
            boundary_mani=np.asarray(s["boundary_mani"], dtype=np.float64).reshape(-1, 2),
            T_mani_plane=np.asarray(s["T_mani_plane"], dtype=np.float64),
            T_plane_mani=np.asarray(s["T_plane_mani"], dtype=np.float64),
            reflectivity=float(s["reflectivity"]),
        )
        for s in surfaces
    ]
