"""Voxel pose and box geometry.

Everything here works in scanner world coordinates: RAS+ millimetres, the
same space a NIfTI affine maps into. The image grid is never used to define
the voxel, so image orientation and resolution do not affect the geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Corner pairs forming the 12 edges of a box whose corners are enumerated by
# the sign pattern of (x, y, z) in binary order (see VoxelPose.corners).
_BOX_EDGES = [
    (0, 1), (2, 3), (4, 5), (6, 7),  # along axis 2
    (0, 2), (1, 3), (4, 6), (5, 7),  # along axis 1
    (0, 4), (1, 5), (2, 6), (3, 7),  # along axis 0
]


def rotation_matrix(axis, angle: float) -> np.ndarray:
    """Right-handed rotation of `angle` radians about unit `axis` (Rodrigues)."""
    k = np.asarray(axis, dtype=float)
    k = k / np.linalg.norm(k)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


@dataclass(frozen=True)
class VoxelPose:
    """An oriented box in world space.

    center: (3,) box centre, RAS mm.
    axes:   (3, 3) columns are unit vectors of the box axes in RAS. By
            convention the columns are (readout, phase, slice) so that
            `size` maps onto Siemens ReadoutFOV, PhaseFOV and Thickness.
    size:   (3,) full edge lengths in mm along each column of `axes`.
    """

    center: np.ndarray
    axes: np.ndarray
    size: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "center", np.asarray(self.center, dtype=float).reshape(3))
        object.__setattr__(self, "axes", np.asarray(self.axes, dtype=float).reshape(3, 3))
        object.__setattr__(self, "size", np.asarray(self.size, dtype=float).reshape(3))

    @classmethod
    def default(cls, center=(0.0, 0.0, 0.0), size=(20.0, 20.0, 20.0)) -> "VoxelPose":
        from .siemens import pose_from_siemens, SiemensVoxel  # transverse, no in-plane rotation

        sv = SiemensVoxel(position=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0), inplane_rot=0.0,
                          readout_fov=size[0], phase_fov=size[1], thickness=size[2])
        return cls(center=center, axes=pose_from_siemens(sv).axes, size=size)

    @property
    def half(self) -> np.ndarray:
        return self.size / 2

    @property
    def volume_mm3(self) -> float:
        return float(np.prod(self.size))

    def corners(self) -> np.ndarray:
        """(8, 3) corner positions in world space."""
        signs = np.array([[(i >> 2) & 1, (i >> 1) & 1, i & 1] for i in range(8)]) * 2 - 1
        return self.center + (signs * self.half) @ self.axes.T

    def local(self, points) -> np.ndarray:
        """World points (N, 3) -> coordinates along the box axes, relative to centre."""
        return (np.asarray(points, dtype=float) - self.center) @ self.axes

    def contains(self, points, tol: float = 1e-9) -> np.ndarray:
        return np.all(np.abs(self.local(points)) <= self.half + tol, axis=-1)

    def translated(self, delta) -> "VoxelPose":
        return VoxelPose(self.center + np.asarray(delta, dtype=float), self.axes, self.size)

    def rotated(self, axis, angle: float) -> "VoxelPose":
        """Rotate about a world-space axis passing through the box centre."""
        return VoxelPose(self.center, rotation_matrix(axis, angle) @ self.axes, self.size)

    def with_size(self, size) -> "VoxelPose":
        return VoxelPose(self.center, self.axes, size)

    def to_dict(self) -> dict:
        return {"center_ras_mm": self.center.tolist(),
                "axes_ras": self.axes.T.tolist(),  # list of (readout, phase, slice) vectors
                "size_mm": self.size.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "VoxelPose":
        return cls(center=d["center_ras_mm"], axes=np.array(d["axes_ras"]).T, size=d["size_mm"])


@dataclass(frozen=True)
class Plane:
    """A viewing plane. Screen right = h, screen up = v, normal n = h x v.

    Plane coordinates (u, w) of a world point p are (p.h, p.v); the plane
    itself is the set of points with p.n == offset.
    """

    h: np.ndarray
    v: np.ndarray
    offset: float = 0.0

    @property
    def n(self) -> np.ndarray:
        return np.cross(self.h, self.v)

    def at_offset(self, offset: float) -> "Plane":
        return Plane(self.h, self.v, float(offset))

    def through(self, point) -> "Plane":
        return self.at_offset(float(np.dot(point, self.n)))

    def to_world(self, u, w) -> np.ndarray:
        u = np.asarray(u, dtype=float)[..., None]
        w = np.asarray(w, dtype=float)[..., None]
        return u * self.h + w * self.v + self.offset * self.n

    def to_plane(self, points) -> np.ndarray:
        p = np.asarray(points, dtype=float)
        return np.stack([p @ self.h, p @ self.v], axis=-1)


def section_polygon(pose: VoxelPose, plane: Plane, eps: float = 1e-9) -> np.ndarray:
    """Cross-section of the box with a plane, as (K, 2) plane coordinates.

    Vertices are ordered counter-clockwise. Returns an empty (0, 2) array if
    the plane misses the box.
    """
    corners = pose.corners()
    s = corners @ plane.n - plane.offset
    pts = [corners[i] for i in range(8) if abs(s[i]) <= eps]
    for i, j in _BOX_EDGES:
        if (s[i] < -eps and s[j] > eps) or (s[i] > eps and s[j] < -eps):
            t = s[i] / (s[i] - s[j])
            pts.append(corners[i] + t * (corners[j] - corners[i]))
    if len(pts) < 3:
        return np.zeros((0, 2))
    uv = plane.to_plane(np.array(pts))
    # Drop near-duplicates (plane through a corner or along an edge).
    uv = uv[np.unique(np.round(uv / 1e-6), axis=0, return_index=True)[1]]
    centroid = uv.mean(axis=0)
    order = np.argsort(np.arctan2(uv[:, 1] - centroid[1], uv[:, 0] - centroid[0]))
    return uv[order]


def polygon_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def point_in_polygon(point, poly: np.ndarray) -> bool:
    """Even-odd ray casting test."""
    if len(poly) < 3:
        return False
    x, y = point
    inside = False
    for (x1, y1), (x2, y2) in zip(poly, np.roll(poly, -1, axis=0)):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def nearest_vertex(point, poly: np.ndarray, tol: float) -> int | None:
    """Index of the polygon vertex within `tol` of `point` (closest wins), else None."""
    if len(poly) == 0:
        return None
    d = np.hypot(*(np.asarray(poly) - point).T)
    i = int(np.argmin(d))
    return i if d[i] <= tol else None


def voxel_mask(pose: VoxelPose, shape, affine: np.ndarray, supersample: int = 3) -> np.ndarray:
    """Fraction of each image voxel covered by the box, on the image grid.

    Each image voxel is sampled at supersample**3 sub-points, so edge voxels
    get partial-volume fractions. Only the box's bounding region is evaluated.
    """
    shape = tuple(shape[:3])
    mask = np.zeros(shape, dtype=np.float32)
    inv = np.linalg.inv(affine)
    corner_idx = pose.corners() @ inv[:3, :3].T + inv[:3, 3]
    lo = np.clip(np.floor(corner_idx.min(axis=0)).astype(int) - 1, 0, np.array(shape) - 1)
    hi = np.clip(np.ceil(corner_idx.max(axis=0)).astype(int) + 1, 0, np.array(shape) - 1)
    if np.any(hi < lo):
        return mask

    grids = np.meshgrid(*[np.arange(lo[a], hi[a] + 1) for a in range(3)], indexing="ij")
    idx = np.stack([g.ravel() for g in grids], axis=-1).astype(float)
    offsets = (np.arange(supersample) + 0.5) / supersample - 0.5
    counts = np.zeros(len(idx), dtype=np.float32)
    for dx in offsets:
        for dy in offsets:
            for dz in offsets:
                sub = idx + (dx, dy, dz)
                world = sub @ affine[:3, :3].T + affine[:3, 3]
                counts += pose.contains(world)
    region = counts.reshape(grids[0].shape) / supersample**3
    mask[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1] = region
    return mask
