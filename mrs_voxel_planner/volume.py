"""T1 volume loading and reslicing onto arbitrary world-space planes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates

from .geometry import Plane


@dataclass
class PlaneImage:
    image: np.ndarray  # (rows, cols); row 0 is the lowest w, col 0 the lowest u
    u0: float  # plane coordinates of the image's lower-left edge
    w0: float
    spacing: float

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """(u0, w0, width, height) in mm, for placing the image in a view."""
        rows, cols = self.image.shape
        return self.u0, self.w0, cols * self.spacing, rows * self.spacing


class Volume:
    def __init__(self, data: np.ndarray, affine: np.ndarray, name: str = ""):
        data = np.asarray(data)
        if data.ndim == 4:
            data = data[..., 0]
        self.data = np.ascontiguousarray(data, dtype=np.float32)
        self.affine = np.asarray(affine, dtype=float)
        self.inv_affine = np.linalg.inv(self.affine)
        self.name = name
        self.voxel_sizes = np.linalg.norm(self.affine[:3, :3], axis=0)
        nz = self.data[self.data > 0]
        self.display_range = (float(np.percentile(nz, 1)), float(np.percentile(nz, 99.5))) \
            if nz.size else (0.0, 1.0)

    @classmethod
    def load(cls, path) -> "Volume":
        import nibabel as nib

        img = nib.load(str(path))
        return cls(img.get_fdata(dtype=np.float32), img.affine, name=str(path))

    def world_corners(self) -> np.ndarray:
        """(8, 3) world positions of the outer corners of the image grid."""
        n = np.array(self.data.shape)
        idx = np.array([[(i >> 2) & 1, (i >> 1) & 1, i & 1] for i in range(8)]) * n - 0.5
        return idx @ self.affine[:3, :3].T + self.affine[:3, 3]

    def world_center(self) -> np.ndarray:
        return self.world_corners().mean(axis=0)

    def sample_plane(self, plane: Plane, spacing: float | None = None) -> PlaneImage:
        """Trilinear reslice of the volume onto `plane`, covering the whole volume."""
        spacing = float(spacing or self.voxel_sizes.min())
        uv = plane.to_plane(self.world_corners())
        (u0, w0), (u1, w1) = uv.min(axis=0), uv.max(axis=0)
        cols = max(1, int(np.ceil((u1 - u0) / spacing)))
        rows = max(1, int(np.ceil((w1 - w0) / spacing)))
        u = u0 + (np.arange(cols) + 0.5) * spacing
        w = w0 + (np.arange(rows) + 0.5) * spacing
        world = plane.to_world(*np.meshgrid(u, w))  # (rows, cols, 3)
        idx = world @ self.inv_affine[:3, :3].T + self.inv_affine[:3, 3]
        img = map_coordinates(self.data, np.moveaxis(idx, -1, 0), order=1, mode="constant",
                              cval=0.0, prefilter=False)
        return PlaneImage(img, float(u0), float(w0), spacing)


def window_levels(auto_range, brightness: float, contrast: float) -> tuple[float, float]:
    """Display (black, white) levels from brightness/contrast settings in [-100, 100].

    0/0 gives `auto_range`. Brightness +100 shifts the window down by one full
    auto width (brighter image); contrast +/-100 narrows/widens it 16-fold.
    """
    lo, hi = auto_range
    width = (hi - lo) * 2.0 ** (-contrast / 25)
    centre = (lo + hi) / 2 - brightness / 100 * (hi - lo)
    return centre - width / 2, centre + width / 2


def make_phantom(shape=(160, 192, 160), voxel_mm: float = 1.0) -> Volume:
    """Crude T1-like head (scalp, skull, CSF, GM, WM, ventricles) for testing."""
    affine = np.diag([voxel_mm, voxel_mm, voxel_mm, 1.0])
    affine[:3, 3] = -(np.array(shape) - 1) / 2 * voxel_mm
    i, j, k = np.indices(shape, dtype=np.float32)
    x, y, z = (affine[a, a] * g + affine[a, 3] for a, g in enumerate((i, j, k)))

    def ell(rx, ry, rz, cx=0.0, cy=0.0, cz=0.0):
        return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 + ((z - cz) / rz) ** 2 <= 1

    data = np.zeros(shape, dtype=np.float32)
    data[ell(75, 92, 75)] = 400  # scalp
    data[ell(70, 87, 70)] = 60  # skull
    data[ell(66, 83, 66)] = 150  # CSF
    data[ell(63, 80, 63)] = 550  # GM
    data[ell(52, 68, 52)] = 800  # WM
    data[ell(6, 22, 12, cx=-9, cy=-5, cz=8) | ell(6, 22, 12, cx=9, cy=-5, cz=8)] = 150  # ventricles
    return Volume(data, affine, name="phantom")
