import numpy as np

from mrs_voxel_planner.geometry import Plane, rotation_matrix
from mrs_voxel_planner.volume import Volume


def test_sample_plane_follows_affine():
    # Intensity = world x coordinate, on an oblique anisotropic grid.
    affine = np.eye(4)
    affine[:3, :3] = rotation_matrix([0, 1, 1], 0.4) @ np.diag([1.0, 1.5, 2.0])
    affine[:3, 3] = [-30, -30, -30]
    idx = np.indices((50, 40, 30)).reshape(3, -1).T
    world_x = (idx @ affine[:3, :3].T + affine[:3, 3])[:, 0]
    vol = Volume(world_x.reshape(50, 40, 30), affine)

    plane = Plane(h=np.array([0, 0, 1.0]), v=np.array([1.0, 0, 0]), offset=-5.0)  # screen up = +x
    pim = vol.sample_plane(plane, spacing=1.0)
    rows, cols = pim.image.shape
    w = pim.w0 + (np.arange(rows) + 0.5) * pim.spacing
    inside = pim.image != 0
    expected = np.broadcast_to(w[:, None], pim.image.shape)
    # Away from the grid edge trilinear interpolation of a linear field is exact.
    interior = inside & np.roll(inside, 2, 0) & np.roll(inside, -2, 0) \
        & np.roll(inside, 2, 1) & np.roll(inside, -2, 1)
    assert interior.sum() > 200
    assert np.allclose(pim.image[interior], expected[interior], atol=1e-3)


def test_world_bounds_span_voxel_centres():
    affine = np.eye(4)
    affine[:3, :3] = rotation_matrix([1, 2, 3], 0.5) @ np.diag([1.0, 1.5, 2.0])
    affine[:3, 3] = [-20, 10, 5]
    vol = Volume(np.zeros((12, 9, 7)), affine)
    idx = np.indices(vol.data.shape).reshape(3, -1).T
    centres = idx @ affine[:3, :3].T + affine[:3, 3]
    lo, hi = vol.world_bounds()
    assert np.allclose(lo, centres.min(axis=0))
    assert np.allclose(hi, centres.max(axis=0))


def test_window_levels():
    from mrs_voxel_planner.volume import window_levels

    assert window_levels((100, 300), 0, 0) == (100, 300)
    lo, hi = window_levels((100, 300), 50, 0)  # brighter: window shifts down by half a width
    assert (lo, hi) == (0, 200)
    lo, hi = window_levels((100, 300), 0, 25)  # more contrast: half the width, same centre
    assert (lo, hi) == (150, 250)
    lo, hi = window_levels((100, 300), 0, -100)
    assert hi - lo == 200 * 16 and (lo + hi) / 2 == 200
