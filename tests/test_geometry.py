import numpy as np
import pytest

from mrs_voxel_planner.geometry import (Plane, VoxelPose, nearest_vertex, point_in_polygon,
                                        polygon_area, rotation_matrix, section_polygon, voxel_mask)

AXIAL = Plane(h=np.array([-1.0, 0, 0]), v=np.array([0, 1.0, 0]))


def random_pose(rng, center=(0, 0, 0)):
    axes = rotation_matrix(rng.normal(size=3), rng.uniform(0, np.pi)) @ np.eye(3)
    return VoxelPose(center, axes, rng.uniform(10, 30, size=3))


def test_rotation_matrix_is_proper_rotation():
    R = rotation_matrix([1, 2, 3], 0.7)
    assert np.allclose(R @ R.T, np.eye(3))
    assert np.isclose(np.linalg.det(R), 1)
    assert np.allclose(rotation_matrix([0, 0, 1], np.pi / 2) @ [1, 0, 0], [0, 1, 0])


def test_contains():
    pose = VoxelPose((10, 0, 0), np.eye(3), (20, 10, 4))
    assert pose.contains([[10, 0, 0], [19.9, 4.9, 1.9]]).all()
    assert not pose.contains([[20.1, 0, 0]]).any()


def test_axial_section_of_aligned_box_is_rectangle():
    pose = VoxelPose((5, -3, 12), np.eye(3), (20, 30, 10))
    poly = section_polygon(pose, AXIAL.through(pose.center))
    assert len(poly) == 4
    assert np.isclose(abs(polygon_area(poly)), 600)
    assert polygon_area(poly) > 0  # counter-clockwise


def test_section_rotated_in_plane_keeps_area():
    pose = VoxelPose((0, 0, 0), np.eye(3), (20, 30, 10)).rotated([0, 0, 1], np.radians(37))
    poly = section_polygon(pose, AXIAL)
    assert np.isclose(abs(polygon_area(poly)), 600)


def test_section_can_be_hexagon_and_misses_outside():
    pose = VoxelPose((0, 0, 0), np.eye(3), (20, 20, 20))
    diag = Plane(h=np.array([1, -1, 0]) / np.sqrt(2), v=np.array([1, 1, -2]) / np.sqrt(6))
    assert len(section_polygon(pose, diag)) == 6
    assert len(section_polygon(pose, AXIAL.at_offset(50))) == 0


def test_point_in_polygon():
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    assert point_in_polygon((0.5, 0.5), sq)
    assert not point_in_polygon((1.5, 0.5), sq)


@pytest.mark.parametrize("seed", range(3))
def test_mask_volume_matches_box_on_oblique_grid(seed):
    rng = np.random.default_rng(seed)
    pose = random_pose(rng, center=(3.3, -2.1, 5.7))
    affine = np.eye(4)
    affine[:3, :3] = rotation_matrix([1, 1, 0], 0.3) @ np.diag([1.0, 1.2, 0.9])
    affine[:3, 3] = [-40, -40, -40]
    shape = (80, 70, 90)
    # centre the box inside the grid
    pose = VoxelPose(affine[:3, :3] @ (np.array(shape) / 2) + affine[:3, 3], pose.axes, pose.size)
    mask = voxel_mask(pose, shape, affine, supersample=4)
    vol = mask.sum() * abs(np.linalg.det(affine[:3, :3]))
    assert vol == pytest.approx(pose.volume_mm3, rel=0.02)
    assert mask.max() == 1 and mask.min() == 0


def test_pose_dict_round_trip():
    pose = random_pose(np.random.default_rng(1), center=(1, 2, 3))
    back = VoxelPose.from_dict(pose.to_dict())
    assert np.allclose(back.axes, pose.axes) and np.allclose(back.center, pose.center)


def test_nearest_vertex():
    sq = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    assert nearest_vertex((9.5, 10.4), sq, tol=1) == 2
    assert nearest_vertex((5, 5), sq, tol=1) is None
    assert nearest_vertex((0, 0), np.zeros((0, 2)), tol=1) is None
