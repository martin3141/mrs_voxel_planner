import warnings

import nibabel as nib
import numpy as np
import pytest

from mrs_voxel_planner.siemens import (CORONAL, SAGITTAL, TRANSVERSE, SiemensVoxel, main_orientation,
                                       normal_from_orientation_string, orientation_string,
                                       phase_readout, pose_from_nifti_mrs, pose_from_siemens,
                                       siemens_from_pose, siemens_from_rda)


def test_main_orientation_and_ties():
    assert main_orientation((0, 0, 1)) == TRANSVERSE
    assert main_orientation((0.1, -0.9, 0.2)) == CORONAL
    assert main_orientation((-0.8, 0.5, 0.3)) == SAGITTAL
    r = 1 / np.sqrt(2)
    assert main_orientation((r, 0, r)) == TRANSVERSE
    assert main_orientation((0, r, r)) == TRANSVERSE
    assert main_orientation((r, r, 0)) == CORONAL


def test_unrotated_transverse_axes():
    pose = pose_from_siemens(SiemensVoxel((10, 20, 30), (0, 0, 1), 0.0, 20, 25, 15))
    # LPS readout (-1,0,0) = RAS (+1,0,0); LPS phase (0,1,0) = RAS (0,-1,0)
    assert np.allclose(pose.axes, [[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    assert np.allclose(pose.center, (-10, -20, 30))
    assert np.allclose(pose.size, (20, 25, 15))


@pytest.mark.parametrize("seed", range(50))
def test_round_trip(seed):
    rng = np.random.default_rng(seed)
    n = rng.normal(size=3)
    n /= np.linalg.norm(n)
    sv = SiemensVoxel(tuple(rng.uniform(-50, 50, 3)), tuple(n), rng.uniform(-np.pi, np.pi),
                      *rng.uniform(10, 30, 3))
    back = siemens_from_pose(pose_from_siemens(sv))
    assert np.allclose(back.position, sv.position)
    assert np.allclose(back.normal, sv.normal)
    assert np.isclose(np.angle(np.exp(1j * (back.inplane_rot - sv.inplane_rot))), 0, atol=1e-9)


def test_rotating_pose_keeps_siemens_frame_consistent():
    # After an arbitrary rotation the readout axis must still equal gs x gp as Siemens defines it.
    pose = pose_from_siemens(SiemensVoxel((0, 0, 0), (0, 0, 1), 0.3, 20, 20, 20))
    pose = pose.rotated([0.3, 1, 0.2], 0.9)
    sv = siemens_from_pose(pose)
    assert np.allclose(pose_from_siemens(sv).axes, pose.axes)


def write_rda(path, sv, row=None, col=None):
    lines = [">>> Begin of header <<<", "SeriesDescription: svs_se_30"]
    for k, v in zip(("Sag", "Cor", "Tra"), sv.position):
        lines.append(f"VOIPosition{k}: {v}")
    for k, v in zip(("Sag", "Cor", "Tra"), sv.normal):
        lines.append(f"VOINormal{k}: {v}")
    lines += [f"VOIRotationInPlane: {sv.inplane_rot}", f"VOIThickness: {sv.thickness}",
              f"VOIPhaseFOV: {sv.phase_fov}", f"VOIReadoutFOV: {sv.readout_fov}"]
    if row is not None:
        lines += [f"RowVector[{i}]: {v}" for i, v in enumerate(row)]
        lines += [f"ColumnVector[{i}]: {v}" for i, v in enumerate(col)]
    lines.append(">>> End of header <<<")
    path.write_bytes(("\r\n".join(lines) + "\r\n").encode("latin-1") + b"\x00" * 16)


def test_read_rda_without_voi_geometry(tmp_path):
    (tmp_path / "c.rda").write_text(">>> Begin of header <<<\nVOIPositionSag: 1.0\n"
                                    ">>> End of header <<<\n")
    with pytest.raises(ValueError, match="missing VOIPositionCor"):
        siemens_from_rda(tmp_path / "c.rda")


def test_read_rda(tmp_path):
    sv = SiemensVoxel((1.5, -20.0, 12.0), (0.0, 0.2588190451, 0.9659258263), 0.2, 20, 25, 15)
    gp, gr = phase_readout(sv.normal, sv.inplane_rot)
    write_rda(tmp_path / "a.rda", sv, row=gr, col=gp)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = siemens_from_rda(tmp_path / "a.rda")
    assert np.allclose(got.position, sv.position) and np.isclose(got.inplane_rot, 0.2)

    write_rda(tmp_path / "b.rda", sv, row=gp, col=-gr)  # inconsistent vectors
    with pytest.warns(UserWarning):
        siemens_from_rda(tmp_path / "b.rda")


def test_pose_from_nifti_mrs(tmp_path):
    pose = pose_from_siemens(SiemensVoxel((5, 6, 7), (0.2, 0.1, 0.97), 0.4, 20, 25, 15))
    A = np.eye(4)
    A[:3, :3] = pose.axes * pose.size
    A[:3, 3] = pose.center
    nib.save(nib.Nifti2Image(np.zeros((1, 1, 1, 64), np.complex64), A), tmp_path / "svs.nii.gz")
    got = pose_from_nifti_mrs(tmp_path / "svs.nii.gz")
    assert np.allclose(got.center, pose.center) and np.allclose(got.size, pose.size)
    assert np.allclose(got.axes, pose.axes, atol=1e-6)


# (scanner orientation text, sNormal) pairs reported by a Siemens scanner
# (http://mvpa.blogspot.com/2016/09/multiband-acquisition-sequence-testing.html)
SCANNER_PAIRS = [
    ("T > C -23.9 > S 4.5", (-0.07818081918, 0.4036100151, 0.9115847274)),
    ("T > S 6.3 > C -3.8", (-0.1087323357, 0.06708937544, 0.9918045649)),
]


@pytest.mark.parametrize("text, normal", SCANNER_PAIRS)
def test_orientation_string_matches_scanner(text, normal):
    assert orientation_string(normal) == text
    assert np.allclose(normal_from_orientation_string(text), normal, atol=2e-3)


@pytest.mark.parametrize("text, expected", [
    ("Tra>Cor(-23.9)>Sag(4.5)", "T > C -23.9 > S 4.5"),
    ("sag", "S"),
    ("C > T 12", "C > T 12.0"),
    ("S>T(-30.0)>C(10)", "S > T -30.0 > C 10.0"),
])
def test_orientation_string_parsing(text, expected):
    assert orientation_string(normal_from_orientation_string(text)) == expected


@pytest.mark.parametrize("bad", ["", "T > C", "X > C 3", "T > T 5 > S 1", "T 5", "T > C 1 > S 2 > C 3"])
def test_orientation_string_rejects(bad):
    with pytest.raises(ValueError):
        normal_from_orientation_string(bad)


@pytest.mark.parametrize("seed", range(200))
def test_orientation_string_round_trip(seed):
    rng = np.random.default_rng(seed)
    n = rng.normal(size=3)
    n /= np.linalg.norm(n)
    back = normal_from_orientation_string(orientation_string(n))
    assert abs(abs(back @ n) - 1) < 1e-4  # same axis up to 0.1 deg rounding and sign


def test_orientation_string_ignores_normal_sign_and_simple_cases():
    assert orientation_string((0, 0, 1)) == orientation_string((0, 0, -1)) == "T"
    assert orientation_string((1, 0, 0)) == "S"
    s, c = np.sin(np.radians(10)), np.cos(np.radians(10))
    assert orientation_string((0, -s, c)) == "T > C 10.0"
