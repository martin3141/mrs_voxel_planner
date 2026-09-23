"""Siemens SVS voxel geometry conventions.

Siemens describes a voxel by:
  - position (sPosition: dSag, dCor, dTra), mm in DICOM patient coordinates (LPS+)
  - slice normal (sNormal: dSag, dCor, dTra), unit vector in LPS
  - in-plane rotation (dInPlaneRot), radians
  - dReadoutFOV, dPhaseFOV, dThickness, mm

Phase and readout directions are not stored; the scanner derives them from
the normal and in-plane rotation (the GSL "CalcPRS" routine). That routine is
reproduced here, following the port used by spec2nii.

Internally the planner uses RAS+; LPS <-> RAS flips the first two axes.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass

import numpy as np

from .geometry import VoxelPose

LPS_RAS = np.diag([-1.0, -1.0, 1.0])  # its own inverse

SAGITTAL, CORONAL, TRANSVERSE = "Sagittal", "Coronal", "Transverse"


@dataclass(frozen=True)
class SiemensVoxel:
    position: tuple  # (sag, cor, tra) mm, LPS
    normal: tuple  # (sag, cor, tra), LPS
    inplane_rot: float  # radians
    readout_fov: float
    phase_fov: float
    thickness: float

    def to_dict(self) -> dict:
        return {"sPosition": dict(zip(("dSag", "dCor", "dTra"), map(float, self.position))),
                "sNormal": dict(zip(("dSag", "dCor", "dTra"), map(float, self.normal))),
                "dInPlaneRot": float(self.inplane_rot),
                "dReadoutFOV": float(self.readout_fov),
                "dPhaseFOV": float(self.phase_fov),
                "dThickness": float(self.thickness)}


def main_orientation(normal, tol: float = 1e-6) -> str:
    """Dominant component of the normal. Ties go Transverse > Coronal > Sagittal."""
    s, c, t = np.abs(np.asarray(normal, dtype=float))
    if t >= c - tol and t >= s - tol:
        return TRANSVERSE
    if c >= s - tol:
        return CORONAL
    return SAGITTAL


def phase_readout(normal, inplane_rot: float) -> tuple[np.ndarray, np.ndarray]:
    """Siemens CalcPRS: phase (gp) and readout (gr) unit vectors in LPS."""
    gs = np.asarray(normal, dtype=float)
    gs = gs / np.linalg.norm(gs)
    ori = main_orientation(gs)
    if ori == TRANSVERSE:
        k = 1 / np.hypot(gs[1], gs[2])
        gp = np.array([0.0, gs[2] * k, -gs[1] * k])
    elif ori == CORONAL:
        k = 1 / np.hypot(gs[0], gs[1])
        gp = np.array([gs[1] * k, -gs[0] * k, 0.0])
    else:
        k = 1 / np.hypot(gs[0], gs[1])
        gp = np.array([-gs[1] * k, gs[0] * k, 0.0])
    gr = np.cross(gs, gp)
    if inplane_rot != 0.0:
        gp = np.cos(inplane_rot) * gp - np.sin(inplane_rot) * gr
        gr = np.cross(gs, gp)
    return gp, gr


def pose_from_siemens(sv: SiemensVoxel) -> VoxelPose:
    gs = np.asarray(sv.normal, dtype=float)
    gs = gs / np.linalg.norm(gs)
    gp, gr = phase_readout(gs, sv.inplane_rot)
    axes_lps = np.column_stack([gr, gp, gs])
    return VoxelPose(center=LPS_RAS @ np.asarray(sv.position, dtype=float),
                     axes=LPS_RAS @ axes_lps,
                     size=(sv.readout_fov, sv.phase_fov, sv.thickness))


def siemens_from_pose(pose: VoxelPose) -> SiemensVoxel:
    """Inverse of pose_from_siemens.

    Uses the pose's slice axis as the normal and its phase axis to recover the
    in-plane rotation. The readout axis is implied (gs x gp), so a pose whose
    axes came from elsewhere keeps its box but may flip readout sign.
    """
    axes_lps = LPS_RAS @ pose.axes
    gs, gp = axes_lps[:, 2], axes_lps[:, 1]
    gp0, gr0 = phase_readout(gs, 0.0)
    phi = float(np.arctan2(-np.dot(gp, gr0), np.dot(gp, gp0)))
    return SiemensVoxel(position=tuple(LPS_RAS @ pose.center), normal=tuple(gs), inplane_rot=phi,
                        readout_fov=pose.size[0], phase_fov=pose.size[1], thickness=pose.size[2])


_AXIS_INDEX = {"S": 0, "C": 1, "T": 2}
_AXIS_LETTER = "SCT"


def orientation_string(normal) -> str:
    """Scanner-style orientation of a slice normal, e.g. 'T > C -23.9 > S 4.5'.

    The normal starts on the main axis, tilts towards axis A by alpha, then
    towards axis B by beta, where a tilt of theta towards X moves the normal
    by -sin(theta) along X (LPS):
        n = cos(beta) * (cos(alpha) e_main - sin(alpha) e_A) - sin(beta) e_B
    The larger tilt is listed first. This matches scanner-reported
    (string, sNormal) pairs; the sign of the normal is ignored.
    """
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    m = _AXIS_INDEX[main_orientation(n)[0]]
    if n[m] < 0:
        n = -n
    # Larger component first; ties follow the Transverse > Coronal > Sagittal priority.
    a, b = sorted((k for k in range(3) if k != m), key=lambda k: (-round(abs(n[k]), 12), -k))
    alpha = round(float(np.degrees(np.arctan2(-n[a], n[m]))), 1) + 0.0
    beta = round(float(np.degrees(np.arcsin(np.clip(-n[b], -1, 1)))), 1) + 0.0
    s = _AXIS_LETTER[m]
    if alpha or beta:
        s += f" > {_AXIS_LETTER[a]} {alpha:.1f}"
    if beta:
        s += f" > {_AXIS_LETTER[b]} {beta:.1f}"
    return s


def normal_from_orientation_string(text: str) -> np.ndarray:
    """Inverse of orientation_string (LPS unit normal).

    Accepts 'T > C -23.9 > S 4.5', 'Tra>Cor(-23.9)>Sag(4.5)', or just 'Sag'.
    """
    parts = [p.strip() for p in text.split(">")]
    if not 1 <= len(parts) <= 3:
        raise ValueError(f"expected 1 to 3 '>'-separated terms: {text!r}")
    axes, angles = [], []
    for i, part in enumerate(parts):
        m = re.fullmatch(r"(?i)(sag|cor|tra|s|c|t)[a-z]*\s*\(?\s*([-+]?\d*\.?\d+)?\s*\)?", part)
        if not m or (i == 0) != (m.group(2) is None):
            raise ValueError(f"cannot parse {part!r} in {text!r}")
        axes.append(_AXIS_INDEX[m.group(1)[0].upper()])
        angles.append(float(m.group(2) or 0.0))
    if len(set(axes)) != len(axes):
        raise ValueError(f"repeated axis in {text!r}")
    n = np.eye(3)[axes[0]]
    for ax, ang in zip(axes[1:], np.radians(angles[1:])):
        n = np.cos(ang) * n - np.sin(ang) * np.eye(3)[ax]
    return n


def format_position(position_lps) -> str:
    """Scanner-style position, e.g. 'L3.0 P12.5 H20.0'."""
    s, c, t = position_lps
    return (f"{'L' if s >= 0 else 'R'}{abs(s):.1f} "
            f"{'P' if c >= 0 else 'A'}{abs(c):.1f} "
            f"{'H' if t >= 0 else 'F'}{abs(t):.1f}")


def read_rda_header(path) -> dict:
    """Parse the text header of a Siemens .rda file into {key: str}."""
    header = {}
    with open(path, "rb") as f:
        for raw in f:
            line = raw.decode("latin-1").strip()
            if line.startswith(">>> End of header"):
                break
            m = re.match(r"^([^:]+):\s*(.*)$", line)
            if m:
                header[m.group(1).strip()] = m.group(2).strip()
    return header


def siemens_from_rda(path) -> SiemensVoxel:
    """Read the VOI geometry from a Siemens .rda export."""
    h = read_rda_header(path)
    f = lambda k: float(h[k])  # noqa: E731
    sv = SiemensVoxel(position=(f("VOIPositionSag"), f("VOIPositionCor"), f("VOIPositionTra")),
                      normal=(f("VOINormalSag"), f("VOINormalCor"), f("VOINormalTra")),
                      inplane_rot=f("VOIRotationInPlane"),
                      readout_fov=f("VOIReadoutFOV"), phase_fov=f("VOIPhaseFOV"),
                      thickness=f("VOIThickness"))
    # The RDA also stores the row/column direction cosines. If they don't match
    # the derived readout/phase axes, the CalcPRS port or units are wrong.
    try:
        row = np.array([f(f"RowVector[{i}]") for i in range(3)])
        col = np.array([f(f"ColumnVector[{i}]") for i in range(3)])
    except KeyError:
        return sv
    gp, gr = phase_readout(sv.normal, sv.inplane_rot)
    if not (abs(abs(row @ gr) - 1) < 1e-3 and abs(abs(col @ gp) - 1) < 1e-3):
        warnings.warn(f"{path}: RowVector/ColumnVector do not match readout/phase derived from "
                      "VOINormal and VOIRotationInPlane; check the geometry.")
    return sv


def pose_from_nifti_mrs(path) -> VoxelPose:
    """Voxel from a single-voxel NIfTI-MRS file (e.g. converted by spec2nii).

    The affine of a 1x1x1 NIfTI-MRS voxel encodes centre, orientation and size
    directly, so this works for any vendor. Axis order is whatever the
    converter wrote, which may not be (readout, phase, slice).
    """
    import nibabel as nib

    A = nib.load(str(path)).affine
    M = A[:3, :3]
    size = np.linalg.norm(M, axis=0)
    return VoxelPose(center=A[:3, 3], axes=M / size, size=size)
