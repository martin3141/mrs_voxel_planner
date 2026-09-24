# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A PyQt6 + pyqtgraph desktop tool that emulates placing a single-voxel MRS VOI on a 3D T1-weighted NIfTI image, using Siemens scanner geometry conventions. See README.md for the user-facing controls and file formats.

## Commands

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'   # setup
.venv/bin/pytest                                              # all tests (fast, no GUI)
.venv/bin/pytest tests/test_siemens.py::test_main_orientation_and_ties   # single test
.venv/bin/mrs_voxel_planner --phantom                         # run with synthetic head
.venv/bin/mrs_voxel_planner T1.nii.gz --voxel scan.rda        # T1 + scanner voxel (.json/.rda/NIfTI-MRS)
```

No linter or formatter is configured. Code uses a ~100-column line width.

## Architecture

Dependency direction: `viewer` → `siemens` → `geometry` ← `volume`. Only `viewer.py` and `app.py` import Qt; everything else is pure numpy/scipy/nibabel and is what the tests cover.

- **One world space: RAS+ mm.** `VoxelPose` (geometry.py) is an oriented box (`center`, `axes` as 3×3 columns, `size`) defined purely in world coordinates, never on the image grid. The image is sampled onto planes through its affine (`Volume.sample_plane`, trilinear via `map_coordinates`), and the voxel mask is rasterised back onto the T1 grid (`voxel_mask`, supersampled for partial volume). Changing image resolution or orientation must never change the voxel geometry.
- **Axis-column convention.** `VoxelPose.axes` columns are (readout, phase, slice), so `size` maps onto Siemens (ReadoutFOV, PhaseFOV, Thickness). Keep that ordering whenever you construct or modify a pose.
- **Siemens boundary is siemens.py.** Siemens values live in LPS (`sPosition`, `sNormal`, `dInPlaneRot`); `LPS_RAS` (diag(-1,-1,1), its own inverse) converts. `phase_readout` is a port of the scanner's CalcPRS routine (following spec2nii) and derives phase/readout from the normal + in-plane rotation; `main_orientation` tie-breaking (Transverse > Coronal > Sagittal) affects it. `siemens_from_pose` inverts `pose_from_siemens` by recovering in-plane rotation from the phase axis. Orientation strings like `T > C -23.9 > S 4.5` are parsed/produced here too.
- **Views (viewer.py).** `VIEWS` defines three fixed anatomical `Plane`s (h = screen right, v = screen up, n = h×v) in radiological convention. On every `refresh` each `SliceView` moves its plane through `MainWindow.slice_point`, which tracks the voxel centre while `follow_voxel` is on (the default) and is set independently otherwise (Slices panel, `request_step` via ctrl+wheel). It then draws `section_polygon(pose, plane)` (up to 6 vertices when the voxel is oblique; every vertex is a rotate handle), and caches the resliced image by plane offset. Mouse drags in `VoxelViewBox` become `request_move` (translate in-plane) or `request_rotate` (rotate about the view's normal `plane.n`).
- **State flow.** `MainWindow.pose` is the single source of truth for the voxel (`slice_point`/`follow_voxel` only control the views; change them via `set_slice_point`/`set_follow_voxel`). All edits (mouse, `ParameterPanel` fields, file loads) go through `MainWindow.set_pose`, which refreshes all views and the panel. `VoxelPose` and `Plane` are frozen dataclasses: use `translated`/`rotated`/`with_size`/`through` to derive new ones instead of mutating.
- **Voxel JSON** = `VoxelPose.to_dict()` (`center_ras_mm`, `axes_ras` stored as rows, i.e. transposed, and `size_mm`) plus a `siemens` block. `from_dict` reads only the pose fields.
