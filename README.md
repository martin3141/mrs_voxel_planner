# mrs_voxel_planner — MRS Voxel Planner

Emulates positioning a single-voxel MRS VOI on a 3D T1-weighted image, using
Siemens geometry conventions.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

## Run

```bash
.venv/bin/mrs_voxel_planner T1.nii.gz                        # your image
.venv/bin/mrs_voxel_planner $FSLDIR/data/standard/MNI152_T1_1mm.nii.gz
.venv/bin/mrs_voxel_planner --phantom                        # synthetic head
.venv/bin/mrs_voxel_planner T1.nii.gz --voxel scan.rda       # show a real scanner voxel
```

In any view:

| Action | Effect |
|---|---|
| drag a corner handle (■) | rotate the voxel about its centre, in that view's plane |
| shift + left-drag | same rotation, grabbing from anywhere |
| left-drag inside voxel | move the voxel within that plane |
| left-drag elsewhere | pan |
| wheel / right-drag | zoom |
| middle-drag | contrast (left/right) and brightness (up/down) of the T1 |
| ctrl + wheel | step that view's slice by one image voxel (turns off **Follow voxel**) |
| arrow keys | move the voxel 1 mm within the last-clicked view |
| ctrl + Z / ctrl + shift + Z | undo / redo voxel changes (also in the **Edit** menu) |

Rotating in the transverse, coronal and sagittal views turns the voxel about the
head–foot, anterior–posterior and left–right axes respectively, so combining them reaches any
orientation. Once the voxel is oblique its cut through a view can have 5 or 6 corners, and every
one of them is a handle.

The panel below the views shows and edits the Siemens parameters: position (LPS), readout FOV, phase FOV,
thickness, orientation angles, and in-plane rotation, plus the derived slice normal.

The **Slices** section sets where the views cut the T1. With **Follow voxel** ticked (the default)
all three views pass through the voxel centre. Untick it to set the sagittal, coronal and transverse
slice positions (LPS mm) independently of the voxel, e.g. to check where its edges fall. Ticking it
again snaps the views back to the voxel centre. Loading a voxel always moves the slices to it, and
slices stay within the image. **Show slice lines** draws dashed lines where each view's slice cuts
the other two, in the colour of that view's title.

The **Display** section has brightness and contrast sliders, shared by all three views.
**Auto** resets them to the 1st–99.5th percentile intensity window.

The **Angles** field uses the scanner's notation, e.g. `T > C -23.9 > S 4.5`. You can type one in
(`Tra>Cor(-23.9)>Sag(4.5)` is also accepted) and press Enter to set the slice normal; the voxel
centre and in-plane rotation are kept. The convention is checked against scanner-reported
(string, sNormal) pairs, which were all transverse. Coronal- and sagittal-based strings follow the
same rule but haven't been checked against a scanner yet.

**File menu:** load a voxel from JSON, Siemens `.rda` or single-voxel NIfTI-MRS (spec2nii);
save the voxel as JSON (world pose plus Siemens `sPosition`/`sNormal`/`dInPlaneRot`); export a
partial-volume voxel mask on the T1 grid (use with FAST/SPM segmentations for tissue fractions).
File dialogs open in the last folder used, which is remembered between sessions.

## Design

- `geometry.py`: the voxel is an oriented box (centre, axes, size) in RAS mm world space.
  Covers plane sections, hit testing and mask rasterisation. It never depends on the image grid.
- `siemens.py`: Siemens normal and in-plane rotation ↔ axes (a port of the scanner's
  CalcPRS routine), LPS↔RAS conversion, and RDA/NIfTI-MRS readers.
- `volume.py`: NIfTI loading and trilinear reslicing onto any plane using the affine.
- `viewer.py`: PyQt6 + pyqtgraph user interface. Radiological transverse, coronal and sagittal
  views through the voxel centre, or through independently set slice positions.

## Validating against the scanner

The box shape does not depend on convention details, but the reported in-plane rotation and the
readout/phase assignment do. Check them against a real acquisition: load the scan's `.rda`
(the reader cross-checks RowVector/ColumnVector against the derived axes and shows a warning on
a mismatch), or its spec2nii NIfTI-MRS, on the same subject's T1.

Run the tests with `.venv/bin/pytest`.

## License

MIT. See [LICENSE](LICENSE).
