"""Qt user interface: three orthogonal views plus a Siemens-style parameter panel.

The views are fixed anatomical planes (radiological convention, as on the
scanner) that always pass through the voxel centre. The voxel appears as the
polygon where the box cuts each plane.

Mouse, in any view:
  drag a corner handle         rotate the voxel about the axis normal to that
                               plane (through the voxel centre)
  shift + left-drag            same rotation, from anywhere in the view
  left-drag inside the voxel   move it within that plane
  left-drag elsewhere          pan;  wheel / right-drag  zoom
  middle-drag                  image contrast (left/right) and brightness (up/down)

Rotating in each of the three views gives rotations about all three axes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

from .geometry import (Plane, VoxelPose, nearest_vertex, point_in_polygon, section_polygon,
                       voxel_mask)
from .siemens import (SiemensVoxel, format_position, normal_from_orientation_string,
                      orientation_string, pose_from_nifti_mrs, pose_from_siemens,
                      siemens_from_pose, siemens_from_rda)
from .volume import Volume, window_levels

pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)

# Radiological display: patient left on screen right; sagittal viewed from the
# patient's left with anterior on screen left.
VIEWS = {
    "Transverse": Plane(h=np.array([-1.0, 0, 0]), v=np.array([0, 1.0, 0])),
    "Coronal": Plane(h=np.array([-1.0, 0, 0]), v=np.array([0, 0, 1.0])),
    "Sagittal": Plane(h=np.array([0, -1.0, 0]), v=np.array([0, 0, 1.0])),
}
VIEW_LABELS = {"Transverse": ("R", "L", "A"), "Coronal": ("R", "L", "H"),
               "Sagittal": ("A", "P", "H")}  # screen left, screen right, screen top

VOXEL_PEN = pg.mkPen((255, 200, 0), width=2)
HANDLE_PX = 9  # handle size, and grab radius, in screen pixels
WINDOW_DRAG_GAIN = 0.5  # brightness/contrast units per pixel of middle-drag
Cursor = QtCore.Qt.CursorShape


class VoxelViewBox(pg.ViewBox):
    """ViewBox that turns drags on the voxel into translate/rotate requests."""

    def __init__(self, view: "SliceView"):
        super().__init__(lockAspect=True, invertY=False, enableMenu=False)
        self.view = view
        self._mode = None
        self._last = None

    def _hit(self, uv) -> str | None:
        """What a press at view point `uv` would grab: 'rotate', 'move' or None."""
        tol = HANDLE_PX * max(self.viewPixelSize())
        if nearest_vertex(uv, self.view.polygon, tol) is not None:
            return "rotate"
        if point_in_polygon(uv, self.view.polygon):
            return "move"
        return None

    def hoverEvent(self, ev):
        if ev.isExit() or self._mode is not None:
            return
        p = self.mapToView(ev.pos())
        hit = self._hit((p.x(), p.y()))
        self.view.setCursor({"rotate": Cursor.CrossCursor, "move": Cursor.SizeAllCursor}
                            .get(hit, Cursor.ArrowCursor))

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() == QtCore.Qt.MouseButton.MiddleButton:
            ev.accept()
            d = ev.pos() - ev.lastPos()  # screen pixels, y down
            self.view.window.display.nudge(brightness=-d.y() * WINDOW_DRAG_GAIN,
                                           contrast=d.x() * WINDOW_DRAG_GAIN)
            return
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            return super().mouseDragEvent(ev, axis)
        pos = self.mapSceneToView(ev.scenePos())
        pos = np.array([pos.x(), pos.y()])
        if ev.isStart():
            start = self.mapSceneToView(ev.buttonDownScenePos())
            self._last = np.array([start.x(), start.y()])
            if ev.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
                self._mode = "rotate"
            else:
                self._mode = self._hit(self._last)
        if self._mode is None:
            return super().mouseDragEvent(ev, axis)
        ev.accept()
        if self._mode == "move":
            self.view.request_move(pos - self._last)
        else:
            c = self.view.center_uv()
            a0 = np.arctan2(*(self._last - c)[::-1])
            a1 = np.arctan2(*(pos - c)[::-1])
            self.view.request_rotate(a1 - a0)
        self._last = pos
        if ev.isFinish():
            self._mode = None


class SliceView(pg.GraphicsLayoutWidget):
    def __init__(self, name: str, window: "MainWindow"):
        super().__init__()
        self.name = name
        self.window = window
        self.plane = VIEWS[name]
        self.polygon = np.zeros((0, 2))
        self._cached_offset = None

        lo, hi, top = VIEW_LABELS[name]
        self.addLabel(f"<b>{name}</b> &nbsp; {lo} ← → {hi}, up = {top}", row=0, col=0)
        self.vb = VoxelViewBox(self)
        self.addItem(self.vb, row=1, col=0)
        self.image = pg.ImageItem()
        self.outline = pg.PlotCurveItem(pen=VOXEL_PEN)
        self.centre = pg.ScatterPlotItem(size=6, pen=None, brush=(255, 200, 0))
        self.handles = pg.ScatterPlotItem(size=HANDLE_PX, symbol="s", pen=VOXEL_PEN,
                                          brush=(0, 0, 0, 160))
        for item in (self.image, self.outline, self.centre, self.handles):
            self.vb.addItem(item)

    def set_volume(self, vol: Volume):
        self._cached_offset = None

    def refresh(self, vol: Volume | None, pose: VoxelPose, reset_range: bool = False):
        self.plane = self.plane.through(pose.center)
        if vol is not None and self._cached_offset != round(self.plane.offset, 3):
            pim = vol.sample_plane(self.plane)
            self.image.setImage(pim.image, levels=self.window.display.levels(), autoLevels=False)
            self.image.setRect(QtCore.QRectF(*pim.extent))
            self._cached_offset = round(self.plane.offset, 3)
            if reset_range:
                self.vb.autoRange(padding=0.02)
        self.polygon = section_polygon(pose, self.plane)
        closed = np.vstack([self.polygon, self.polygon[:1]]) if len(self.polygon) else self.polygon
        self.outline.setData(closed[:, 0], closed[:, 1]) if len(closed) else self.outline.clear()
        self.handles.setData(self.polygon[:, 0], self.polygon[:, 1])
        self.centre.setData(*[[v] for v in self.center_uv()])

    def center_uv(self) -> np.ndarray:
        return self.plane.to_plane(self.window.pose.center)

    def request_move(self, duv):
        delta = duv[0] * self.plane.h + duv[1] * self.plane.v
        self.window.set_pose(self.window.pose.translated(delta))

    def request_rotate(self, angle: float):
        self.window.set_pose(self.window.pose.rotated(self.plane.n, angle))


class ParameterPanel(QtWidgets.QWidget):
    """Siemens-style protocol parameters, editable and kept in sync with the pose."""

    def __init__(self, window: "MainWindow"):
        super().__init__()
        self.window = window
        form = QtWidgets.QFormLayout(self)

        def spin(lo, hi, step, suffix, decimals=1):
            s = QtWidgets.QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setSingleStep(step)
            s.setDecimals(decimals)
            s.setSuffix(suffix)
            s.setKeyboardTracking(False)
            s.valueChanged.connect(self._edited)
            return s

        form.addRow(QtWidgets.QLabel("<b>Position (LPS, mm)</b>"))
        self.pos = [spin(-300, 300, 1, " mm") for _ in range(3)]
        for label, s in zip(("Sag", "Cor", "Tra"), self.pos):
            form.addRow(label, s)
        self.pos_text = QtWidgets.QLabel()
        form.addRow("", self.pos_text)

        form.addRow(QtWidgets.QLabel("<b>Size</b>"))
        self.size = [spin(1, 200, 1, " mm") for _ in range(3)]
        for label, s in zip(("Readout FOV", "Phase FOV", "Thickness"), self.size):
            form.addRow(label, s)

        form.addRow(QtWidgets.QLabel("<b>Orientation</b>"))
        self.orient_edit = QtWidgets.QLineEdit()
        self.orient_edit.setToolTip("Scanner-style angles, e.g. 'T > C -12.3 > S 4.1'. "
                                    "Edit and press Enter to set the slice normal.")
        self.orient_edit.editingFinished.connect(self._orientation_edited)
        form.addRow("Angles", self.orient_edit)
        self.inplane = spin(-180, 180, 1, " °")
        form.addRow("In-plane rot.", self.inplane)
        self.orient_text = QtWidgets.QLabel()
        self.orient_text.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.orient_text.setToolTip("Slice normal (sNormal: Sag, Cor, Tra), LPS")
        form.addRow("Normal", self.orient_text)
        reset = QtWidgets.QPushButton("Reset to transverse")
        reset.clicked.connect(self._reset_orientation)
        form.addRow(reset)

        self.volume_text = QtWidgets.QLabel()
        form.addRow("Volume", self.volume_text)

    def _widgets(self):
        return [*self.pos, *self.size, self.inplane]

    def show_pose(self, pose: VoxelPose):
        sv = siemens_from_pose(pose)
        for w in self._widgets():
            w.blockSignals(True)
        for s, v in zip(self.pos, sv.position):
            s.setValue(v)
        for s, v in zip(self.size, (sv.readout_fov, sv.phase_fov, sv.thickness)):
            s.setValue(v)
        self.inplane.setValue(np.degrees(sv.inplane_rot))
        for w in self._widgets():
            w.blockSignals(False)
        self.pos_text.setText(format_position(sv.position))
        n = sv.normal
        self.orient_edit.setText(orientation_string(n))
        self.orient_edit.setCursorPosition(0)
        self.orient_text.setText(f"({n[0]:+.4f}, {n[1]:+.4f}, {n[2]:+.4f})")
        self.volume_text.setText(f"{pose.volume_mm3 / 1000:.2f} mL")

    def _edited(self):
        old = siemens_from_pose(self.window.pose)
        sv = SiemensVoxel(position=tuple(s.value() for s in self.pos), normal=old.normal,
                          inplane_rot=np.radians(self.inplane.value()),
                          readout_fov=self.size[0].value(), phase_fov=self.size[1].value(),
                          thickness=self.size[2].value())
        self.window.set_pose(pose_from_siemens(sv))

    def _orientation_edited(self):
        if not self.orient_edit.isModified():
            return
        self.orient_edit.setModified(False)
        old = siemens_from_pose(self.window.pose)
        try:
            normal = normal_from_orientation_string(self.orient_edit.text())
        except ValueError as e:
            self.window.statusBar().showMessage(str(e), 5000)
            self.orient_edit.setText(orientation_string(old.normal))
            return
        self.window.set_pose(pose_from_siemens(SiemensVoxel(
            old.position, tuple(normal), old.inplane_rot, old.readout_fov, old.phase_fov,
            old.thickness)))

    def _reset_orientation(self):
        p = self.window.pose
        self.window.set_pose(VoxelPose.default(center=p.center, size=p.size))


class DisplayPanel(QtWidgets.QWidget):
    """Brightness/contrast of the T1, shared by all views."""

    def __init__(self, window: "MainWindow"):
        super().__init__()
        self.window = window
        self.auto_range = (0.0, 1.0)
        self._drag_remainder = np.zeros(2)  # fractional slider steps from middle-drag
        form = QtWidgets.QFormLayout(self)
        form.addRow(QtWidgets.QLabel("<b>Display</b>"))

        def slider():
            s = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            s.setRange(-100, 100)
            s.valueChanged.connect(self._apply)
            return s

        self.brightness = slider()
        self.contrast = slider()
        form.addRow("Brightness", self.brightness)
        form.addRow("Contrast", self.contrast)
        self.levels_text = QtWidgets.QLabel()
        reset = QtWidgets.QPushButton("Auto")
        reset.setToolTip("Reset to the 1st–99.5th percentile window")
        reset.clicked.connect(self.reset)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.levels_text, stretch=1)
        row.addWidget(reset)
        form.addRow(row)

    def levels(self) -> tuple[float, float]:
        return window_levels(self.auto_range, self.brightness.value(), self.contrast.value())

    def set_auto_range(self, auto_range):
        self.auto_range = auto_range
        self.reset()

    def reset(self):
        for s in (self.brightness, self.contrast):
            s.blockSignals(True)
            s.setValue(0)
            s.blockSignals(False)
        self._apply()

    def nudge(self, brightness: float = 0.0, contrast: float = 0.0):
        # Accumulate fractional steps so slow drags still move the integer sliders.
        self._drag_remainder += (brightness, contrast)
        steps = np.trunc(self._drag_remainder)
        self._drag_remainder -= steps
        self.brightness.setValue(self.brightness.value() + int(steps[0]))
        self.contrast.setValue(self.contrast.value() + int(steps[1]))

    def _apply(self):
        lo, hi = self.levels()
        self.levels_text.setText(f"Level {(lo + hi) / 2:.0f}   Width {hi - lo:.0f}")
        for v in self.window.views:
            v.image.setLevels((lo, hi))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MRS Voxel Planner")
        self.volume: Volume | None = None
        self.pose = VoxelPose.default()

        self.views = [SliceView(name, self) for name in VIEWS]
        self.display = DisplayPanel(self)
        self.panel = ParameterPanel(self)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)
        splitter = QtWidgets.QSplitter()
        for v in self.views:
            splitter.addWidget(v)
        layout.addWidget(splitter, stretch=1)
        side_widget = QtWidgets.QWidget()
        side_layout = QtWidgets.QVBoxLayout(side_widget)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.addWidget(self.panel)
        side_layout.addWidget(self.display)
        side_layout.addStretch(1)
        side = QtWidgets.QScrollArea()
        side.setWidget(side_widget)
        side.setWidgetResizable(True)
        side.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        side.setFixedWidth(340)
        layout.addWidget(side)
        self.setCentralWidget(central)
        self._build_menu()
        self.statusBar().showMessage("Open a T1 image (File > Open T1…)")
        self.set_pose(self.pose)

    def _build_menu(self):
        m = self.menuBar().addMenu("&File")
        for text, slot, key in [
            ("Open T1…", self.open_t1, "Ctrl+O"),
            (None, None, None),
            ("Load voxel (JSON / Siemens RDA / NIfTI-MRS)…", self.load_voxel, "Ctrl+L"),
            ("Save voxel…", self.save_voxel, "Ctrl+S"),
            ("Export voxel mask…", self.export_mask, "Ctrl+E"),
            (None, None, None),
            ("Quit", self.close, "Ctrl+Q"),
        ]:
            if text is None:
                m.addSeparator()
            else:
                m.addAction(text, key, slot)

    # --- state -----------------------------------------------------------

    def set_volume(self, vol: Volume, centre_voxel: bool = True):
        self.volume = vol
        for v in self.views:
            v.set_volume(vol)
        self.display.set_auto_range(vol.display_range)
        if centre_voxel:
            self.pose = VoxelPose(vol.world_center(), self.pose.axes, self.pose.size)
        self.set_pose(self.pose, reset_range=True)
        self.statusBar().showMessage(f"{vol.name}  {vol.data.shape}  "
                                     f"{np.round(vol.voxel_sizes, 2).tolist()} mm")

    def set_pose(self, pose: VoxelPose, reset_range: bool = False):
        self.pose = pose
        for v in self.views:
            v.refresh(self.volume, pose, reset_range=reset_range)
        self.panel.show_pose(pose)

    # --- file actions ----------------------------------------------------

    def open_t1(self, path=None):
        path = path or QtWidgets.QFileDialog.getOpenFileName(
            self, "Open T1", "", "NIfTI (*.nii *.nii.gz)")[0]
        if path:
            self.set_volume(Volume.load(path))

    def load_voxel(self, path=None):
        path = path or QtWidgets.QFileDialog.getOpenFileName(
            self, "Load voxel", "", "Voxel (*.json *.rda *.nii *.nii.gz)")[0]
        if not path:
            return
        p = Path(path)
        if p.suffix.lower() == ".json":
            pose = VoxelPose.from_dict(json.loads(p.read_text()))
        elif p.suffix.lower() == ".rda":
            pose = pose_from_siemens(siemens_from_rda(p))
        else:
            pose = pose_from_nifti_mrs(p)
        self.set_pose(pose)

    def save_voxel(self):
        path = QtWidgets.QFileDialog.getSaveFileName(self, "Save voxel", "voxel.json",
                                                     "JSON (*.json)")[0]
        if path:
            d = self.pose.to_dict()
            d["siemens"] = siemens_from_pose(self.pose).to_dict()
            Path(path).write_text(json.dumps(d, indent=2))

    def export_mask(self):
        if self.volume is None:
            return
        path = QtWidgets.QFileDialog.getSaveFileName(self, "Export mask", "voxel_mask.nii.gz",
                                                     "NIfTI (*.nii.gz *.nii)")[0]
        if not path:
            return
        import nibabel as nib

        mask = voxel_mask(self.pose, self.volume.data.shape, self.volume.affine)
        nib.save(nib.Nifti1Image(mask, self.volume.affine), path)
        vox_ml = np.prod(self.volume.voxel_sizes) / 1000
        self.statusBar().showMessage(f"Saved {path}: mask {mask.sum() * vox_ml:.2f} mL "
                                     f"(box {self.pose.volume_mm3 / 1000:.2f} mL)")
