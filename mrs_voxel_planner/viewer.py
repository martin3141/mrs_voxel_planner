"""Qt user interface: three orthogonal views plus a Siemens-style parameter panel.

The views are fixed anatomical planes (radiological convention, as on the
scanner). By default they pass through the voxel centre; untick "Follow voxel"
to position the slices independently. The voxel appears as the polygon where
the box cuts each plane.

Mouse, in any view:
  drag a corner handle         rotate the voxel about the axis normal to that
                               plane (through the voxel centre)
  shift + left-drag            same rotation, from anywhere in the view
  left-drag inside the voxel   move it within that plane
  left-drag elsewhere          pan;  wheel / right-drag  zoom
  middle-drag                  image contrast (left/right) and brightness (up/down)
  ctrl + wheel                 step that view's slice (stops the slices following the voxel)

Keyboard: arrow keys move the voxel 1 mm in the last-clicked view; ctrl+Z / ctrl+shift+Z
undo and redo voxel changes.

Rotating in each of the three views gives rotations about all three axes.
"""

from __future__ import annotations

import json
import warnings
from contextlib import contextmanager
from itertools import product
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

from .geometry import (Plane, VoxelPose, nearest_vertex, point_in_polygon, section_polygon,
                       voxel_mask)
from .siemens import (LPS_RAS, SiemensVoxel, format_position, normal_from_orientation_string,
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
# Each view's colour: its title, and the line marking its slice in the other views.
SLICE_COLOURS = {"Transverse": (80, 160, 255), "Coronal": (80, 220, 120),
                 "Sagittal": (255, 90, 160)}
HANDLE_PX = 9  # handle size, and grab radius, in screen pixels
WINDOW_DRAG_GAIN = 0.5  # brightness/contrast units per pixel of middle-drag
NUDGE_MM = 1.0  # arrow-key voxel step
UNDO_LIMIT = 100
Cursor = QtCore.Qt.CursorShape


def _same_pose(a: VoxelPose, b: VoxelPose) -> bool:
    return all(np.array_equal(x, y) for x, y in
               ((a.center, b.center), (a.axes, b.axes), (a.size, b.size)))


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

    def wheelEvent(self, ev, axis=None):
        if ev.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
            ev.accept()
            self.view.request_step(int(np.sign(ev.delta())))
        else:
            super().wheelEvent(ev, axis)

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
            if self._mode is not None:
                self.view.window.remember_pose()  # one undo step per drag
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
        colour = "#{:02x}{:02x}{:02x}".format(*SLICE_COLOURS[name])
        self.addLabel(f"<b style='color: {colour}'>{name}</b> &nbsp; {lo} ← → {hi}, up = {top}",
                      row=0, col=0)
        self.vb = VoxelViewBox(self)
        self.addItem(self.vb, row=1, col=0)
        self.image = pg.ImageItem()
        self.outline = pg.PlotCurveItem(pen=VOXEL_PEN)
        self.centre = pg.ScatterPlotItem(size=6, pen=None, brush=(255, 200, 0))
        self.handles = pg.ScatterPlotItem(size=HANDLE_PX, symbol="s", pen=VOXEL_PEN,
                                          brush=(0, 0, 0, 160))
        # Where the other two views' slices cut this one.
        self.slice_lines = {}
        for other in VIEWS:
            if other != name:
                d = np.cross(self.plane.n, VIEWS[other].n)
                self.slice_lines[other] = pg.InfiniteLine(
                    angle=np.degrees(np.arctan2(d @ self.plane.v, d @ self.plane.h)),
                    pen=pg.mkPen(SLICE_COLOURS[other], style=QtCore.Qt.PenStyle.DashLine))
        for item in (self.image, *self.slice_lines.values(), self.outline, self.centre,
                     self.handles):
            self.vb.addItem(item)

    def set_volume(self, vol: Volume):
        self._cached_offset = None

    def refresh(self, vol: Volume | None, pose: VoxelPose, slice_point,
                reset_range: bool = False):
        self.plane = self.plane.through(slice_point)
        if vol is not None and self._cached_offset != round(self.plane.offset, 3):
            pim = vol.sample_plane(self.plane)
            self.image.setImage(pim.image, levels=self.window.display.levels(), autoLevels=False)
            self.image.setRect(QtCore.QRectF(*pim.extent))
            self._cached_offset = round(self.plane.offset, 3)
            if reset_range:
                self.vb.autoRange(padding=0.02)
        for line in self.slice_lines.values():
            line.setPos(self.plane.to_plane(slice_point))
            line.setVisible(self.window.show_slice_lines)
        self.polygon = section_polygon(pose, self.plane)
        closed = np.vstack([self.polygon, self.polygon[:1]]) if len(self.polygon) else self.polygon
        self.outline.setData(closed[:, 0], closed[:, 1]) if len(closed) else self.outline.clear()
        self.handles.setData(self.polygon[:, 0], self.polygon[:, 1])
        if len(self.polygon):
            self.centre.setData(*[[v] for v in self.center_uv()])
        else:
            self.centre.clear()

    def center_uv(self) -> np.ndarray:
        return self.plane.to_plane(self.window.pose.center)

    def keyPressEvent(self, ev):
        Key = QtCore.Qt.Key
        step = {Key.Key_Left: (-1, 0), Key.Key_Right: (1, 0),
                Key.Key_Up: (0, 1), Key.Key_Down: (0, -1)}.get(ev.key())
        if step is None:
            return super().keyPressEvent(ev)
        self.window.remember_pose()
        self.request_move(np.multiply(step, NUDGE_MM))

    def request_move(self, duv):
        delta = duv[0] * self.plane.h + duv[1] * self.plane.v
        self.window.set_pose(self.window.pose.translated(delta))

    def request_rotate(self, angle: float):
        self.window.set_pose(self.window.pose.rotated(self.plane.n, angle))

    def request_step(self, steps: int):
        """Move this view's slice by whole image voxels, towards S, A or R for steps > 0."""
        vol = self.window.volume
        spacing = vol.voxel_sizes.min() if vol is not None else 1.0
        self.window.set_follow_voxel(False)
        self.window.set_slice_point(self.window.slice_point
                                    + steps * spacing * np.abs(self.plane.n))


class ParameterPanel(QtWidgets.QWidget):
    """Siemens-style protocol parameters, editable and kept in sync with the pose."""

    def __init__(self, window: "MainWindow"):
        super().__init__()
        self.window = window
        columns = QtWidgets.QHBoxLayout(self)
        columns.setContentsMargins(0, 0, 0, 0)

        def column(title):
            form = QtWidgets.QFormLayout()
            form.addRow(QtWidgets.QLabel(f"<b>{title}</b>"))
            if columns.count():
                columns.addSpacing(24)
            columns.addLayout(form)
            return form

        def spin(lo, hi, step, suffix, decimals=1):
            s = QtWidgets.QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setSingleStep(step)
            s.setDecimals(decimals)
            s.setSuffix(suffix)
            s.setKeyboardTracking(False)
            s.valueChanged.connect(self._edited)
            return s

        form = column("Position (LPS, mm)")
        self.pos = [spin(-300, 300, 1, " mm") for _ in range(3)]
        for label, s in zip(("Sag", "Cor", "Tra"), self.pos):
            form.addRow(label, s)
        self.pos_text = QtWidgets.QLabel()
        # Reserve the widest text the position range allows, so the column doesn't resize.
        fm = self.pos_text.fontMetrics()
        self.pos_text.setMinimumWidth(max(
            fm.horizontalAdvance(format_position(np.multiply(signs, 888.8)))
            for signs in product((1, -1), repeat=3)))
        form.addRow("", self.pos_text)

        form = column("Size")
        self.size = [spin(1, 200, 1, " mm") for _ in range(3)]
        for label, s in zip(("Readout FOV", "Phase FOV", "Thickness"), self.size):
            form.addRow(label, s)
        self.volume_text = QtWidgets.QLabel()
        # Room for the largest volume the size limits allow (200^3 mm^3 = 8000.00 mL).
        self.volume_text.setMinimumWidth(
            self.volume_text.fontMetrics().horizontalAdvance("8888.88 mL"))
        form.addRow("Volume", self.volume_text)

        form = column("Orientation")
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
        self.window.remember_pose()
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
        self.window.remember_pose()
        self.window.set_pose(pose_from_siemens(SiemensVoxel(
            old.position, tuple(normal), old.inplane_rot, old.readout_fov, old.phase_fov,
            old.thickness)))

    def _reset_orientation(self):
        p = self.window.pose
        self.window.remember_pose()
        self.window.set_pose(VoxelPose.default(center=p.center, size=p.size))


class SlicePanel(QtWidgets.QWidget):
    """Where the views cut the volume: through the voxel centre, or set independently."""

    def __init__(self, window: "MainWindow"):
        super().__init__()
        self.window = window
        form = QtWidgets.QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        form.addRow(QtWidgets.QLabel("<b>Slices (LPS, mm)</b>"))
        self.follow = QtWidgets.QCheckBox("Follow voxel")
        self.follow.setChecked(True)
        self.follow.setToolTip("Keep all three views through the voxel centre. Untick to set "
                               "the slices yourself, here or with ctrl + wheel in a view.")
        self.follow.toggled.connect(self.window.set_follow_voxel)
        form.addRow(self.follow)
        self.lines = QtWidgets.QCheckBox("Show slice lines")
        self.lines.setToolTip("Mark where each view's slice cuts the other two views, "
                              "in the colour of that view's title.")
        self.lines.toggled.connect(self.window.set_show_slice_lines)
        form.addRow(self.lines)
        self.pos = []
        for label in ("Sag", "Cor", "Tra"):
            s = QtWidgets.QDoubleSpinBox()
            s.setRange(-300, 300)
            s.setDecimals(1)
            s.setSuffix(" mm")
            s.setKeyboardTracking(False)
            s.valueChanged.connect(self._edited)
            form.addRow(label, s)
            self.pos.append(s)

    def set_bounds(self, lo_ras, hi_ras):
        a, b = LPS_RAS @ lo_ras, LPS_RAS @ hi_ras
        for s, lo, hi in zip(self.pos, np.minimum(a, b), np.maximum(a, b)):
            s.blockSignals(True)
            s.setRange(lo, hi)
            s.blockSignals(False)

    def show_point(self, point_ras):
        for s, v in zip(self.pos, LPS_RAS @ point_ras):
            s.blockSignals(True)
            s.setValue(v)
            s.blockSignals(False)
            s.setEnabled(not self.window.follow_voxel)

    def _edited(self):
        self.window.set_slice_point(LPS_RAS @ np.array([s.value() for s in self.pos]))


class DisplayPanel(QtWidgets.QWidget):
    """Brightness/contrast of the T1, shared by all views."""

    def __init__(self, window: "MainWindow"):
        super().__init__()
        self.window = window
        self.auto_range = (0.0, 1.0)
        self._drag_remainder = np.zeros(2)  # fractional slider steps from middle-drag
        form = QtWidgets.QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
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
        self.slice_point = self.pose.center  # RAS point all three views pass through
        self.follow_voxel = True
        self.show_slice_lines = False
        self._undo: list[VoxelPose] = []
        self._redo: list[VoxelPose] = []
        self.settings = QtCore.QSettings("mrs_voxel_planner", "mrs_voxel_planner")

        self.views = [SliceView(name, self) for name in VIEWS]
        self.display = DisplayPanel(self)
        self.slices = SlicePanel(self)
        self.panel = ParameterPanel(self)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        self.views_row = QtWidgets.QSplitter()
        for v in self.views:
            self.views_row.addWidget(v)
        layout.addWidget(self.views_row, stretch=1)
        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(24)
        controls.addWidget(self.panel, alignment=QtCore.Qt.AlignmentFlag.AlignTop)
        controls.addWidget(self.slices, alignment=QtCore.Qt.AlignmentFlag.AlignTop)
        controls.addWidget(self.display, alignment=QtCore.Qt.AlignmentFlag.AlignTop)
        controls.addStretch(1)
        layout.addLayout(controls)
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
            ("Export figure…", self.export_figure, "Ctrl+Shift+E"),
            (None, None, None),
            ("Quit", self.close, "Ctrl+Q"),
        ]:
            if text is None:
                m.addSeparator()
            else:
                m.addAction(text, key, slot)
        e = self.menuBar().addMenu("&Edit")
        e.addAction("Undo voxel change", "Ctrl+Z", self.undo)
        e.addAction("Redo voxel change", "Ctrl+Shift+Z", self.redo)

    # --- state -----------------------------------------------------------

    def set_volume(self, vol: Volume, centre_voxel: bool = True):
        self.volume = vol
        for v in self.views:
            v.set_volume(vol)
        self.display.set_auto_range(vol.display_range)
        self.slices.set_bounds(*vol.world_bounds())
        if centre_voxel:
            self.pose = VoxelPose(vol.world_center(), self.pose.axes, self.pose.size)
        self.slice_point = self.pose.center
        self.set_pose(self.pose, reset_range=True)
        self.statusBar().showMessage(f"{vol.name}  {vol.data.shape}  "
                                     f"{np.round(vol.voxel_sizes, 2).tolist()} mm")

    def set_pose(self, pose: VoxelPose, reset_range: bool = False):
        self.pose = pose
        if self.follow_voxel:
            self.slice_point = pose.center
        self._refresh_views(reset_range)
        self.panel.show_pose(pose)

    def remember_pose(self):
        """Record the current pose for undo. Call before a user edit changes it."""
        if not self._undo or not _same_pose(self._undo[-1], self.pose):
            self._undo.append(self.pose)
            del self._undo[:-UNDO_LIMIT]
        self._redo.clear()

    def undo(self):
        if self._undo:
            self._redo.append(self.pose)
            self.set_pose(self._undo.pop())

    def redo(self):
        if self._redo:
            self._undo.append(self.pose)
            self.set_pose(self._redo.pop())

    def set_slice_point(self, point):
        self.slice_point = np.asarray(point, dtype=float)
        self._refresh_views()

    def set_follow_voxel(self, on: bool):
        if on == self.follow_voxel:
            return
        self.follow_voxel = on
        self.slices.follow.setChecked(on)
        if on:
            self.slice_point = self.pose.center
        self._refresh_views()

    def set_show_slice_lines(self, on: bool):
        if on == self.show_slice_lines:
            return
        self.show_slice_lines = on
        self.slices.lines.setChecked(on)
        self._refresh_views()

    def _refresh_views(self, reset_range: bool = False):
        if self.volume is not None:  # keep the slices within the image
            self.slice_point = np.clip(self.slice_point, *self.volume.world_bounds())
        for v in self.views:
            v.refresh(self.volume, self.pose, self.slice_point, reset_range=reset_range)
        self.slices.show_point(self.slice_point)

    # --- file actions ----------------------------------------------------

    @contextmanager
    def _reporting(self, action: str):
        """Show errors and warnings from a file action in a dialog, not just on stderr.

        Warnings are shown every time (Python normally shows each only once), so a
        second RDA whose direction vectors don't match is still reported.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                yield
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, action,
                                               f"{action} failed.\n\n{type(e).__name__}: {e}")
        if caught:
            QtWidgets.QMessageBox.warning(self, action,
                                          "\n\n".join(str(w.message) for w in caught))

    def _choose_file(self, title: str, filters: str, save_name: str | None = None) -> str:
        """Open (or, given save_name, save) file dialog starting in the last folder used."""
        last = self.settings.value("last_dir", "", type=str)
        if save_name is None:
            return QtWidgets.QFileDialog.getOpenFileName(self, title, last, filters)[0]
        start = str(Path(last) / save_name) if last else save_name
        return QtWidgets.QFileDialog.getSaveFileName(self, title, start, filters)[0]

    def _remember_dir(self, path):
        self.settings.setValue("last_dir", str(Path(path).resolve().parent))

    def open_t1(self, path=None):
        path = path or self._choose_file("Open T1", "NIfTI (*.nii *.nii.gz)")
        if path:
            self._remember_dir(path)
            with self._reporting("Open T1"):
                self.set_volume(Volume.load(path))

    def load_voxel(self, path=None):
        path = path or self._choose_file("Load voxel", "Voxel (*.json *.rda *.nii *.nii.gz)")
        if not path:
            return
        self._remember_dir(path)
        p = Path(path)
        with self._reporting("Load voxel"):
            if p.suffix.lower() == ".json":
                pose = VoxelPose.from_dict(json.loads(p.read_text()))
            elif p.suffix.lower() == ".rda":
                pose = pose_from_siemens(siemens_from_rda(p))
            else:
                pose = pose_from_nifti_mrs(p)
            self.slice_point = pose.center  # show it even if the slices aren't following
            self.remember_pose()
            self.set_pose(pose)

    def save_voxel(self):
        path = self._choose_file("Save voxel", "JSON (*.json)", "voxel.json")
        if path:
            self._remember_dir(path)
            d = self.pose.to_dict()
            d["siemens"] = siemens_from_pose(self.pose).to_dict()
            with self._reporting("Save voxel"):
                Path(path).write_text(json.dumps(d, indent=2))

    def export_mask(self):
        if self.volume is None:
            return
        path = self._choose_file("Export mask", "NIfTI (*.nii.gz *.nii)", "voxel_mask.nii.gz")
        if not path:
            return
        self._remember_dir(path)
        import nibabel as nib

        mask = voxel_mask(self.pose, self.volume.data.shape, self.volume.affine)
        with self._reporting("Export voxel mask"):
            nib.save(nib.Nifti1Image(mask, self.volume.affine), path)
            vox_ml = np.prod(self.volume.voxel_sizes) / 1000
            self.statusBar().showMessage(f"Saved {path}: mask {mask.sum() * vox_ml:.2f} mL "
                                         f"(box {self.pose.volume_mm3 / 1000:.2f} mL)")

    def export_figure(self):
        """Save the three views, as currently shown, as an image (e.g. for a methods figure)."""
        path = self._choose_file("Export figure", "Images (*.png *.jpg *.tif)", "voxel_figure.png")
        if not path:
            return
        self._remember_dir(path)
        with self._reporting("Export figure"):
            if not self.views_row.grab().save(path):
                raise OSError(f"could not write {path} (unsupported image type or folder)")
            self.statusBar().showMessage(f"Saved {path}")
