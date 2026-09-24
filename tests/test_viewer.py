import json

import numpy as np
import pytest
from PyQt6 import QtCore, QtGui, QtWidgets

from mrs_voxel_planner.geometry import VoxelPose
from mrs_voxel_planner.siemens import SiemensVoxel, phase_readout
from mrs_voxel_planner.viewer import MainWindow
from mrs_voxel_planner.volume import make_phantom
from test_siemens import write_rda

Key = QtCore.Qt.Key
LEFT = QtCore.Qt.MouseButton.LeftButton
CTRL = QtCore.Qt.KeyboardModifier.ControlModifier


@pytest.fixture
def dialogs(monkeypatch):
    """Record message boxes instead of blocking on them."""
    shown = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical",
                        staticmethod(lambda parent, title, text: shown.append(("error", text))))
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning",
                        staticmethod(lambda parent, title, text: shown.append(("warning", text))))
    return shown


@pytest.fixture
def window(qtbot, tmp_path, dialogs):
    # Keep the remembered folder out of the user's real config.
    QtCore.QSettings.setPath(QtCore.QSettings.Format.NativeFormat,
                             QtCore.QSettings.Scope.UserScope, str(tmp_path / "config"))
    w = MainWindow()
    qtbot.addWidget(w)
    w.set_volume(make_phantom(shape=(64, 64, 64), voxel_mm=2.0))
    w.resize(1500, 720)
    with qtbot.waitActive(w):  # laid out, and active so clicks give a view keyboard focus
        w.show()
        w.activateWindow()
    return w


def voxel_screen_pos(view) -> QtCore.QPoint:
    return view.mapFromScene(view.vb.mapViewToScene(QtCore.QPointF(*view.center_uv())))


def drag(qtbot, view, dx: int):
    """Left-drag from the voxel centre, dx pixels to the right."""
    p0 = voxel_screen_pos(view)
    qtbot.mousePress(view.viewport(), LEFT, pos=p0)
    for d in range(5, dx + 1, 5):
        qtbot.mouseMove(view.viewport(), p0 + QtCore.QPoint(d, 0))
    qtbot.mouseRelease(view.viewport(), LEFT, pos=p0 + QtCore.QPoint(dx, 0))


def wheel(view, modifiers=QtCore.Qt.KeyboardModifier.NoModifier):
    """One notch of wheel-up over the middle of a view."""
    c = QtCore.QPointF(view.viewport().rect().center())
    ev = QtGui.QWheelEvent(c, view.viewport().mapToGlobal(c), QtCore.QPoint(0, 0),
                           QtCore.QPoint(0, 120), QtCore.Qt.MouseButton.NoButton, modifiers,
                           QtCore.Qt.ScrollPhase.NoScrollPhase, False)
    QtWidgets.QApplication.sendEvent(view.viewport(), ev)


def patch_file_dialogs(monkeypatch, result=""):
    """Make file dialogs return `result`; returns the start paths they were opened with."""
    starts = []

    def dialog(parent, title, start, filters):
        starts.append(start)
        return result, ""

    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", staticmethod(dialog))
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", staticmethod(dialog))
    return starts


# --- slices -----------------------------------------------------------------


def test_slices_follow_voxel_by_default(window):
    window.set_pose(window.pose.translated([10, -6, 4]))
    np.testing.assert_allclose(window.slice_point, window.pose.center)
    assert not window.slices.pos[0].isEnabled()


def test_unfollowed_slices_stay_put_and_can_be_set(window):
    window.slices.follow.setChecked(False)
    before = window.slice_point.copy()
    window.set_pose(window.pose.translated([10, -6, 4]))
    np.testing.assert_allclose(window.slice_point, before)
    assert window.slices.pos[0].isEnabled()

    window.slices.pos[2].setValue(12.0)  # Tra: LPS and RAS agree on z
    assert window.slice_point[2] == pytest.approx(12.0)

    window.slices.follow.setChecked(True)
    np.testing.assert_allclose(window.slice_point, window.pose.center)


def test_ctrl_wheel_steps_slice_and_stops_following(window):
    view = window.views[0]
    zoom = view.vb.viewRange()
    wheel(view, CTRL)
    assert not window.follow_voxel and not window.slices.follow.isChecked()
    assert window.slice_point[2] == pytest.approx(2.0)  # one 2 mm image voxel towards S
    assert view.vb.viewRange() == zoom

    wheel(view)  # plain wheel zooms, slice unchanged
    assert window.slice_point[2] == pytest.approx(2.0)
    assert view.vb.viewRange() != zoom


def test_voxel_not_drawn_when_off_slice(window):
    window.set_follow_voxel(False)
    window.set_pose(window.pose.translated([0, 0, 40]))
    transverse, coronal = window.views[0], window.views[1]
    assert len(transverse.polygon) == 0 and len(transverse.centre.data) == 0
    assert len(coronal.polygon) == 4


def test_slices_are_clamped_to_the_image(window):
    lo, hi = window.volume.world_bounds()
    window.views[0].request_step(1000)
    assert window.slice_point[2] == pytest.approx(hi[2])
    assert window.slices.pos[2].value() == pytest.approx(hi[2])

    window.slices.pos[0].setValue(-300)  # LPS sag -300 = RAS x +300
    assert window.slice_point[0] == pytest.approx(hi[0])


def test_slice_lines(window):
    lines = [line for v in window.views for line in v.slice_lines.values()]
    assert not any(line.isVisible() for line in lines)

    window.slices.lines.setChecked(True)
    assert all(line.isVisible() for line in lines)
    window.set_follow_voxel(False)
    window.set_slice_point([10.0, -20.0, 30.0])
    transverse = window.views[0]  # screen right = -x, up = +y
    coronal, sagittal = transverse.slice_lines["Coronal"], transverse.slice_lines["Sagittal"]
    assert (coronal.pos().x(), coronal.pos().y()) == pytest.approx((-10, -20))
    assert coronal.angle % 180 == pytest.approx(0)  # horizontal
    assert sagittal.angle % 180 == pytest.approx(90)  # vertical

    window.set_show_slice_lines(False)
    assert not window.slices.lines.isChecked()
    assert not any(line.isVisible() for line in lines)


# --- files ------------------------------------------------------------------


def test_loading_a_voxel_moves_the_slices_to_it(window, tmp_path):
    window.set_follow_voxel(False)
    path = tmp_path / "voxel.json"
    path.write_text(json.dumps(VoxelPose.default(center=(20.0, -30.0, 15.0)).to_dict()))
    window.load_voxel(str(path))
    np.testing.assert_allclose(window.slice_point, (20, -30, 15))
    assert all(len(v.polygon) for v in window.views)
    assert not window.follow_voxel


def test_bad_voxel_file_shows_error_and_keeps_voxel(window, tmp_path, dialogs):
    before = window.pose.center.copy()
    path = tmp_path / "bad.json"
    path.write_text("garbage")
    window.load_voxel(str(path))
    assert len(dialogs) == 1
    kind, text = dialogs[0]
    assert kind == "error" and "JSONDecodeError" in text
    np.testing.assert_allclose(window.pose.center, before)


def test_mismatched_rda_warns_every_time(window, tmp_path, dialogs):
    sv = SiemensVoxel((1.5, -20.0, 12.0), (0.0, 0.2588190451, 0.9659258263), 0.2, 20, 25, 15)
    gp, gr = phase_readout(sv.normal, sv.inplane_rot)
    write_rda(tmp_path / "bad.rda", sv, row=gp, col=-gr)
    for _ in range(2):
        window.load_voxel(str(tmp_path / "bad.rda"))
    assert [kind for kind, _ in dialogs] == ["warning", "warning"]
    assert "RowVector/ColumnVector" in dialogs[0][1]
    np.testing.assert_allclose(window.pose.center, (-1.5, 20.0, 12.0))  # still loaded (RAS)


def test_file_dialogs_start_in_last_folder(window, tmp_path, monkeypatch):
    starts = patch_file_dialogs(monkeypatch)
    window.load_voxel()
    assert starts == [""]

    data = tmp_path / "data"
    data.mkdir()
    (data / "voxel.json").write_text(json.dumps(VoxelPose.default().to_dict()))
    window.load_voxel(str(data / "voxel.json"))  # as from --voxel
    window.load_voxel()
    window.save_voxel()
    assert starts[1:] == [str(data), str(data / "voxel.json")]


def patch_scale_dialog(monkeypatch, choice: int | None):
    """Make the figure-resolution dialog pick `choice`x (None = cancel); returns its defaults."""
    defaults = []

    def get_item(parent, title, label, items, current, editable):
        defaults.append(items[current])
        if choice is None:
            return "", False
        return next(i for i in items if i.startswith(f"{choice}×")), True

    monkeypatch.setattr(QtWidgets.QInputDialog, "getItem", staticmethod(get_item))
    return defaults


def test_export_figure(window, tmp_path, monkeypatch, dialogs):
    size = window.views_row.size()
    path = tmp_path / "figure.png"
    patch_file_dialogs(monkeypatch, str(path))
    defaults = patch_scale_dialog(monkeypatch, 3)
    window.export_figure()
    image = QtGui.QImage(str(path))
    assert (image.width(), image.height()) == (size.width() * 3, size.height() * 3)
    assert defaults[0].startswith("2×")  # default the first time
    assert dialogs == []

    window.export_figure()  # the last choice is remembered
    assert defaults[1].startswith("3×")

    path.unlink()
    patch_scale_dialog(monkeypatch, None)  # cancelling writes nothing
    window.export_figure()
    assert not path.exists()

    patch_file_dialogs(monkeypatch, str(tmp_path / "missing" / "figure.png"))
    patch_scale_dialog(monkeypatch, 1)
    window.export_figure()
    assert [kind for kind, _ in dialogs] == ["error"]


def test_figure_matches_the_screen(window):
    # At 1x the redrawn figure is the same picture as a screenshot of the views.
    shot = window.views_row.grab().toImage()
    shot.setDevicePixelRatio(1)
    figure = window.render_figure(1)
    assert figure.size() == shot.size()
    assert figure.convertToFormat(shot.format()) == shot


# --- editing ----------------------------------------------------------------


def test_arrow_keys_nudge_voxel_in_clicked_view(window, qtbot):
    transverse = window.views[0]
    qtbot.mouseClick(transverse.viewport(), LEFT, pos=QtCore.QPoint(20, 20))
    assert transverse.hasFocus()
    start = window.pose.center.copy()
    qtbot.keyClick(transverse.viewport(), Key.Key_Right)  # screen right = patient left = -x
    qtbot.keyClick(transverse.viewport(), Key.Key_Up)  # screen up = anterior = +y
    np.testing.assert_allclose(window.pose.center - start, [-1, 1, 0])


def test_each_drag_is_one_undo_step(window, qtbot):
    view = window.views[0]
    xs = [window.pose.center[0]]
    for _ in range(2):
        drag(qtbot, view, 40)
        xs.append(window.pose.center[0])
        # A press within the double-click interval is a double-click, not a new drag.
        qtbot.wait(QtWidgets.QApplication.doubleClickInterval() + 50)
    assert xs[0] > xs[1] > xs[2]  # dragging right moves towards patient left
    assert len(window._undo) == 2
    window.undo()
    assert window.pose.center[0] == pytest.approx(xs[1])
    window.undo()
    assert window.pose.center[0] == pytest.approx(xs[0])


def test_undo_redo_panel_edits(window):
    window.panel.size[0].setValue(30)
    window.panel.size[1].setValue(35)
    window.undo()
    assert window.pose.size.tolist() == [30, 20, 20]
    window.undo()
    assert window.pose.size.tolist() == [20, 20, 20]
    window.redo()
    assert window.pose.size.tolist() == [30, 20, 20]
    window.panel.size[2].setValue(10)  # a new edit discards the redo history
    window.redo()
    assert window.pose.size.tolist() == [30, 20, 10]


def test_ctrl_z_undoes_nudge(window, qtbot):
    view = window.views[1]
    qtbot.mouseClick(view.viewport(), LEFT, pos=QtCore.QPoint(20, 20))
    before = window.pose.center.copy()
    qtbot.keyClick(view.viewport(), Key.Key_Left)
    assert not np.allclose(window.pose.center, before)
    qtbot.keyClick(window, Key.Key_Z, CTRL)
    np.testing.assert_allclose(window.pose.center, before)


# --- layout -----------------------------------------------------------------


def test_panel_columns_keep_their_width(window, qtbot):
    def column_x():
        return [w.mapTo(window, QtCore.QPoint()).x()
                for w in (window.panel.size[0], window.panel.inplane)]

    xs = column_x()
    for pos, size in [((-299.9, -299.9, -299.9), (200, 200, 200)), ((0, 5, 0), (5, 5, 5))]:
        for s, v in zip(window.panel.pos, pos):
            s.setValue(v)
        for s, v in zip(window.panel.size, size):
            s.setValue(v)
        qtbot.wait(0)
        assert column_x() == xs
