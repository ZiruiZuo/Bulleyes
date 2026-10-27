import json

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from matplotlib.backend_bases import FigureCanvasBase, MouseButton, MouseEvent

from cardiac_bullseye.territory.io import (
    load_center_overrides_json,
    load_deleted_slice_indices,
    save_landmarks_json,
)
from cardiac_bullseye.territory.models import LandmarkPair
from cardiac_bullseye.territory.models import StudyData


def _closed_ring_stack(
    slice_count: int,
    shape: tuple[int, int],
    *,
    center_xy: tuple[int, int] | None = None,
) -> np.ndarray:
    height, width = shape
    center_x, center_y = center_xy or (width // 2, height // 2)
    mask = np.zeros((slice_count, height, width), dtype=np.uint8)
    mask[:, center_y - 1 : center_y + 2, center_x - 1 : center_x + 2] = 1
    mask[:, center_y, center_x] = 0
    return mask


def test_delete_button_marks_slice_deleted_and_advances(monkeypatch, tmp_path):
    from cardiac_bullseye.territory import landmarks as module

    callbacks = {}

    class FakeButton:
        def __init__(self, _axis, label):
            self.label = label

        def on_clicked(self, callback):
            callbacks[self.label] = callback

    def click_delete(_canvas, timeout=0):
        callbacks["Delete"](None)

    monkeypatch.setattr(module, "Button", FakeButton)
    monkeypatch.setattr(module, "_ensure_interactive_backend", lambda: None)
    monkeypatch.setattr(module.plt, "show", lambda **_kwargs: None)
    monkeypatch.setattr(FigureCanvasBase, "start_event_loop", click_delete)
    study = StudyData(
        mr=np.ones((2, 9, 9), dtype=float),
        territory_mask=_closed_ring_stack(2, (9, 9)),
        slice_positions=np.asarray([0.0, 1.0]),
    )
    output = tmp_path / "landmarks.json"

    selected = module.select_landmarks_interactively(study, output)

    assert selected == [None, None]
    assert load_deleted_slice_indices(output, 2) == {0, 1}


def test_dragged_center_is_retained_when_the_slice_is_skipped(monkeypatch, tmp_path):
    from cardiac_bullseye.territory import landmarks as module

    callbacks = {}

    class FakeButton:
        def __init__(self, _axis, label):
            self.label = label

        def on_clicked(self, callback):
            callbacks[self.label] = callback

    def drag_center_then_skip(canvas, timeout=0):
        axis = canvas.figure.axes[0]

        def dispatch(name, point, button=MouseButton.LEFT):
            display_x, display_y = axis.transData.transform(point)
            event = MouseEvent(
                name,
                canvas,
                display_x,
                display_y,
                button=button,
            )
            canvas.callbacks.process(name, event)

        dispatch("button_press_event", (4.0, 4.0))
        dispatch("motion_notify_event", (6.0, 5.0))
        dispatch("button_release_event", (6.0, 5.0))
        callbacks["Skip"](None)

    monkeypatch.setattr(module, "Button", FakeButton)
    monkeypatch.setattr(module, "_ensure_interactive_backend", lambda: None)
    monkeypatch.setattr(module.plt, "show", lambda **_kwargs: None)
    monkeypatch.setattr(FigureCanvasBase, "start_event_loop", drag_center_then_skip)
    study = StudyData(
        mr=np.ones((1, 9, 9), dtype=float),
        territory_mask=_closed_ring_stack(1, (9, 9)),
        slice_positions=np.asarray([0.0]),
    )
    output = tmp_path / "landmarks.json"

    selected = module.select_landmarks_interactively(
        study, output, resume_existing=False
    )

    assert selected == [None]
    assert load_center_overrides_json(
        output, 1, expected_image_shape=study.mr.shape
    ) == [(6.0, 5.0)]


def test_fresh_annotation_ignores_existing_landmarks_and_deleted_slices(
    monkeypatch, tmp_path
):
    from cardiac_bullseye.territory import landmarks as module

    callbacks = {}

    class FakeButton:
        def __init__(self, _axis, label):
            self.label = label

        def on_clicked(self, callback):
            callbacks[self.label] = callback

    def click_delete(_canvas, timeout=0):
        callbacks["Delete"](None)

    monkeypatch.setattr(module, "Button", FakeButton)
    monkeypatch.setattr(module, "_ensure_interactive_backend", lambda: None)
    monkeypatch.setattr(module.plt, "show", lambda **_kwargs: None)
    monkeypatch.setattr(FigureCanvasBase, "start_event_loop", click_delete)
    study = StudyData(
        mr=np.ones((2, 9, 9), dtype=float),
        territory_mask=_closed_ring_stack(2, (9, 9)),
        slice_positions=np.asarray([0.0, 1.0]),
    )
    output = tmp_path / "landmarks.json"
    pair = LandmarkPair(aip=(2.0, 3.0), iip=(4.0, 5.0))
    save_landmarks_json(output, [pair, None], deleted_slice_indices={1})

    selected = module.select_landmarks_interactively(
        study, output, resume_existing=False
    )

    assert selected == [None, None]
    assert load_deleted_slice_indices(output, 2) == {0, 1}


def test_cancel_is_error_preserves_progress_file(monkeypatch, tmp_path):
    from cardiac_bullseye.territory import landmarks as module

    callbacks = {}

    class FakeButton:
        def __init__(self, _axis, label):
            self.label = label

        def on_clicked(self, callback):
            callbacks[self.label] = callback

    def click_quit(_canvas, timeout=0):
        callbacks["Save && Quit"](None)

    monkeypatch.setattr(module, "Button", FakeButton)
    monkeypatch.setattr(module, "_ensure_interactive_backend", lambda: None)
    monkeypatch.setattr(module.plt, "show", lambda **_kwargs: None)
    monkeypatch.setattr(FigureCanvasBase, "start_event_loop", click_quit)
    study = StudyData(
        mr=np.ones((1, 9, 9), dtype=float),
        territory_mask=_closed_ring_stack(1, (9, 9)),
        slice_positions=np.asarray([0.0]),
    )
    output = tmp_path / "landmarks.json"

    with pytest.raises(RuntimeError, match=r"^AIP/IIP annotation was cancelled$"):
        module.select_landmarks_interactively(
            study,
            output,
            resume_existing=False,
            cancel_is_error=True,
        )

    assert output.exists()


def test_cancel_defaults_to_returning_landmarks(monkeypatch, tmp_path):
    from cardiac_bullseye.territory import landmarks as module

    callbacks = {}

    class FakeButton:
        def __init__(self, _axis, label):
            self.label = label

        def on_clicked(self, callback):
            callbacks[self.label] = callback

    def click_quit(_canvas, timeout=0):
        callbacks["Save && Quit"](None)

    monkeypatch.setattr(module, "Button", FakeButton)
    monkeypatch.setattr(module, "_ensure_interactive_backend", lambda: None)
    monkeypatch.setattr(module.plt, "show", lambda **_kwargs: None)
    monkeypatch.setattr(FigureCanvasBase, "start_event_loop", click_quit)
    study = StudyData(
        mr=np.ones((1, 9, 9), dtype=float),
        territory_mask=_closed_ring_stack(1, (9, 9)),
        slice_positions=np.asarray([0.0]),
    )
    output = tmp_path / "landmarks.json"

    assert module.select_landmarks_interactively(
        study, output, resume_existing=False
    ) == [None]
    assert output.exists()


def test_landmark_gui_zoom_preserves_view_and_selected_image_coordinates(
    monkeypatch, tmp_path
):
    from cardiac_bullseye.territory import landmarks as module

    callbacks = {}

    class FakeButton:
        def __init__(self, _axis, label):
            self.label = label

        def on_clicked(self, callback):
            callbacks[self.label] = callback

    def zoom_select_and_accept(canvas, timeout=0):
        axis = canvas.figure.axes[0]
        full_x_span = abs(axis.get_xlim()[1] - axis.get_xlim()[0])

        def dispatch(name, point, button, *, step=0):
            display_x, display_y = axis.transData.transform(point)
            event = MouseEvent(
                name,
                canvas,
                display_x,
                display_y,
                button=button,
                step=step,
            )
            canvas.callbacks.process(name, event)

        dispatch("scroll_event", (10.0, 10.0), "up", step=1)
        zoomed_x_span = abs(axis.get_xlim()[1] - axis.get_xlim()[0])
        assert zoomed_x_span < full_x_span

        callbacks["Fit"](None)
        assert abs(axis.get_xlim()[1] - axis.get_xlim()[0]) == pytest.approx(
            full_x_span
        )
        dispatch("scroll_event", (10.0, 10.0), "up", step=1)
        zoomed_x_span = abs(axis.get_xlim()[1] - axis.get_xlim()[0])

        dispatch("button_press_event", (6.25, 5.5), MouseButton.LEFT)
        assert abs(axis.get_xlim()[1] - axis.get_xlim()[0]) == pytest.approx(
            zoomed_x_span
        )
        dispatch("button_press_event", (13.75, 14.5), MouseButton.LEFT)
        callbacks["Accept"](None)

    monkeypatch.setattr(module, "Button", FakeButton)
    monkeypatch.setattr(module, "_ensure_interactive_backend", lambda: None)
    monkeypatch.setattr(module.plt, "show", lambda **_kwargs: None)
    monkeypatch.setattr(FigureCanvasBase, "start_event_loop", zoom_select_and_accept)
    study = StudyData(
        mr=np.ones((1, 21, 21), dtype=float),
        territory_mask=_closed_ring_stack(1, (21, 21)),
        slice_positions=np.asarray([0.0]),
    )

    selected = module.select_landmarks_interactively(
        study, tmp_path / "landmarks.json", resume_existing=False
    )

    assert selected[0] is not None
    assert selected[0].aip == pytest.approx((6.25, 5.5))
    assert selected[0].iip == pytest.approx((13.75, 14.5))


def test_transposed_landmark_view_matches_nifti_axes_and_maps_clicks_back(
    monkeypatch, tmp_path
):
    from cardiac_bullseye.territory import landmarks as module

    callbacks = {}

    class FakeButton:
        def __init__(self, _axis, label):
            self.label = label

        def on_clicked(self, callback):
            callbacks[self.label] = callback

    def inspect_select_and_accept(canvas, timeout=0):
        axis = canvas.figure.axes[0]
        np.testing.assert_array_equal(
            axis.images[0].get_array(),
            [
                [0.0, 6.0, 12.0, 18.0],
                [1.0, 7.0, 13.0, 19.0],
                [2.0, 8.0, 14.0, 20.0],
                [3.0, 9.0, 15.0, 21.0],
                [4.0, 10.0, 16.0, 22.0],
                [5.0, 11.0, 17.0, 23.0],
            ],
        )
        assert axis.get_xlim() == pytest.approx((-0.5, 3.5))
        assert axis.get_ylim() == pytest.approx((5.5, -0.5))
        center_marker = next(line for line in axis.lines if line.get_marker() == "+")
        assert tuple(center_marker.get_xdata()) == pytest.approx((1.0,))
        assert tuple(center_marker.get_ydata()) == pytest.approx((4.0,))

        def dispatch(point):
            display_x, display_y = axis.transData.transform(point)
            event = MouseEvent(
                "button_press_event",
                canvas,
                display_x,
                display_y,
                button=MouseButton.LEFT,
            )
            canvas.callbacks.process("button_press_event", event)

        dispatch((0.75, 4.25))
        dispatch((2.5, 1.25))
        callbacks["Accept"](None)

    monkeypatch.setattr(module, "Button", FakeButton)
    monkeypatch.setattr(module, "_ensure_interactive_backend", lambda: None)
    monkeypatch.setattr(module.plt, "show", lambda **_kwargs: None)
    monkeypatch.setattr(FigureCanvasBase, "start_event_loop", inspect_select_and_accept)
    myocardium = _closed_ring_stack(1, (4, 6), center_xy=(4, 1))
    study = StudyData(
        mr=np.arange(24, dtype=float).reshape(1, 4, 6),
        territory_mask=myocardium,
        slice_positions=np.asarray([0.0]),
    )

    selected = module.select_landmarks_interactively(
        study,
        tmp_path / "landmarks.json",
        resume_existing=False,
        transpose_for_display=True,
    )

    assert selected[0] is not None
    assert selected[0].aip == pytest.approx((4.25, 0.75))
    assert selected[0].iip == pytest.approx((1.25, 2.5))


def test_single_reference_gui_selects_once_and_propagates_to_all_slices(
    monkeypatch, tmp_path
):
    from cardiac_bullseye.territory import landmarks as module

    callbacks = {}

    class FakeButton:
        def __init__(self, _axis, label):
            self.label = label

        def on_clicked(self, callback):
            callbacks[self.label] = callback

    def select_reference(canvas, timeout=0):
        axis = canvas.figure.axes[0]
        assert "Slice 2/3" in axis.get_title()

        def dispatch(point):
            display_x, display_y = axis.transData.transform(point)
            event = MouseEvent(
                "button_press_event",
                canvas,
                display_x,
                display_y,
                button=MouseButton.LEFT,
            )
            canvas.callbacks.process("button_press_event", event)

        dispatch((7.0, 4.0))
        dispatch((4.0, 8.0))
        callbacks["Accept Reference"](None)

    monkeypatch.setattr(module, "Button", FakeButton)
    monkeypatch.setattr(module, "_ensure_interactive_backend", lambda: None)
    monkeypatch.setattr(module.plt, "show", lambda **_kwargs: None)
    monkeypatch.setattr(FigureCanvasBase, "start_event_loop", select_reference)
    study = StudyData(
        mr=np.ones((3, 9, 9), dtype=float),
        territory_mask=_closed_ring_stack(3, (9, 9)),
        slice_positions=np.asarray([0.0, 1.0, 2.0]),
    )

    selected = module.select_single_reference_interactively(
        study,
        tmp_path / "landmarks.json",
        resume_existing=False,
    )

    assert all(pair is not None for pair in selected)
    assert selected[1].aip == pytest.approx((7.0, 4.0))
    assert selected[1].iip == pytest.approx((4.0, 8.0))
    for pair in selected:
        assert pair.aip == pytest.approx((7.0, 4.0))
        assert pair.iip == pytest.approx((4.0, 8.0))

    payload = json.loads(
        (tmp_path / "landmarks.json").read_text(encoding="utf-8")
    )
    assert payload["annotation_mode"] == "single-reference"
    assert payload["reference_slice_index"] == 1


def test_single_reference_gui_can_delete_a_slice_before_accepting(
    monkeypatch, tmp_path
):
    from cardiac_bullseye.territory import landmarks as module

    callbacks = {}

    class FakeButton:
        def __init__(self, _axis, label):
            self.label = label

        def on_clicked(self, callback):
            callbacks[self.label] = callback

    def delete_then_select(canvas, timeout=0):
        axis = canvas.figure.axes[0]
        callbacks["Previous"](None)
        assert "Slice 1/3" in axis.get_title()
        callbacks["Delete"](None)
        callbacks["Next"](None)

        def dispatch(point):
            display_x, display_y = axis.transData.transform(point)
            event = MouseEvent(
                "button_press_event",
                canvas,
                display_x,
                display_y,
                button=MouseButton.LEFT,
            )
            canvas.callbacks.process("button_press_event", event)

        dispatch((7.0, 4.0))
        dispatch((4.0, 8.0))
        callbacks["Accept Reference"](None)

    monkeypatch.setattr(module, "Button", FakeButton)
    monkeypatch.setattr(module, "_ensure_interactive_backend", lambda: None)
    monkeypatch.setattr(module.plt, "show", lambda **_kwargs: None)
    monkeypatch.setattr(FigureCanvasBase, "start_event_loop", delete_then_select)
    study = StudyData(
        mr=np.ones((3, 9, 9), dtype=float),
        territory_mask=_closed_ring_stack(3, (9, 9)),
        slice_positions=np.asarray([0.0, 1.0, 2.0]),
    )
    output = tmp_path / "landmarks.json"

    selected = module.select_single_reference_interactively(
        study, output, resume_existing=False
    )

    assert selected[0] is None
    assert selected[1] is not None
    assert selected[2] is not None
    assert load_deleted_slice_indices(output, 3) == {0}
