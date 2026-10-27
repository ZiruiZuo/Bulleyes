from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import Button

from .geometry import estimate_lv_centers
from .interaction import zoom_limits_about_point
from .io import (
    load_center_overrides_json,
    load_deleted_slice_indices,
    load_landmarks_json,
    save_landmarks_json,
)
from .models import LandmarkPair, StudyData


def propagate_reference_landmarks(
    study: StudyData,
    reference_slice_index: int,
    reference_pair: LandmarkPair,
    *,
    deleted_slice_indices: set[int] | None = None,
    center_overrides: list[tuple[float, float] | None] | None = None,
) -> list[LandmarkPair | None]:
    """Apply one slice's AIP/IIP directions around every slice's LV center."""

    if not 0 <= reference_slice_index < study.slice_count:
        raise ValueError("reference_slice_index is outside the short-axis stack")
    deleted = set() if deleted_slice_indices is None else set(deleted_slice_indices)
    if reference_slice_index in deleted:
        raise ValueError("The reference slice cannot be deleted")

    myocardium_masks = study.territory_mask > 0
    if not myocardium_masks[reference_slice_index].any():
        raise ValueError("The reference slice must contain myocardium")
    centers = estimate_lv_centers(
        myocardium_masks,
        center_overrides=center_overrides,
    )
    reference_center = centers[reference_slice_index]
    aip_offset = np.asarray(reference_pair.aip, dtype=float) - reference_center
    iip_offset = np.asarray(reference_pair.iip, dtype=float) - reference_center
    aip_angle = float(np.arctan2(aip_offset[1], aip_offset[0]))
    iip_angle = float(np.arctan2(iip_offset[1], iip_offset[0]))
    angle_difference = float(
        np.arctan2(np.sin(iip_angle - aip_angle), np.cos(iip_angle - aip_angle))
    )
    if (
        np.linalg.norm(aip_offset) <= np.finfo(float).eps
        or np.linalg.norm(iip_offset) <= np.finfo(float).eps
        or np.isclose(angle_difference, 0.0)
    ):
        raise ValueError("AIP and IIP must define two distinct directions")

    propagated: list[LandmarkPair | None] = []
    for slice_index, center in enumerate(centers):
        if slice_index in deleted or not myocardium_masks[slice_index].any():
            propagated.append(None)
            continue
        if slice_index == reference_slice_index:
            propagated.append(reference_pair)
            continue
        propagated.append(
            LandmarkPair(
                aip=tuple((center + aip_offset).tolist()),
                iip=tuple((center + iip_offset).tolist()),
            )
        )
    return propagated


def annotation_display_transform(
    image: np.ndarray, *, transpose: bool = False
) -> np.ndarray:
    """Transform a stored short-axis plane for annotation display only."""

    array = np.asarray(image)
    return array.T if transpose else array


def original_xy_to_display_xy(
    point_xy: tuple[float, float],
    *,
    transpose: bool = False,
) -> tuple[float, float]:
    """Map an original-image point into the annotation view."""

    original_x, original_y = point_xy
    if transpose:
        return float(original_y), float(original_x)
    return float(original_x), float(original_y)


def display_xy_to_original_xy(
    point_xy: tuple[float, float],
    *,
    transpose: bool = False,
) -> tuple[float, float]:
    """Map an annotation-view click back to original image pixels."""

    display_x, display_y = point_xy
    if transpose:
        return float(display_y), float(display_x)
    return float(display_x), float(display_y)


def _ensure_interactive_backend() -> None:
    """Move figures out of an IDE Plots pane into an interactive GUI window."""

    backend = matplotlib.get_backend().lower()
    non_interactive = (
        "inline" in backend or backend.endswith("agg") or "matplotlib_inline" in backend
    )
    if not non_interactive:
        return

    errors: list[str] = []
    for candidate in ("QtAgg", "TkAgg"):
        try:
            plt.switch_backend(candidate)
            return
        except (ImportError, RuntimeError) as error:
            errors.append(f"{candidate}: {error}")

    raise RuntimeError(
        "AIP/IIP annotation requires an interactive Matplotlib window, but "
        "neither QtAgg nor TkAgg could be enabled. In Spyder, set Preferences "
        "> IPython console > Graphics > Backend to Qt, restart the console, "
        f"and run main.py again. Backend errors: {'; '.join(errors)}"
    )


def select_landmarks_interactively(
    study: StudyData,
    output_path: str | Path,
    *,
    revisit_completed: bool = False,
    resume_existing: bool = True,
    transpose_for_display: bool = False,
    cancel_is_error: bool = False,
) -> list[LandmarkPair | None]:
    """Select AIP followed by IIP in a standalone interactive window.

    All actions use buttons inside the figure. Progress is saved after every
    slice. Set ``resume_existing=False`` to start with blank annotations even
    when ``output_path`` already exists.
    """

    centers = estimate_lv_centers(study.territory_mask > 0)
    _ensure_interactive_backend()
    output_path = Path(output_path)
    if resume_existing:
        landmarks = load_landmarks_json(
            output_path,
            study.slice_count,
            expected_image_shape=study.mr.shape,
        )
        deleted_slice_indices = load_deleted_slice_indices(
            output_path,
            study.slice_count,
        )
        center_overrides = load_center_overrides_json(
            output_path,
            study.slice_count,
            expected_image_shape=study.mr.shape,
        )
    else:
        landmarks = [None] * study.slice_count
        deleted_slice_indices = set()
        center_overrides = [None] * study.slice_count
    display_shape = (
        study.mr.shape[1:][::-1] if transpose_for_display else study.mr.shape[1:]
    )

    figure, axis = plt.subplots(figsize=(8, 8))
    figure.subplots_adjust(bottom=0.16)
    figure.canvas.manager.set_window_title("AIP / IIP annotation")

    accept_axis = figure.add_axes((0.035, 0.035, 0.12, 0.055))
    redo_axis = figure.add_axes((0.17, 0.035, 0.12, 0.055))
    skip_axis = figure.add_axes((0.305, 0.035, 0.12, 0.055))
    delete_axis = figure.add_axes((0.44, 0.035, 0.12, 0.055))
    fit_axis = figure.add_axes((0.575, 0.035, 0.12, 0.055))
    quit_axis = figure.add_axes((0.71, 0.035, 0.25, 0.055))
    accept_button = Button(accept_axis, "Accept")
    redo_button = Button(redo_axis, "Redo")
    skip_button = Button(skip_axis, "Skip")
    delete_button = Button(delete_axis, "Delete")
    fit_button = Button(fit_axis, "Fit")
    quit_button = Button(quit_axis, "Save && Quit")

    state: dict[str, object] = {
        "points": [],
        "action": None,
        "slice_index": 0,
        "pan_start": None,
        "dragging_center": False,
        "closed": False,
    }

    def full_view() -> None:
        height, width = display_shape
        axis.set_xlim(-0.5, width - 0.5)
        axis.set_ylim(height - 0.5, -0.5)

    def draw_current_slice(
        message: str | None = None, *, preserve_view: bool = True
    ) -> None:
        slice_index = int(state["slice_index"])
        original_image = study.mr[slice_index]
        original_myocardium = study.territory_mask[slice_index] > 0
        image = annotation_display_transform(
            original_image, transpose=transpose_for_display
        )
        myocardium = annotation_display_transform(
            original_myocardium, transpose=transpose_for_display
        )
        points = state["points"]
        assert isinstance(points, list)

        old_xlim = axis.get_xlim() if preserve_view and axis.has_data() else None
        old_ylim = axis.get_ylim() if preserve_view and axis.has_data() else None
        axis.clear()
        finite = image[np.isfinite(image)]
        if finite.size:
            vmin, vmax = np.percentile(finite, [1, 99])
        else:
            vmin, vmax = 0.0, 1.0
        axis.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
        if original_myocardium.any():
            axis.contour(
                myocardium,
                levels=[0.5],
                colors=["yellow"],
                linewidths=1.0,
            )
            center_original = center_overrides[slice_index] or tuple(
                centers[slice_index]
            )
            center_x, center_y = original_xy_to_display_xy(
                center_original, transpose=transpose_for_display
            )
            axis.plot(
                center_x,
                center_y,
                marker="+",
                color="#39ff88",
                markersize=12,
                markeredgewidth=2,
            )

        colors = ("red", "cyan")
        labels = ("AIP", "IIP")
        for point_index, point in enumerate(points):
            display_point = original_xy_to_display_xy(
                point, transpose=transpose_for_display
            )
            axis.plot(
                display_point[0],
                display_point[1],
                "o",
                color=colors[point_index],
                markeredgecolor="black",
                markersize=9,
            )
            axis.text(
                display_point[0] + 3,
                display_point[1],
                labels[point_index],
                color=colors[point_index],
                fontsize=10,
                weight="bold",
            )
        if len(points) == 2:
            first_display = original_xy_to_display_xy(
                points[0], transpose=transpose_for_display
            )
            second_display = original_xy_to_display_xy(
                points[1], transpose=transpose_for_display
            )
            axis.plot(
                [first_display[0], second_display[0]],
                [first_display[1], second_display[1]],
                "w--",
                linewidth=1,
            )

        instruction = (
            "Left click: AIP then IIP | Drag green +: adjust center | "
            "Right click: undo | "
            "Wheel: zoom | Middle drag: pan | Skip: interpolate"
            if message is None
            else message
        )
        axis.set_title(f"Slice {slice_index + 1}/{study.slice_count}\n{instruction}")
        if old_xlim is None or old_ylim is None:
            full_view()
        else:
            axis.set_xlim(old_xlim)
            axis.set_ylim(old_ylim)
        axis.set_axis_off()
        figure.canvas.draw_idle()

    def stop_with_action(action: str) -> None:
        state["action"] = action
        figure.canvas.stop_event_loop()

    def on_accept(_event) -> None:
        points = state["points"]
        assert isinstance(points, list)
        if len(points) != 2:
            draw_current_slice("Select exactly two points before Accept")
            return
        stop_with_action("accept")

    def on_redo(_event) -> None:
        state["points"] = []
        center_overrides[int(state["slice_index"])] = None
        draw_current_slice()

    def on_skip(_event) -> None:
        stop_with_action("skip")

    def on_delete(_event) -> None:
        center_overrides[int(state["slice_index"])] = None
        stop_with_action("delete")

    def on_fit(_event) -> None:
        full_view()
        figure.canvas.draw_idle()

    def on_quit(_event) -> None:
        stop_with_action("quit")

    def on_close(_event) -> None:
        state["closed"] = True
        state["action"] = "quit"
        try:
            figure.canvas.stop_event_loop()
        except AttributeError:
            pass

    def on_mouse_press(event) -> None:
        if event.inaxes is not axis or event.xdata is None or event.ydata is None:
            return
        if event.button is MouseButton.MIDDLE:
            state["pan_start"] = (
                float(event.x),
                float(event.y),
                axis.get_xlim(),
                axis.get_ylim(),
            )
            return
        if event.button is MouseButton.LEFT:
            slice_index = int(state["slice_index"])
            if (study.territory_mask[slice_index] > 0).any():
                center_original = center_overrides[slice_index] or tuple(
                    centers[slice_index]
                )
                center_display = original_xy_to_display_xy(
                    center_original,
                    transpose=transpose_for_display,
                )
                center_canvas = axis.transData.transform(center_display)
                if np.hypot(
                    float(event.x) - float(center_canvas[0]),
                    float(event.y) - float(center_canvas[1]),
                ) <= 12.0:
                    state["dragging_center"] = True
                    return
        points = state["points"]
        assert isinstance(points, list)
        if event.button is MouseButton.RIGHT:
            if points:
                points.pop()
                draw_current_slice()
            return
        if event.button is not MouseButton.LEFT:
            return
        if len(points) == 2:
            points.clear()
        points.append(
            display_xy_to_original_xy(
                (float(event.xdata), float(event.ydata)),
                transpose=transpose_for_display,
            )
        )
        draw_current_slice()

    def on_mouse_motion(event) -> None:
        if state["dragging_center"]:
            if event.inaxes is axis and event.xdata is not None and event.ydata is not None:
                height, width = display_shape
                display_point = (
                    float(np.clip(event.xdata, 0.0, width - 1.0)),
                    float(np.clip(event.ydata, 0.0, height - 1.0)),
                )
                center_overrides[int(state["slice_index"])] = (
                    display_xy_to_original_xy(
                        display_point,
                        transpose=transpose_for_display,
                    )
                )
                draw_current_slice()
            return
        pan_start = state["pan_start"]
        if pan_start is None:
            return
        start_x, start_y, start_xlim, start_ylim = pan_start
        x_shift = (
            -(float(event.x) - start_x)
            * (start_xlim[1] - start_xlim[0])
            / axis.bbox.width
        )
        y_shift = (
            -(float(event.y) - start_y)
            * (start_ylim[1] - start_ylim[0])
            / axis.bbox.height
        )
        axis.set_xlim(start_xlim[0] + x_shift, start_xlim[1] + x_shift)
        axis.set_ylim(start_ylim[0] + y_shift, start_ylim[1] + y_shift)
        figure.canvas.draw_idle()

    def on_mouse_release(event) -> None:
        if event.button is MouseButton.LEFT:
            state["dragging_center"] = False
        if event.button is MouseButton.MIDDLE:
            state["pan_start"] = None

    def on_scroll(event) -> None:
        if event.inaxes is not axis or event.xdata is None or event.ydata is None:
            return
        scale = 0.8 if event.button == "up" else 1.25
        xlim, ylim = zoom_limits_about_point(
            axis.get_xlim(),
            axis.get_ylim(),
            (float(event.xdata), float(event.ydata)),
            scale=scale,
            image_shape=display_shape,
        )
        axis.set_xlim(xlim)
        axis.set_ylim(ylim)
        figure.canvas.draw_idle()

    accept_button.on_clicked(on_accept)
    redo_button.on_clicked(on_redo)
    skip_button.on_clicked(on_skip)
    delete_button.on_clicked(on_delete)
    fit_button.on_clicked(on_fit)
    quit_button.on_clicked(on_quit)
    figure.canvas.mpl_connect("button_press_event", on_mouse_press)
    figure.canvas.mpl_connect("motion_notify_event", on_mouse_motion)
    figure.canvas.mpl_connect("button_release_event", on_mouse_release)
    figure.canvas.mpl_connect("scroll_event", on_scroll)
    figure.canvas.mpl_connect("close_event", on_close)
    plt.show(block=False)

    for slice_index in range(study.slice_count):
        existing = landmarks[slice_index]
        if (
            existing is not None or slice_index in deleted_slice_indices
        ) and not revisit_completed:
            continue

        state["slice_index"] = slice_index
        state["action"] = None
        if existing is None:
            state["points"] = []
        else:
            state["points"] = [existing.aip, existing.iip]
        state["pan_start"] = None
        state["dragging_center"] = False
        draw_current_slice(preserve_view=False)
        figure.canvas.start_event_loop(timeout=-1)

        action = state["action"]
        if action == "quit" or state["closed"]:
            save_landmarks_json(
                output_path,
                landmarks,
                deleted_slice_indices=deleted_slice_indices,
                image_shape=study.mr.shape,
                center_overrides=center_overrides,
            )
            if not state["closed"]:
                plt.close(figure)
            if cancel_is_error:
                raise RuntimeError("AIP/IIP annotation was cancelled")
            return landmarks
        if action == "skip":
            landmarks[slice_index] = None
            deleted_slice_indices.discard(slice_index)
        elif action == "delete":
            landmarks[slice_index] = None
            deleted_slice_indices.add(slice_index)
        elif action == "accept":
            points = state["points"]
            assert isinstance(points, list) and len(points) == 2
            landmarks[slice_index] = LandmarkPair(
                aip=tuple(points[0]),
                iip=tuple(points[1]),
            )
            deleted_slice_indices.discard(slice_index)
        save_landmarks_json(
            output_path,
            landmarks,
            deleted_slice_indices=deleted_slice_indices,
            image_shape=study.mr.shape,
            center_overrides=center_overrides,
        )

    if not state["closed"]:
        plt.close(figure)
    return landmarks


def select_single_reference_interactively(
    study: StudyData,
    output_path: str | Path,
    *,
    resume_existing: bool = True,
    transpose_for_display: bool = False,
    cancel_is_error: bool = False,
) -> list[LandmarkPair | None]:
    """Select one AIP/IIP pair and propagate its directions to every slice."""

    centers = estimate_lv_centers(study.territory_mask > 0)
    _ensure_interactive_backend()
    output_path = Path(output_path)
    if resume_existing:
        deleted_slice_indices = load_deleted_slice_indices(
            output_path, study.slice_count
        )
        center_overrides = load_center_overrides_json(
            output_path,
            study.slice_count,
            expected_image_shape=study.mr.shape,
        )
    else:
        deleted_slice_indices = set()
        center_overrides = [None] * study.slice_count

    nonempty_indices = [
        index
        for index in range(study.slice_count)
        if (study.territory_mask[index] > 0).any()
    ]
    if not nonempty_indices:
        raise ValueError("No short-axis slice contains myocardium")
    initial_index = nonempty_indices[len(nonempty_indices) // 2]
    display_shape = (
        study.mr.shape[1:][::-1] if transpose_for_display else study.mr.shape[1:]
    )

    figure, axis = plt.subplots(figsize=(10, 8))
    figure.subplots_adjust(bottom=0.16)
    figure.canvas.manager.set_window_title("Single-reference AIP / IIP annotation")

    button_specs = (
        ("Accept Reference", 0.02, 0.18),
        ("Redo", 0.21, 0.09),
        ("Previous", 0.31, 0.11),
        ("Next", 0.43, 0.09),
        ("Delete", 0.53, 0.10),
        ("Restore", 0.64, 0.10),
        ("Fit", 0.75, 0.09),
        ("Cancel", 0.85, 0.12),
    )
    buttons = {}
    for label, left, width in button_specs:
        button_axis = figure.add_axes((left, 0.035, width, 0.055))
        buttons[label] = Button(button_axis, label)

    state: dict[str, object] = {
        "points": [],
        "slice_index": initial_index,
        "pan_start": None,
        "dragging_center": False,
        "accepted": False,
        "closed": False,
    }

    def full_view() -> None:
        height, width = display_shape
        axis.set_xlim(-0.5, width - 0.5)
        axis.set_ylim(height - 0.5, -0.5)

    def draw_current_slice(
        message: str | None = None, *, preserve_view: bool = True
    ) -> None:
        slice_index = int(state["slice_index"])
        original_image = study.mr[slice_index]
        original_myocardium = study.territory_mask[slice_index] > 0
        image = annotation_display_transform(
            original_image, transpose=transpose_for_display
        )
        myocardium = annotation_display_transform(
            original_myocardium, transpose=transpose_for_display
        )
        points = state["points"]
        assert isinstance(points, list)

        old_xlim = axis.get_xlim() if preserve_view and axis.has_data() else None
        old_ylim = axis.get_ylim() if preserve_view and axis.has_data() else None
        axis.clear()
        finite = image[np.isfinite(image)]
        vmin, vmax = np.percentile(finite, [1, 99]) if finite.size else (0.0, 1.0)
        axis.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
        if original_myocardium.any():
            axis.contour(myocardium, levels=[0.5], colors=["yellow"], linewidths=1.0)
            center_original = center_overrides[slice_index] or tuple(
                centers[slice_index]
            )
            center_display = original_xy_to_display_xy(
                center_original, transpose=transpose_for_display
            )
            axis.plot(
                center_display[0],
                center_display[1],
                marker="+",
                color="#39ff88",
                markersize=12,
                markeredgewidth=2,
            )

        for point_index, point in enumerate(points):
            display_point = original_xy_to_display_xy(
                point, transpose=transpose_for_display
            )
            color = ("red", "cyan")[point_index]
            label = ("AIP", "IIP")[point_index]
            axis.plot(
                display_point[0],
                display_point[1],
                "o",
                color=color,
                markeredgecolor="black",
                markersize=9,
            )
            axis.text(
                display_point[0] + 3,
                display_point[1],
                label,
                color=color,
                fontsize=10,
                weight="bold",
            )

        status = "DELETED" if slice_index in deleted_slice_indices else "retained"
        instruction = message or (
            "Choose one reference slice; left click AIP then IIP | "
            "Drag green +: adjust center | Wheel: zoom | Middle drag: pan"
        )
        axis.set_title(
            f"Slice {slice_index + 1}/{study.slice_count} [{status}]\n{instruction}"
        )
        if old_xlim is None or old_ylim is None:
            full_view()
        else:
            axis.set_xlim(old_xlim)
            axis.set_ylim(old_ylim)
        axis.set_axis_off()
        figure.canvas.draw_idle()

    def navigate(step: int) -> None:
        new_index = int(np.clip(
            int(state["slice_index"]) + step, 0, study.slice_count - 1
        ))
        if new_index != int(state["slice_index"]):
            state["slice_index"] = new_index
            state["points"] = []
            state["pan_start"] = None
            state["dragging_center"] = False
            draw_current_slice(preserve_view=False)

    def on_accept(_event) -> None:
        slice_index = int(state["slice_index"])
        points = state["points"]
        assert isinstance(points, list)
        if slice_index in deleted_slice_indices:
            draw_current_slice("Restore this slice before using it as reference")
            return
        if not (study.territory_mask[slice_index] > 0).any():
            draw_current_slice("Reference slice must contain myocardium")
            return
        if len(points) != 2:
            draw_current_slice("Select exactly two points before Accept Reference")
            return
        state["accepted"] = True
        figure.canvas.stop_event_loop()

    def on_redo(_event) -> None:
        state["points"] = []
        center_overrides[int(state["slice_index"])] = None
        draw_current_slice()

    def on_delete(_event) -> None:
        slice_index = int(state["slice_index"])
        deleted_slice_indices.add(slice_index)
        state["points"] = []
        draw_current_slice()

    def on_restore(_event) -> None:
        deleted_slice_indices.discard(int(state["slice_index"]))
        draw_current_slice()

    def on_cancel(_event) -> None:
        figure.canvas.stop_event_loop()

    def on_close(_event) -> None:
        state["closed"] = True
        try:
            figure.canvas.stop_event_loop()
        except AttributeError:
            pass

    def on_mouse_press(event) -> None:
        if event.inaxes is not axis or event.xdata is None or event.ydata is None:
            return
        if event.button is MouseButton.MIDDLE:
            state["pan_start"] = (
                float(event.x),
                float(event.y),
                axis.get_xlim(),
                axis.get_ylim(),
            )
            return
        points = state["points"]
        assert isinstance(points, list)
        if event.button is MouseButton.RIGHT:
            if points:
                points.pop()
                draw_current_slice()
            return
        if event.button is not MouseButton.LEFT:
            return

        slice_index = int(state["slice_index"])
        if (study.territory_mask[slice_index] > 0).any():
            center_original = center_overrides[slice_index] or tuple(
                centers[slice_index]
            )
            center_display = original_xy_to_display_xy(
                center_original, transpose=transpose_for_display
            )
            center_canvas = axis.transData.transform(center_display)
            if np.hypot(
                float(event.x) - float(center_canvas[0]),
                float(event.y) - float(center_canvas[1]),
            ) <= 12.0:
                state["dragging_center"] = True
                return
        if len(points) == 2:
            points.clear()
        points.append(
            display_xy_to_original_xy(
                (float(event.xdata), float(event.ydata)),
                transpose=transpose_for_display,
            )
        )
        draw_current_slice()

    def on_mouse_motion(event) -> None:
        if state["dragging_center"]:
            if event.inaxes is axis and event.xdata is not None and event.ydata is not None:
                height, width = display_shape
                display_point = (
                    float(np.clip(event.xdata, 0.0, width - 1.0)),
                    float(np.clip(event.ydata, 0.0, height - 1.0)),
                )
                center_overrides[int(state["slice_index"])] = (
                    display_xy_to_original_xy(
                        display_point, transpose=transpose_for_display
                    )
                )
                draw_current_slice()
            return
        pan_start = state["pan_start"]
        if pan_start is None:
            return
        start_x, start_y, start_xlim, start_ylim = pan_start
        x_shift = (
            -(float(event.x) - start_x)
            * (start_xlim[1] - start_xlim[0])
            / axis.bbox.width
        )
        y_shift = (
            -(float(event.y) - start_y)
            * (start_ylim[1] - start_ylim[0])
            / axis.bbox.height
        )
        axis.set_xlim(start_xlim[0] + x_shift, start_xlim[1] + x_shift)
        axis.set_ylim(start_ylim[0] + y_shift, start_ylim[1] + y_shift)
        figure.canvas.draw_idle()

    def on_mouse_release(event) -> None:
        if event.button is MouseButton.LEFT:
            state["dragging_center"] = False
        if event.button is MouseButton.MIDDLE:
            state["pan_start"] = None

    def on_scroll(event) -> None:
        if event.inaxes is not axis or event.xdata is None or event.ydata is None:
            return
        scale = 0.8 if event.button == "up" else 1.25
        xlim, ylim = zoom_limits_about_point(
            axis.get_xlim(),
            axis.get_ylim(),
            (float(event.xdata), float(event.ydata)),
            scale=scale,
            image_shape=display_shape,
        )
        axis.set_xlim(xlim)
        axis.set_ylim(ylim)
        figure.canvas.draw_idle()

    buttons["Accept Reference"].on_clicked(on_accept)
    buttons["Redo"].on_clicked(on_redo)
    buttons["Previous"].on_clicked(lambda _event: navigate(-1))
    buttons["Next"].on_clicked(lambda _event: navigate(1))
    buttons["Delete"].on_clicked(on_delete)
    buttons["Restore"].on_clicked(on_restore)
    buttons["Fit"].on_clicked(lambda _event: (full_view(), figure.canvas.draw_idle()))
    buttons["Cancel"].on_clicked(on_cancel)
    figure.canvas.mpl_connect("button_press_event", on_mouse_press)
    figure.canvas.mpl_connect("motion_notify_event", on_mouse_motion)
    figure.canvas.mpl_connect("button_release_event", on_mouse_release)
    figure.canvas.mpl_connect("scroll_event", on_scroll)
    figure.canvas.mpl_connect("close_event", on_close)
    plt.show(block=False)
    draw_current_slice(preserve_view=False)
    figure.canvas.start_event_loop(timeout=-1)

    if not state["accepted"]:
        empty = [None] * study.slice_count
        save_landmarks_json(
            output_path,
            empty,
            deleted_slice_indices=deleted_slice_indices,
            image_shape=study.mr.shape,
            center_overrides=center_overrides,
            annotation_mode="single-reference",
        )
        if not state["closed"]:
            plt.close(figure)
        if cancel_is_error:
            raise RuntimeError("AIP/IIP annotation was cancelled")
        return empty

    reference_slice_index = int(state["slice_index"])
    points = state["points"]
    assert isinstance(points, list) and len(points) == 2
    reference_pair = LandmarkPair(aip=tuple(points[0]), iip=tuple(points[1]))
    landmarks = propagate_reference_landmarks(
        study,
        reference_slice_index,
        reference_pair,
        deleted_slice_indices=deleted_slice_indices,
        center_overrides=center_overrides,
    )
    save_landmarks_json(
        output_path,
        landmarks,
        deleted_slice_indices=deleted_slice_indices,
        image_shape=study.mr.shape,
        center_overrides=center_overrides,
        annotation_mode="single-reference",
        reference_slice_index=reference_slice_index,
    )
    if not state["closed"]:
        plt.close(figure)
    return landmarks


def load_saved_landmarks(
    study: StudyData,
    input_path: str | Path,
) -> list[LandmarkPair | None]:
    """Load compatible saved AIP/IIP points without opening an annotation UI."""

    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(
            "reuse_saved_points=true requires an existing landmark file: "
            f"{input_path}"
        )
    return load_landmarks_json(
        input_path,
        study.slice_count,
        expected_image_shape=study.mr.shape,
    )
