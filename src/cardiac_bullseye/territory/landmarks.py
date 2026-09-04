from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import Button

from .geometry import estimate_lv_center
from .io import load_landmarks_json, save_landmarks_json
from .models import LandmarkPair, StudyData


def annotation_display_transform(image: np.ndarray) -> np.ndarray:
    """Rotate 90 degrees clockwise, then mirror left-to-right for display."""

    return np.fliplr(np.rot90(np.asarray(image), k=-1))


def original_xy_to_display_xy(
    point_xy: tuple[float, float],
) -> tuple[float, float]:
    """Map an original-image point into the transformed annotation view."""

    original_x, original_y = point_xy
    return float(original_y), float(original_x)


def display_xy_to_original_xy(
    point_xy: tuple[float, float],
) -> tuple[float, float]:
    """Map a clicked annotation-view point back to original NIfTI pixels."""

    display_x, display_y = point_xy
    return float(display_y), float(display_x)


def _ensure_interactive_backend() -> None:
    """Move figures out of an IDE Plots pane into an interactive GUI window."""

    backend = matplotlib.get_backend().lower()
    non_interactive = (
        "inline" in backend
        or backend.endswith("agg")
        or "matplotlib_inline" in backend
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
) -> list[LandmarkPair | None]:
    """Select AIP followed by IIP in a standalone interactive window.

    All actions use buttons inside the figure. Progress is saved after every
    slice, so closing the window is safe and a later run can resume.
    """

    _ensure_interactive_backend()
    output_path = Path(output_path)
    landmarks = load_landmarks_json(output_path, study.slice_count)

    figure, axis = plt.subplots(figsize=(8, 8))
    figure.subplots_adjust(bottom=0.14)
    figure.canvas.manager.set_window_title("AIP / IIP annotation")

    accept_axis = figure.add_axes((0.16, 0.035, 0.14, 0.055))
    redo_axis = figure.add_axes((0.32, 0.035, 0.14, 0.055))
    skip_axis = figure.add_axes((0.48, 0.035, 0.14, 0.055))
    quit_axis = figure.add_axes((0.64, 0.035, 0.18, 0.055))
    accept_button = Button(accept_axis, "Accept")
    redo_button = Button(redo_axis, "Redo")
    skip_button = Button(skip_axis, "Skip")
    quit_button = Button(quit_axis, "Save && Quit")

    state: dict[str, object] = {
        "points": [],
        "action": None,
        "slice_index": 0,
        "closed": False,
    }

    def draw_current_slice(message: str | None = None) -> None:
        slice_index = int(state["slice_index"])
        original_image = study.mr[slice_index]
        original_myocardium = study.territory_mask[slice_index] > 0
        image = annotation_display_transform(original_image)
        myocardium = annotation_display_transform(original_myocardium)
        points = state["points"]
        assert isinstance(points, list)

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
            try:
                center_original = estimate_lv_center(original_myocardium)
                center_x, center_y = original_xy_to_display_xy(center_original)
                axis.plot(
                    center_x,
                    center_y,
                    marker="+",
                    color="#39ff88",
                    markersize=12,
                    markeredgewidth=2,
                )
            except ValueError:
                pass

        colors = ("red", "cyan")
        labels = ("AIP", "IIP")
        for point_index, point in enumerate(points):
            display_point = original_xy_to_display_xy(point)
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
            first_display = original_xy_to_display_xy(points[0])
            second_display = original_xy_to_display_xy(points[1])
            axis.plot(
                [first_display[0], second_display[0]],
                [first_display[1], second_display[1]],
                "w--",
                linewidth=1,
            )

        instruction = (
            "Left click: AIP then IIP | Right click: undo last point"
            if message is None
            else message
        )
        axis.set_title(
            f"Slice {slice_index + 1}/{study.slice_count}\n{instruction}"
        )
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
        draw_current_slice()

    def on_skip(_event) -> None:
        stop_with_action("skip")

    def on_quit(_event) -> None:
        stop_with_action("quit")

    def on_close(_event) -> None:
        state["closed"] = True
        state["action"] = "quit"
        try:
            figure.canvas.stop_event_loop()
        except AttributeError:
            pass

    def on_mouse_click(event) -> None:
        if event.inaxes is not axis or event.xdata is None or event.ydata is None:
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
            display_xy_to_original_xy((float(event.xdata), float(event.ydata)))
        )
        draw_current_slice()

    accept_button.on_clicked(on_accept)
    redo_button.on_clicked(on_redo)
    skip_button.on_clicked(on_skip)
    quit_button.on_clicked(on_quit)
    figure.canvas.mpl_connect("button_press_event", on_mouse_click)
    figure.canvas.mpl_connect("close_event", on_close)
    plt.show(block=False)

    for slice_index in range(study.slice_count):
        existing = landmarks[slice_index]
        if existing is not None and not revisit_completed:
            continue

        state["slice_index"] = slice_index
        state["action"] = None
        if existing is None:
            state["points"] = []
        else:
            state["points"] = [existing.aip, existing.iip]
        draw_current_slice()
        figure.canvas.start_event_loop(timeout=-1)

        action = state["action"]
        if action == "quit" or state["closed"]:
            save_landmarks_json(output_path, landmarks)
            if not state["closed"]:
                plt.close(figure)
            return landmarks
        if action == "skip":
            landmarks[slice_index] = None
        elif action == "accept":
            points = state["points"]
            assert isinstance(points, list) and len(points) == 2
            landmarks[slice_index] = LandmarkPair(
                aip=tuple(points[0]),
                iip=tuple(points[1]),
            )
        save_landmarks_json(output_path, landmarks)

    if not state["closed"]:
        plt.close(figure)
    return landmarks
