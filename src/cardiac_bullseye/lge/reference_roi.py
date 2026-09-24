"""Persistence and rasterization for manually drawn normal-myocardium ROIs."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import Button, Slider
from matplotlib.path import Path as MatplotlibPath
from numpy.typing import NDArray

from ..territory.interaction import zoom_limits_about_point
from ..territory.landmarks import annotation_display_transform
from ..territory.models import LandmarkPair, StudyData
from .scar import compute_reference_statistics


ReferencePolygon = list[tuple[float, float]] | None
ReferenceMask = NDArray[np.bool_] | None


def apply_brush_stroke(
    mask: NDArray[np.bool_],
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    *,
    radius: float,
    value: bool,
) -> NDArray[np.bool_]:
    """Paint or erase a circular brush continuously along an image-space line."""

    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("Brush mask must be a two-dimensional boolean array")
    coordinates = np.asarray((*start_xy, *end_xy, radius), dtype=float)
    if not np.isfinite(coordinates).all() or radius <= 0.0:
        raise ValueError("Brush coordinates must be finite and radius must be positive")

    start = np.asarray(start_xy, dtype=float)
    end = np.asarray(end_xy, dtype=float)
    lower_x = max(0, int(np.floor(min(start[0], end[0]) - radius)))
    upper_x = min(mask.shape[1] - 1, int(np.ceil(max(start[0], end[0]) + radius)))
    lower_y = max(0, int(np.floor(min(start[1], end[1]) - radius)))
    upper_y = min(mask.shape[0] - 1, int(np.ceil(max(start[1], end[1]) + radius)))
    if lower_x > upper_x or lower_y > upper_y:
        return mask

    grid_y, grid_x = np.mgrid[lower_y : upper_y + 1, lower_x : upper_x + 1]
    pixels = np.stack((grid_x, grid_y), axis=-1).astype(float)
    direction = end - start
    squared_length = float(np.dot(direction, direction))
    if squared_length == 0.0:
        closest = np.broadcast_to(start, pixels.shape)
    else:
        fractions = np.sum((pixels - start) * direction, axis=-1) / squared_length
        fractions = np.clip(fractions, 0.0, 1.0)
        closest = start + fractions[..., None] * direction
    selected = np.sum((pixels - closest) ** 2, axis=-1) <= radius**2
    target = mask[lower_y : upper_y + 1, lower_x : upper_x + 1]
    target[selected] = bool(value)
    return mask


def _mask_to_row_runs(mask: NDArray[np.bool_]) -> list[list[int]]:
    runs: list[list[int]] = []
    for row_index, row in enumerate(mask):
        padded = np.pad(row.astype(np.int8), (1, 1))
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1) - 1
        runs.extend(
            [int(row_index), int(start), int(end)]
            for start, end in zip(starts, ends, strict=True)
        )
    return runs


def save_reference_roi_masks(
    path: str | Path,
    masks: list[ReferenceMask],
    *,
    scar_result=None,
    sd_multiplier: float | None = None,
    image_shape: tuple[int, int] | None = None,
) -> Path:
    """Save arbitrary brush ROIs as compact inclusive x-runs for each row."""

    nonempty = [np.asarray(mask) for mask in masks if mask is not None]
    if nonempty:
        inferred_shape = nonempty[0].shape
        if image_shape is not None and tuple(image_shape) != inferred_shape:
            raise ValueError("image_shape does not match the supplied ROI masks")
        image_shape = inferred_shape
    if image_shape is None:
        raise ValueError("image_shape is required when every ROI mask is empty")
    image_shape = tuple(image_shape)
    if len(image_shape) != 2 or any(size <= 0 for size in image_shape):
        raise ValueError("ROI masks must have two positive dimensions")

    slices = []
    for slice_index, mask in enumerate(masks):
        if mask is None:
            slices.append(None)
            continue
        boolean_mask = np.asarray(mask, dtype=bool)
        if boolean_mask.shape != image_shape:
            raise ValueError("All ROI masks must have the same image shape")
        item = {"row_runs_y_x0_x1": _mask_to_row_runs(boolean_mask)}
        if scar_result is not None:
            item.update(
                {
                    "valid_voxel_count": int(
                        scar_result.reference_voxel_counts[slice_index]
                    ),
                    "mean": float(scar_result.reference_means[slice_index]),
                    "std": float(scar_result.reference_stds[slice_index]),
                    "threshold": float(scar_result.thresholds[slice_index]),
                }
            )
        slices.append(item)

    payload = {
        "coordinate_convention": "zero-based image pixels; inclusive row runs",
        "storage": "brush-mask-row-runs",
        "image_shape_yx": [int(image_shape[0]), int(image_shape[1])],
        "slices": slices,
    }
    if sd_multiplier is not None:
        payload["scar_sd_multiplier"] = float(sd_multiplier)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary_path.replace(path)
    return path


def load_reference_roi_masks(
    path: str | Path,
    expected_shape: tuple[int, int, int],
) -> list[ReferenceMask]:
    """Load brush masks, converting legacy polygon ROIs when necessary."""

    if len(expected_shape) != 3 or any(size <= 0 for size in expected_shape):
        raise ValueError("expected_shape must contain three positive dimensions")
    path = Path(path)
    if not path.exists():
        return [None] * expected_shape[0]
    payload = json.loads(path.read_text(encoding="utf-8"))
    slices = payload.get("slices", [])
    if len(slices) != expected_shape[0]:
        raise ValueError(
            f"Reference ROI file contains {len(slices)} slices, but the study "
            f"has {expected_shape[0]}"
        )

    height, width = expected_shape[1:]
    saved_shape = payload.get("image_shape_yx")
    if saved_shape is not None and tuple(saved_shape) != (height, width):
        raise ValueError(
            f"Reference ROI image shape is {tuple(saved_shape)}, expected "
            f"{(height, width)}"
        )
    masks: list[ReferenceMask] = []
    for item in slices:
        if item is None:
            masks.append(None)
            continue
        if "vertices_xy" in item:
            converted = rasterize_reference_roi_polygons(
                (1, height, width), [_validated_polygon(item["vertices_xy"])]
            )[0]
            masks.append(converted)
            continue
        mask = np.zeros((height, width), dtype=bool)
        for run in item.get("row_runs_y_x0_x1", []):
            if len(run) != 3 or any(isinstance(value, bool) for value in run):
                raise ValueError("Each brush ROI row run must be [y, x0, x1]")
            row, start, end = map(int, run)
            if not (0 <= row < height and 0 <= start <= end < width):
                raise ValueError(f"Brush ROI row run is out of bounds: {run}")
            mask[row, start : end + 1] = True
        masks.append(mask)
    return masks


def _validated_polygon(vertices) -> list[tuple[float, float]]:
    polygon = [tuple(map(float, point)) for point in vertices]
    if len(polygon) < 3:
        raise ValueError("A reference ROI polygon must contain at least 3 vertices")
    coordinates = np.asarray(polygon, dtype=float)
    if coordinates.shape != (len(polygon), 2) or not np.isfinite(coordinates).all():
        raise ValueError("Reference ROI vertices must be finite (x, y) pairs")
    return polygon


def save_reference_roi_polygons(
    path: str | Path,
    polygons: list[ReferencePolygon],
    *,
    scar_result=None,
    sd_multiplier: float | None = None,
) -> Path:
    """Save one optional ``(x, y)`` polygon per cropped short-axis slice."""

    slices = []
    for slice_index, polygon in enumerate(polygons):
        if polygon is None:
            slices.append(None)
            continue
        item = {"vertices_xy": _validated_polygon(polygon)}
        if scar_result is not None:
            item.update(
                {
                    "valid_voxel_count": int(
                        scar_result.reference_voxel_counts[slice_index]
                    ),
                    "mean": float(scar_result.reference_means[slice_index]),
                    "std": float(scar_result.reference_stds[slice_index]),
                    "threshold": float(scar_result.thresholds[slice_index]),
                }
            )
        slices.append(item)
    payload = {
        "coordinate_convention": "zero-based (x, y) image pixel coordinates",
        "slices": slices,
    }
    if sd_multiplier is not None:
        payload["scar_sd_multiplier"] = float(sd_multiplier)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary_path.replace(path)
    return path


def load_reference_roi_polygons(
    path: str | Path,
    expected_slice_count: int,
) -> list[ReferencePolygon]:
    """Load saved polygons or return an empty annotation set."""

    path = Path(path)
    if not path.exists():
        return [None] * expected_slice_count
    payload = json.loads(path.read_text(encoding="utf-8"))
    slices = payload.get("slices", [])
    if len(slices) != expected_slice_count:
        raise ValueError(
            f"Reference ROI file contains {len(slices)} slices, but the study "
            f"has {expected_slice_count}"
        )
    return [
        None if item is None else _validated_polygon(item["vertices_xy"])
        for item in slices
    ]


def rasterize_reference_roi_polygons(
    volume_shape: tuple[int, int, int],
    polygons: list[ReferencePolygon],
) -> NDArray[np.bool_]:
    """Rasterize per-slice polygons at integer pixel-centre coordinates."""

    if len(volume_shape) != 3 or any(size <= 0 for size in volume_shape):
        raise ValueError("volume_shape must contain three positive dimensions")
    if len(polygons) != volume_shape[0]:
        raise ValueError("There must be one reference ROI entry per slice")
    _, height, width = volume_shape
    grid_y, grid_x = np.mgrid[:height, :width]
    points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    masks = np.zeros(volume_shape, dtype=bool)
    for slice_index, vertices in enumerate(polygons):
        if vertices is None:
            continue
        polygon = MatplotlibPath(_validated_polygon(vertices))
        masks[slice_index] = polygon.contains_points(points, radius=1.0e-9).reshape(
            height, width
        )
    return masks


def _ensure_interactive_backend() -> None:
    backend = matplotlib.get_backend().lower()
    if "inline" not in backend and not backend.endswith("agg"):
        return
    errors = []
    for candidate in ("QtAgg", "TkAgg"):
        try:
            plt.switch_backend(candidate)
            return
        except (ImportError, RuntimeError) as error:
            errors.append(f"{candidate}: {error}")
    raise RuntimeError(
        "Normal-myocardium ROI annotation requires an interactive Matplotlib "
        f"backend. Backend errors: {'; '.join(errors)}"
    )


def select_reference_rois_interactively(
    study: StudyData,
    landmarks: list[LandmarkPair | None],
    output_path: str | Path,
    min_roi_voxels: int,
    *,
    sd_multiplier: float = 5.0,
    deleted_slice_indices: Iterable[int] = (),
    revisit_completed: bool = False,
    resume_existing: bool = True,
) -> list[ReferenceMask]:
    """Paint one normal-myocardium brush mask on every retained slice.

    Set ``resume_existing=False`` to start every retained slice with a blank
    mask, even when ``output_path`` already contains saved annotations.
    """

    if len(landmarks) != study.slice_count:
        raise ValueError("There must be one landmark entry per slice")
    if isinstance(min_roi_voxels, bool) or min_roi_voxels < 2:
        raise ValueError("min_roi_voxels must be an integer of at least 2")
    output_path = Path(output_path)
    roi_masks = (
        load_reference_roi_masks(output_path, study.mr.shape)
        if resume_existing
        else [None] * study.slice_count
    )
    deleted = {int(index) for index in deleted_slice_indices}
    invalid_deleted = sorted(
        index for index in deleted if index < 0 or index >= study.slice_count
    )
    if invalid_deleted:
        raise ValueError(f"Deleted slice indices are out of range: {invalid_deleted}")
    stale_indices = [
        index
        for index in range(study.slice_count)
        if (index in deleted or not np.any(study.territory_mask[index] > 0))
        and roi_masks[index] is not None
    ]
    for index in stale_indices:
        roi_masks[index] = None
    if stale_indices:
        save_reference_roi_masks(output_path, roi_masks, image_shape=study.mr.shape[1:])
    pending = [
        index
        for index in range(study.slice_count)
        if (
            index not in deleted
            and np.any(study.territory_mask[index] > 0)
            and (revisit_completed or roi_masks[index] is None)
        )
    ]
    if not pending:
        return roi_masks

    _ensure_interactive_backend()
    figure, axis = plt.subplots(figsize=(8, 8))
    figure.subplots_adjust(bottom=0.23)
    figure.canvas.manager.set_window_title("Normal myocardium ROI annotation")
    accept_button = Button(figure.add_axes((0.035, 0.035, 0.13, 0.055)), "Accept")
    undo_button = Button(figure.add_axes((0.18, 0.035, 0.13, 0.055)), "Undo")
    clear_button = Button(figure.add_axes((0.325, 0.035, 0.13, 0.055)), "Clear")
    fit_button = Button(figure.add_axes((0.47, 0.035, 0.13, 0.055)), "Fit")
    quit_button = Button(figure.add_axes((0.615, 0.035, 0.25, 0.055)), "Save && Quit")
    brush_slider = Slider(
        figure.add_axes((0.18, 0.125, 0.64, 0.035)),
        "Brush radius",
        1.0,
        20.0,
        valinit=4.0,
        valstep=1.0,
    )
    state: dict[str, object] = {
        "slice_index": pending[0],
        "mask": np.zeros(study.mr.shape[1:], dtype=bool),
        "undo": [],
        "stroke_before": None,
        "last_xy": None,
        "paint_value": None,
        "pan_start": None,
        "action": None,
        "closed": False,
    }

    def current_selected_mask() -> NDArray[np.bool_]:
        mask = state["mask"]
        assert isinstance(mask, np.ndarray)
        slice_index = int(state["slice_index"])
        return (
            mask
            & (study.territory_mask[slice_index] > 0)
            & np.isfinite(study.mr[slice_index])
        )

    def full_view() -> None:
        height, width = study.mr.shape[1:]
        axis.set_xlim(-0.5, width - 0.5)
        axis.set_ylim(height - 0.5, -0.5)

    def draw_current_slice(
        message: str | None = None, *, preserve_view: bool = True
    ) -> None:
        slice_index = int(state["slice_index"])
        image = annotation_display_transform(study.mr[slice_index])
        myocardium = annotation_display_transform(study.territory_mask[slice_index] > 0)
        old_xlim = axis.get_xlim() if preserve_view and axis.has_data() else None
        old_ylim = axis.get_ylim() if preserve_view and axis.has_data() else None
        axis.clear()
        finite = image[np.isfinite(image)]
        vmin, vmax = np.percentile(finite, [1.0, 99.0]) if finite.size else (0.0, 1.0)
        if vmax <= vmin:
            vmax = vmin + 1.0
        axis.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
        axis.contour(myocardium, levels=[0.5], colors=["yellow"], linewidths=1.0)
        pair = landmarks[slice_index]
        if pair is not None:
            axis.plot(pair.aip[0], pair.aip[1], "o", color="red", markersize=7)
            axis.plot(pair.iip[0], pair.iip[1], "o", color="cyan", markersize=7)
        selected = current_selected_mask()
        roi_overlay = np.zeros((*selected.shape, 4), dtype=float)
        roi_overlay[selected] = (0.0, 0.65, 1.0, 0.5)
        axis.imshow(roi_overlay)
        count = int(np.count_nonzero(selected))
        statistics = ""
        if count >= min_roi_voxels:
            _, mean, std, threshold = compute_reference_statistics(
                image,
                myocardium,
                selected,
                sd_multiplier=sd_multiplier,
                min_roi_voxels=min_roi_voxels,
            )
            preview = myocardium & np.isfinite(image) & (image > threshold)
            overlay = np.zeros((*preview.shape, 4), dtype=float)
            overlay[preview] = (1.0, 0.0, 0.0, 0.38)
            axis.imshow(overlay)
            statistics = (
                f" | voxels={count}, mean={mean:.3f}, std={std:.3f}, "
                f"{sd_multiplier:g}-SD={threshold:.3f}"
            )
        instruction = (
            "Left drag: paint | Right drag: erase | Wheel: zoom | " "Middle drag: pan"
            if message is None
            else message
        )
        axis.set_title(
            f"Slice {slice_index + 1}/{study.slice_count} | "
            f"brush={brush_slider.val:.0f}px{statistics}\n{instruction}"
        )
        if old_xlim is None or old_ylim is None:
            full_view()
        else:
            axis.set_xlim(old_xlim)
            axis.set_ylim(old_ylim)
        axis.set_axis_off()
        figure.canvas.draw_idle()

    def stop(action: str) -> None:
        state["action"] = action
        figure.canvas.stop_event_loop()

    def on_accept(_event) -> None:
        count = int(np.count_nonzero(current_selected_mask()))
        if count < min_roi_voxels:
            draw_current_slice(
                f"ROI has {count} valid myocardial voxels; "
                f"at least {min_roi_voxels} are required"
            )
            return
        stop("accept")

    def on_undo(_event) -> None:
        undo = state["undo"]
        assert isinstance(undo, list)
        if undo:
            state["mask"] = undo.pop()
        draw_current_slice()

    def on_clear(_event) -> None:
        mask = state["mask"]
        undo = state["undo"]
        assert isinstance(mask, np.ndarray) and isinstance(undo, list)
        if np.any(mask):
            undo.append(mask.copy())
            state["mask"] = np.zeros_like(mask)
        draw_current_slice()

    def on_fit(_event) -> None:
        full_view()
        figure.canvas.draw_idle()

    def on_quit(_event) -> None:
        stop("quit")

    def on_close(_event) -> None:
        state["closed"] = True
        state["action"] = "quit"
        try:
            figure.canvas.stop_event_loop()
        except AttributeError:
            pass

    def valid_myocardium() -> NDArray[np.bool_]:
        slice_index = int(state["slice_index"])
        return (study.territory_mask[slice_index] > 0) & np.isfinite(
            study.mr[slice_index]
        )

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
        if event.button not in (MouseButton.LEFT, MouseButton.RIGHT):
            return
        mask = state["mask"]
        assert isinstance(mask, np.ndarray)
        state["stroke_before"] = mask.copy()
        state["paint_value"] = event.button is MouseButton.LEFT
        point = (float(event.xdata), float(event.ydata))
        state["last_xy"] = point
        apply_brush_stroke(
            mask,
            point,
            point,
            radius=float(brush_slider.val),
            value=bool(state["paint_value"]),
        )
        mask &= valid_myocardium()
        draw_current_slice()

    def on_mouse_motion(event) -> None:
        pan_start = state["pan_start"]
        if pan_start is not None:
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
            return
        last_xy = state["last_xy"]
        if (
            last_xy is None
            or event.inaxes is not axis
            or event.xdata is None
            or event.ydata is None
        ):
            return
        mask = state["mask"]
        assert isinstance(mask, np.ndarray)
        point = (float(event.xdata), float(event.ydata))
        apply_brush_stroke(
            mask,
            last_xy,
            point,
            radius=float(brush_slider.val),
            value=bool(state["paint_value"]),
        )
        mask &= valid_myocardium()
        state["last_xy"] = point
        draw_current_slice()

    def on_mouse_release(event) -> None:
        if event.button is MouseButton.MIDDLE:
            state["pan_start"] = None
            return
        if event.button not in (MouseButton.LEFT, MouseButton.RIGHT):
            return
        previous = state["stroke_before"]
        undo = state["undo"]
        assert isinstance(undo, list)
        if isinstance(previous, np.ndarray):
            undo.append(previous)
            if len(undo) > 30:
                del undo[0]
        state["stroke_before"] = None
        state["last_xy"] = None
        state["paint_value"] = None

    def on_scroll(event) -> None:
        if event.inaxes is not axis or event.xdata is None or event.ydata is None:
            return
        scale = 0.8 if event.button == "up" else 1.25
        xlim, ylim = zoom_limits_about_point(
            axis.get_xlim(),
            axis.get_ylim(),
            (float(event.xdata), float(event.ydata)),
            scale=scale,
            image_shape=study.mr.shape[1:],
        )
        axis.set_xlim(xlim)
        axis.set_ylim(ylim)
        figure.canvas.draw_idle()

    def on_brush_radius_changed(_value) -> None:
        if axis.has_data():
            draw_current_slice()

    accept_button.on_clicked(on_accept)
    undo_button.on_clicked(on_undo)
    clear_button.on_clicked(on_clear)
    fit_button.on_clicked(on_fit)
    quit_button.on_clicked(on_quit)
    brush_slider.on_changed(on_brush_radius_changed)
    figure.canvas.mpl_connect("button_press_event", on_mouse_press)
    figure.canvas.mpl_connect("motion_notify_event", on_mouse_motion)
    figure.canvas.mpl_connect("button_release_event", on_mouse_release)
    figure.canvas.mpl_connect("scroll_event", on_scroll)
    figure.canvas.mpl_connect("close_event", on_close)
    plt.show(block=False)

    for slice_index in pending:
        state["slice_index"] = slice_index
        state["action"] = None
        existing = roi_masks[slice_index]
        state["mask"] = (
            np.zeros(study.mr.shape[1:], dtype=bool)
            if existing is None
            else existing.copy()
        )
        state["undo"] = []
        state["stroke_before"] = None
        state["last_xy"] = None
        state["paint_value"] = None
        state["pan_start"] = None
        draw_current_slice(preserve_view=False)
        figure.canvas.start_event_loop(timeout=-1)
        if state["action"] == "quit" or state["closed"]:
            save_reference_roi_masks(
                output_path, roi_masks, image_shape=study.mr.shape[1:]
            )
            if not state["closed"]:
                plt.close(figure)
            return roi_masks
        roi_masks[slice_index] = current_selected_mask().copy()
        save_reference_roi_masks(output_path, roi_masks, image_shape=study.mr.shape[1:])

    if not state["closed"]:
        plt.close(figure)
    return roi_masks
