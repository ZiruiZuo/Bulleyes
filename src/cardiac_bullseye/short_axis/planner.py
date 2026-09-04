"""Matplotlib desktop UI for interactive cardiac short-axis planning."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider, TextBox

from .geometry import (
    LongAxisPlan,
    PlaneGrid,
    build_long_axis_plan,
    build_vla_from_axial_points,
    fit_plane_grid,
    generate_short_axis_stack,
    line_segment_for_level,
    projection_limits,
    resample_plane,
    voxel_axis_world_direction,
)
from .nifti import NiftiVolume, load_nifti, save_nifti, validate_mask_grid
from .metadata import default_geometry_path, save_planning_geometry


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
        "Short-axis planning requires an interactive Matplotlib window, but "
        "neither QtAgg nor TkAgg could be enabled. In Spyder, set Preferences "
        "> IPython console > Graphics > Backend to Qt, restart the console, "
        f"and run cardiac-short-axis again. Backend errors: {'; '.join(errors)}"
    )


class ShortAxisPlanner:
    """Interactive state machine implementing the clinical planning workflow."""

    def __init__(
        self,
        image: NiftiVolume,
        output_path: str | Path,
        mask: NiftiVolume | None = None,
        mask_output_path: str | Path | None = None,
        geometry_output_path: str | Path | None = None,
        slice_spacing_mm: float = 8.0,
        pixel_spacing_mm: float | None = None,
        fov_mm: tuple[float, float] | None = None,
    ) -> None:
        if mask is not None:
            validate_mask_grid(image, mask)
        if slice_spacing_mm <= 0:
            raise ValueError("slice spacing must be positive")
        self.image = image
        self.mask = mask
        self.output_path = Path(output_path).expanduser().resolve()
        self.mask_output_path = (
            Path(mask_output_path).expanduser().resolve() if mask_output_path else None
        )
        self.geometry_output_path = (
            Path(geometry_output_path).expanduser().resolve()
            if geometry_output_path
            else default_geometry_path(self.output_path)
        )
        voxel_sizes = np.linalg.norm(image.affine[:3, :3], axis=0)
        self.pixel_spacing_mm = float(pixel_spacing_mm or np.min(voxel_sizes))
        self.slice_spacing_mm = float(slice_spacing_mm)
        self.fov_mm = fov_mm
        self._display_range = self._calculate_display_range()

        self.stage = "axial"
        self.axial_index = int(image.data.shape[2] // 2)
        self.points: list[np.ndarray] = []
        self.axial_points: list[np.ndarray] = []
        self.vla_grid: PlaneGrid | None = None
        self.hla_grid: PlaneGrid | None = None
        self.plan: LongAxisPlan | None = None
        self.apex_t = 0.0
        self.base_t = 1.0
        self.t_limits = (0.0, 1.0)
        self.drag_boundary: str | None = None
        self.line_artists = []

        self.fig, (self.ax_left, self.ax_right) = plt.subplots(1, 2, figsize=(14, 8))
        self.fig.canvas.manager.set_window_title("NIfTI Cardiac Short-Axis Planner")
        self.fig.subplots_adjust(left=0.04, right=0.98, top=0.88, bottom=0.20, wspace=0.08)
        self.title = self.fig.suptitle("", fontsize=13)
        self.status = self.fig.text(0.04, 0.145, "", fontsize=10, color="#333333")

        slider_ax = self.fig.add_axes([0.11, 0.075, 0.45, 0.035])
        self.slice_slider = Slider(
            slider_ax,
            "Axial slice",
            0,
            image.data.shape[2] - 1,
            valinit=self.axial_index,
            valstep=1,
        )
        self.slice_slider.on_changed(self._on_slice)
        spacing_ax = self.fig.add_axes([0.68, 0.075, 0.10, 0.04])
        self.spacing_box = TextBox(spacing_ax, "SA gap (mm) ", initial=f"{slice_spacing_mm:g}")
        self.spacing_box.on_submit(self._on_spacing)
        back_ax = self.fig.add_axes([0.80, 0.068, 0.055, 0.05])
        reset_ax = self.fig.add_axes([0.86, 0.068, 0.055, 0.05])
        next_ax = self.fig.add_axes([0.92, 0.068, 0.065, 0.05])
        self.back_button = Button(back_ax, "Back")
        self.reset_button = Button(reset_ax, "Reset")
        self.next_button = Button(next_ax, "Next")
        self.back_button.on_clicked(self._on_back)
        self.reset_button.on_clicked(self._on_reset)
        self.next_button.on_clicked(self._on_next)

        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self._render_axial()

    @property
    def display_range(self) -> tuple[float, float]:
        return self._display_range

    def _calculate_display_range(self) -> tuple[float, float]:
        # A bounded sample avoids copying/sorting a complete CT or MR volume.
        flat = self.image.data.ravel()
        stride = max(1, flat.size // 500_000)
        values = flat[::stride]
        values = values[np.isfinite(values)]
        nonzero = values[values != 0]
        sample = nonzero if nonzero.size > 100 else values
        if sample.size == 0:
            return 0.0, 1.0
        low, high = np.percentile(sample, [1.0, 99.0])
        if high <= low:
            high = low + 1.0
        return float(low), float(high)

    def show(self) -> None:
        # Blocking is intentional: the planner owns its desktop window until
        # the user saves or closes it. Non-blocking IDE modes often render only
        # a snapshot in a Plots panel and discard widget interaction.
        plt.ioff()
        plt.show(block=True)

    def _set_message(self, text: str, error: bool = False) -> None:
        self.status.set_text(text)
        self.status.set_color("#b00020" if error else "#333333")
        self.fig.canvas.draw_idle()

    def _show_image(self, ax, image: np.ndarray, mask: np.ndarray | None = None) -> None:
        low, high = self.display_range
        ax.imshow(image, cmap="gray", origin="lower", interpolation="bilinear", vmin=low, vmax=high)
        if mask is not None and np.any(mask > 0):
            alpha = np.where(mask > 0, 0.34, 0.0)
            ax.imshow(mask, cmap="turbo", origin="lower", interpolation="nearest", alpha=alpha)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    def _draw_points(self, ax, points: list[np.ndarray], world_to_pixel: Callable | None = None) -> None:
        labels = ("Apex", "Mitral")
        colors = ("#00e5ff", "#ffb000")
        for index, point in enumerate(points):
            xy = world_to_pixel(point) if world_to_pixel else point
            ax.plot(xy[0], xy[1], "+", ms=15, mew=2.2, color=colors[index])
            ax.annotate(
                labels[index],
                (xy[0], xy[1]),
                xytext=(7, 7),
                textcoords="offset points",
                color=colors[index],
                fontsize=9,
            )
        if len(points) == 2:
            p0 = world_to_pixel(points[0]) if world_to_pixel else points[0]
            p1 = world_to_pixel(points[1]) if world_to_pixel else points[1]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], "--", color="#00ff7f", lw=1.4)

    @staticmethod
    def _world_to_pixel(grid: PlaneGrid, world: np.ndarray) -> np.ndarray:
        delta = np.asarray(world) - grid.center
        x = np.dot(delta, grid.x_axis) / grid.pixel_spacing + (grid.width - 1) / 2.0
        y = np.dot(delta, grid.y_axis) / grid.pixel_spacing + (grid.height - 1) / 2.0
        return np.asarray([x, y])

    def _render_axial(self) -> None:
        self.stage = "axial"
        self.ax_left.clear()
        self.ax_right.clear()
        self.ax_right.set_visible(False)
        self.ax_left.set_visible(True)
        index = self.axial_index
        image = self.image.data[:, :, index].T
        mask = self.mask.data[:, :, index].T if self.mask is not None else None
        self._show_image(self.ax_left, image, mask)
        # Display the native axial view in mirrored left-right orientation.
        # Reversing the axis (instead of copying/flipping the array) keeps
        # event.xdata in the original voxel coordinate system, so landmark
        # clicks still map to the correct NIfTI world positions.
        self.ax_left.invert_xaxis()
        self.ax_left.set_title(f"Native axial view — slice {index}")
        display_points = []
        inverse = np.linalg.inv(self.image.affine)
        for point in self.points:
            voxel = inverse @ [*point, 1.0]
            display_points.append(np.asarray([voxel[0], voxel[1]]))
        self._draw_points(self.ax_left, display_points)
        self.title.set_text("Step 1/3: choose an axial slice, then click Apex and Mitral centre")
        self.slice_slider.ax.set_visible(True)
        self.back_button.ax.set_visible(False)
        self.next_button.label.set_text("Next")
        self._set_message("Mouse wheel/slider changes slice. Point order: Apex first, Mitral centre second.")
        self.fig.canvas.draw_idle()

    def _render_vla_selection(self) -> None:
        assert self.vla_grid is not None
        self.stage = "vla"
        self.ax_left.clear()
        self.ax_right.clear()
        self.ax_right.set_visible(False)
        image = resample_plane(self.image.data, self.image.affine, self.vla_grid, order=1)
        mask = (
            resample_plane(self.mask.data, self.mask.affine, self.vla_grid, order=0)
            if self.mask is not None
            else None
        )
        self._show_image(self.ax_left, image, mask)
        self.ax_left.set_title("Vertical long axis (approx. 2-chamber)")
        self._draw_points(
            self.ax_left, self.points, lambda p: self._world_to_pixel(self.vla_grid, p)
        )
        self.title.set_text("Step 2/3: refine the LV long axis on the vertical long-axis view")
        self.slice_slider.ax.set_visible(False)
        self.back_button.ax.set_visible(True)
        self.next_button.label.set_text("Next")
        self._set_message("Click Apex first and Mitral centre second on this reformatted view.")
        self.fig.canvas.draw_idle()

    def _render_planning(self) -> None:
        assert self.vla_grid is not None and self.hla_grid is not None and self.plan is not None
        self.stage = "plan"
        self.ax_left.clear()
        self.ax_right.clear()
        self.ax_right.set_visible(True)
        for ax, grid, label in (
            (self.ax_left, self.vla_grid, "Vertical long axis / 2-chamber"),
            (self.ax_right, self.hla_grid, "Horizontal long axis / 4-chamber"),
        ):
            image = resample_plane(self.image.data, self.image.affine, grid, order=1)
            mask = (
                resample_plane(self.mask.data, self.mask.affine, grid, order=0)
                if self.mask is not None
                else None
            )
            self._show_image(ax, image, mask)
            ax.set_title(label)
        self.title.set_text("Step 3/3: drag cyan Apex and orange Base boundary lines; then Save")
        self.slice_slider.ax.set_visible(False)
        self.back_button.ax.set_visible(True)
        self.next_button.label.set_text("Save")
        self._draw_locator_lines()
        self._set_message(self._coverage_message())
        self.fig.canvas.draw_idle()

    def _coverage_message(self) -> str:
        count = int(np.floor((self.base_t - self.apex_t) / self.slice_spacing_mm + 1e-7)) + 1
        return (
            f"Coverage {self.base_t - self.apex_t:.1f} mm; {count} slices; "
            f"spacing {self.slice_spacing_mm:g} mm. Drag either coloured boundary in either panel."
        )

    def _draw_locator_lines(self) -> None:
        assert self.plan is not None and self.vla_grid is not None and self.hla_grid is not None
        for artist in self.line_artists:
            try:
                artist.remove()
            except ValueError:
                pass
        self.line_artists.clear()
        positions = self.apex_t + np.arange(
            int(np.floor((self.base_t - self.apex_t) / self.slice_spacing_mm + 1e-7)) + 1
        ) * self.slice_spacing_mm
        for ax, grid in ((self.ax_left, self.vla_grid), (self.ax_right, self.hla_grid)):
            for level in positions:
                segment = line_segment_for_level(grid, self.plan.apex, self.plan.long_axis, level)
                if segment is not None:
                    artist = ax.plot(
                        [segment[0][0], segment[1][0]],
                        [segment[0][1], segment[1][1]],
                        color="white",
                        alpha=0.6,
                        lw=0.8,
                    )[0]
                    self.line_artists.append(artist)
            for level, color, label in (
                (self.apex_t, "#00e5ff", "Apex"),
                (self.base_t, "#ff9f1c", "Base"),
            ):
                segment = line_segment_for_level(grid, self.plan.apex, self.plan.long_axis, level)
                if segment is not None:
                    artist = ax.plot(
                        [segment[0][0], segment[1][0]],
                        [segment[0][1], segment[1][1]],
                        color=color,
                        lw=2.5,
                        label=label,
                    )[0]
                    self.line_artists.append(artist)
            ax.legend(loc="upper right", facecolor="black", labelcolor="white", framealpha=0.55)

    def _on_slice(self, value: float) -> None:
        if self.stage != "axial":
            return
        self.axial_index = int(round(value))
        self.points.clear()
        self._render_axial()

    def _on_spacing(self, text: str) -> None:
        try:
            value = float(text)
            if value <= 0:
                raise ValueError
        except ValueError:
            self._set_message("SA gap must be a positive number.", error=True)
            self.spacing_box.set_val(f"{self.slice_spacing_mm:g}")
            return
        self.slice_spacing_mm = value
        if self.stage == "plan":
            if self.base_t - self.apex_t < value:
                self.base_t = min(self.t_limits[1], self.apex_t + value)
            self._draw_locator_lines()
            self._set_message(self._coverage_message())
            self.fig.canvas.draw_idle()

    def _click_to_world(self, event) -> np.ndarray | None:
        if event.xdata is None or event.ydata is None:
            return None
        if self.stage == "axial" and event.inaxes is self.ax_left:
            return (self.image.affine @ [event.xdata, event.ydata, self.axial_index, 1.0])[:3]
        if self.stage == "vla" and event.inaxes is self.ax_left and self.vla_grid is not None:
            return self.vla_grid.pixel_to_world(event.xdata, event.ydata)
        if self.stage == "plan":
            grid = self.vla_grid if event.inaxes is self.ax_left else self.hla_grid
            if grid is not None and event.inaxes in (self.ax_left, self.ax_right):
                return grid.pixel_to_world(event.xdata, event.ydata)
        return None

    def _on_press(self, event) -> None:
        if event.button != 1:
            return
        world = self._click_to_world(event)
        if world is None:
            return
        if self.stage in ("axial", "vla"):
            if len(self.points) < 2:
                self.points.append(world)
                if self.stage == "axial":
                    self._render_axial()
                else:
                    self._render_vla_selection()
            return
        assert self.plan is not None
        clicked_t = float(np.dot(world - self.plan.apex, self.plan.long_axis))
        distances = {"apex": abs(clicked_t - self.apex_t), "base": abs(clicked_t - self.base_t)}
        boundary = min(distances, key=distances.get)
        threshold = max(5.0, self.slice_spacing_mm * 0.8)
        if distances[boundary] <= threshold:
            self.drag_boundary = boundary

    def _on_motion(self, event) -> None:
        if self.stage != "plan" or self.drag_boundary is None or self.plan is None:
            return
        world = self._click_to_world(event)
        if world is None:
            return
        value = float(np.dot(world - self.plan.apex, self.plan.long_axis))
        value = float(np.clip(value, self.t_limits[0], self.t_limits[1]))
        minimum_gap = min(self.slice_spacing_mm, max(0.1, self.t_limits[1] - self.t_limits[0]))
        if self.drag_boundary == "apex":
            self.apex_t = min(value, self.base_t - minimum_gap)
        else:
            self.base_t = max(value, self.apex_t + minimum_gap)
        self._draw_locator_lines()
        self._set_message(self._coverage_message())
        self.fig.canvas.draw_idle()

    def _on_release(self, _event) -> None:
        self.drag_boundary = None

    def _on_reset(self, _event) -> None:
        if self.stage in ("axial", "vla"):
            self.points.clear()
            self._render_axial() if self.stage == "axial" else self._render_vla_selection()
        else:
            assert self.plan is not None
            self.apex_t = 0.0
            self.base_t = float(np.dot(self.plan.valve - self.plan.apex, self.plan.long_axis))
            self._render_planning()

    def _on_back(self, _event) -> None:
        if self.stage == "vla":
            self.points = list(self.axial_points)
            self._render_axial()
        elif self.stage == "plan":
            assert self.plan is not None
            self.points = [self.plan.apex.copy(), self.plan.valve.copy()]
            self._render_vla_selection()

    def _on_next(self, _event) -> None:
        try:
            if self.stage == "axial":
                if len(self.points) != 2:
                    raise ValueError("select Apex and Mitral centre before continuing")
                self.axial_points = [point.copy() for point in self.points]
                normal = voxel_axis_world_direction(self.image.affine, 2)
                vx, vy, _ = build_vla_from_axial_points(*self.points, normal)
                self.vla_grid = fit_plane_grid(
                    self.image.data.shape,
                    self.image.affine,
                    0.5 * (self.points[0] + self.points[1]),
                    vx,
                    vy,
                    self.pixel_spacing_mm,
                )
                self.points.clear()
                self._render_vla_selection()
            elif self.stage == "vla":
                if len(self.points) != 2:
                    raise ValueError("select Apex and Mitral centre before continuing")
                assert self.vla_grid is not None
                self.plan = build_long_axis_plan(
                    self.vla_grid.x_axis, self.vla_grid.y_axis, self.points[0], self.points[1]
                )
                self.hla_grid = fit_plane_grid(
                    self.image.data.shape,
                    self.image.affine,
                    0.5 * (self.points[0] + self.points[1]),
                    self.plan.hla_x_axis,
                    self.plan.hla_y_axis,
                    self.pixel_spacing_mm,
                )
                self.apex_t = 0.0
                self.base_t = float(np.dot(self.plan.valve - self.plan.apex, self.plan.long_axis))
                self.t_limits = projection_limits(
                    self.image.data.shape, self.image.affine, self.plan.apex, self.plan.long_axis
                )
                self._render_planning()
            else:
                self._save()
        except Exception as exc:
            self._set_message(str(exc), error=True)

    def _save(self) -> None:
        assert self.plan is not None
        result = generate_short_axis_stack(
            self.image.data,
            self.image.affine,
            self.plan,
            self.apex_t,
            self.base_t,
            self.slice_spacing_mm,
            self.pixel_spacing_mm,
            self.fov_mm,
            interpolation_order=1,
        )
        save_nifti(self.output_path, result.data.astype(np.float32), result.affine, self.image.header)
        saved = [str(self.output_path)]
        geometry_path = save_planning_geometry(
            self.geometry_output_path,
            plan=self.plan,
            stack=result,
            apex_t_mm=self.apex_t,
            base_t_mm=self.base_t,
            source_image_path=self.image.path,
            short_axis_path=self.output_path,
        )
        saved.append(str(geometry_path))
        if self.mask is not None:
            mask_result = generate_short_axis_stack(
                self.mask.data,
                self.mask.affine,
                self.plan,
                self.apex_t,
                self.base_t,
                self.slice_spacing_mm,
                self.pixel_spacing_mm,
                self.fov_mm,
                interpolation_order=0,
            )
            output = self.mask_output_path or self.output_path.with_name(
                self.output_path.name.replace(".nii.gz", "_mask.nii.gz")
                if self.output_path.name.endswith(".nii.gz")
                else self.output_path.stem + "_mask.nii"
            )
            dtype = self.mask.data.dtype if np.issubdtype(self.mask.data.dtype, np.integer) else np.int16
            save_nifti(output, np.rint(mask_result.data).astype(dtype), mask_result.affine, self.mask.header)
            saved.append(str(output))
        self._set_message(
            f"Saved {result.data.shape[2]} short-axis slices: " + " | ".join(saved)
        )


def interactive_convert_to_short_axis(
    image_path: str | Path,
    output_path: str | Path,
    mask_path: str | Path | None = None,
    mask_output_path: str | Path | None = None,
    geometry_output_path: str | Path | None = None,
    slice_spacing_mm: float = 8.0,
    pixel_spacing_mm: float | None = None,
    fov_mm: tuple[float, float] | None = None,
) -> ShortAxisPlanner:
    """Open the interactive planner and return its state after the window closes."""
    _ensure_interactive_backend()
    image = load_nifti(image_path)
    mask = load_nifti(mask_path, dtype=None) if mask_path else None
    planner = ShortAxisPlanner(
        image,
        output_path,
        mask,
        mask_output_path,
        geometry_output_path,
        slice_spacing_mm,
        pixel_spacing_mm,
        fov_mm,
    )
    planner.show()
    return planner
