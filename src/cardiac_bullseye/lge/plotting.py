"""Plot continuous raw-intensity LGE bullseyes."""

from __future__ import annotations

import errno
import os
from pathlib import Path
import time
import uuid

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.patches import Wedge

from ..coronary.projection import ProjectedCoronaryPath
from ..territory.aha17 import (
    FOUR_SECTOR_CENTERS_DEG,
    SIX_SECTOR_CENTERS_DEG,
    draw_aha17_boundaries,
)
from ..territory.plotting import (
    CORONARY_AUTO_COLORS,
    CORONARY_DEFAULT_LABEL_COLORS,
)
from .mapping import LgeBullseyeResult
from .scar import AhaScarSummary


def _save_figure_overwriting(
    figure,
    output_path: str | Path,
    *,
    retry_attempts: int = 5,
    retry_delay_seconds: float = 0.2,
) -> Path:
    """Save completely, then atomically replace an existing output image."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix or ".png"
    temporary_path = output_path.with_name(
        f".{output_path.stem}.{uuid.uuid4().hex}.tmp{suffix}"
    )
    try:
        figure.savefig(temporary_path, dpi=220, bbox_inches="tight")
        last_error: OSError | None = None
        for attempt in range(retry_attempts):
            try:
                os.replace(temporary_path, output_path)
                return output_path
            except OSError as error:
                last_error = error
                if attempt + 1 < retry_attempts:
                    time.sleep(retry_delay_seconds)
        assert last_error is not None
        raise OSError(
            last_error.errno or errno.EACCES,
            "Could not overwrite the output image after repeated attempts. "
            "Close any application displaying this PNG and run again",
            str(output_path),
        ) from last_error
    finally:
        if temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                pass


def _scar_heat_colormap() -> LinearSegmentedColormap:
    """Return a white-to-red-orange heat scale for scar burden."""

    return LinearSegmentedColormap.from_list(
        "scar_heat_white_orange_red",
        ("#ffffff", "#fff3bf", "#fdae61", "#f46d43", "#d7301f"),
    )


def _centers_to_edges(values: np.ndarray) -> np.ndarray:
    midpoints = (values[:-1] + values[1:]) / 2.0
    return np.concatenate(
        (
            [values[0] - (midpoints[0] - values[0])],
            midpoints,
            [values[-1] + (values[-1] - midpoints[-1])],
        )
    )


def _draw_projected_coronaries(
    axis,
    coronary_paths: list[ProjectedCoronaryPath] | None,
    *,
    coronary_label_colors: dict[int, str] | None,
    coronary_label_names: dict[int, str] | None,
    coronary_default_color: str | None,
    coronary_linewidth: float,
) -> float:
    """Draw projected coronary polylines and return their largest radius."""

    paths = coronary_paths or []
    label_colors = coronary_label_colors or {}
    label_names = coronary_label_names or {}
    labels = sorted({path.label for path in paths})
    resolved_colors = {
        label: label_colors.get(
            label,
            coronary_default_color
            if coronary_default_color is not None
            else CORONARY_DEFAULT_LABEL_COLORS.get(
                label,
                CORONARY_AUTO_COLORS[index % len(CORONARY_AUTO_COLORS)],
            ),
        )
        for index, label in enumerate(labels)
    }
    for path in paths:
        axis.plot(
            path.xy[:, 0],
            path.xy[:, 1],
            color=resolved_colors[path.label],
            linewidth=coronary_linewidth,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=20,
        )

    finite_radii = [
        np.linalg.norm(path.xy[np.isfinite(path.xy).all(axis=1)], axis=1)
        for path in paths
        if np.isfinite(path.xy).all(axis=1).any()
    ]
    maximum_radius = max(
        (float(values.max()) for values in finite_radii if values.size),
        default=0.0,
    )
    if labels:
        axis.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    color=resolved_colors[label],
                    linewidth=coronary_linewidth,
                    label=label_names.get(
                        label,
                        "Coronary" if len(labels) == 1 else f"Coronary {label}",
                    ),
                )
                for label in labels
            ],
            loc="upper left",
            frameon=False,
        )
    return maximum_radius


def plot_lge_bullseye(
    result: LgeBullseyeResult,
    output_path: str | Path | None = None,
    *,
    title: str = "Continuous LGE intensity bullseye",
    colormap: str = "gray",
    show_numbers: bool = True,
    colorbar_label: str = "LGE intensity (native units)",
    coronary_paths: list[ProjectedCoronaryPath] | None = None,
    coronary_label_colors: dict[int, str] | None = None,
    coronary_label_names: dict[int, str] | None = None,
    coronary_default_color: str | None = None,
    coronary_linewidth: float = 1.8,
    show: bool = True,
):
    """Render LGE intensity; percentiles affect display only, not saved data."""

    values = np.asarray(result.intensity, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("LGE bullseye contains no finite intensity values")
    vmin, vmax = np.nanpercentile(finite, [1.0, 99.0])
    if not np.isfinite([vmin, vmax]).all():
        raise ValueError("LGE bullseye display range is not finite")
    if vmax <= vmin:
        delta = max(abs(float(vmin)) * 0.01, 0.5)
        vmin, vmax = float(vmin - delta), float(vmax + delta)

    figure, axis = plt.subplots(figsize=(8, 8))
    radial_edges = _centers_to_edges(np.asarray(result.radial_positions))
    continuous_apex = bool(result.radial_positions[0] < 1.0)
    radial_edges[0], radial_edges[-1] = (
        0.0 if continuous_apex else 1.0,
        4.0,
    )
    angle_step = 2.0 * np.pi / len(result.angles)
    plot_angle_edges = np.arange(len(result.angles) + 1) * angle_step + np.pi
    angle_grid, radius_grid = np.meshgrid(plot_angle_edges, radial_edges)
    mesh = axis.pcolormesh(
        radius_grid * np.cos(angle_grid),
        radius_grid * np.sin(angle_grid),
        values,
        cmap=colormap,
        vmin=vmin,
        vmax=vmax,
        shading="flat",
        antialiased=False,
        rasterized=True,
    )
    if (
        result.include_apex
        and not continuous_apex
        and np.isfinite(result.apex_intensity)
    ):
        fraction = np.clip((result.apex_intensity - vmin) / (vmax - vmin), 0.0, 1.0)
        apex_color = plt.get_cmap(colormap)(fraction)
    elif result.include_apex and not continuous_apex:
        apex_color = "#d0d0d0"
    if result.include_apex and not continuous_apex:
        axis.add_patch(
            Circle((0.0, 0.0), 1.0, facecolor=apex_color, edgecolor="none")
        )
    draw_aha17_boundaries(
        axis, show_numbers=show_numbers, include_apex=result.include_apex
    )

    maximum_coronary_radius = _draw_projected_coronaries(
        axis,
        coronary_paths,
        coronary_label_colors=coronary_label_colors,
        coronary_label_names=coronary_label_names,
        coronary_default_color=coronary_default_color,
        coronary_linewidth=coronary_linewidth,
    )
    view_limit = max(4.45, maximum_coronary_radius * 1.05 + 0.10)
    axis.set_xlim(-view_limit, view_limit)
    axis.set_ylim(-view_limit, view_limit)

    colorbar = figure.colorbar(mesh, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label)
    axis.set_title(title, pad=16)

    if output_path is not None:
        _save_figure_overwriting(figure, output_path)
    if show:
        plt.show()
    else:
        plt.close(figure)
    return figure, axis


def plot_lge_scar_bullseye(
    summary: AhaScarSummary,
    output_path: str | Path | None = None,
    *,
    title: str | None = None,
    colormap: str | Colormap | None = None,
    include_apex: bool = True,
    coronary_paths: list[ProjectedCoronaryPath] | None = None,
    coronary_label_colors: dict[int, str] | None = None,
    coronary_label_names: dict[int, str] | None = None,
    coronary_default_color: str | None = None,
    coronary_linewidth: float = 1.8,
    show: bool = True,
):
    """Render native-voxel scar percentages for 16 or 17 AHA segments."""

    percentages = np.asarray(summary.scar_percentages, dtype=float)
    if percentages.shape not in ((16,), (17,)):
        raise ValueError("scar_percentages must contain 16 or 17 AHA segments")
    measured_apex = percentages.size == 17
    if measured_apex and not include_apex:
        raise ValueError("include_apex must be True for a 17-value scar summary")
    if np.any(np.isfinite(percentages) & ((percentages < 0.0) | (percentages > 100.0))):
        raise ValueError("scar percentages must lie between 0 and 100")

    figure, axis = plt.subplots(figsize=(8, 8))
    normalization = Normalize(vmin=0.0, vmax=100.0)
    color_map = (
        _scar_heat_colormap()
        if colormap is None
        else plt.get_cmap(colormap) if isinstance(colormap, str) else colormap
    )

    rings = (
        (0, SIX_SECTOR_CENTERS_DEG, 4.0, 1.0, 30.0, 3.5),
        (6, SIX_SECTOR_CENTERS_DEG, 3.0, 1.0, 30.0, 2.5),
        (12, FOUR_SECTOR_CENTERS_DEG, 2.0, 1.0, 45.0, 1.5),
    )
    for start_index, centers, radius, width, half_angle, text_radius in rings:
        for offset, center_deg in enumerate(centers):
            segment_index = start_index + offset
            value = percentages[segment_index]
            color = (
                "#d7d7d7" if not np.isfinite(value) else color_map(normalization(value))
            )
            axis.add_patch(
                Wedge(
                    (0.0, 0.0),
                    radius,
                    float(center_deg - half_angle),
                    float(center_deg + half_angle),
                    width=width,
                    facecolor=color,
                    edgecolor="none",
                )
            )
            angle = np.deg2rad(center_deg)
            label = (
                f"{segment_index + 1}\nN/A"
                if not np.isfinite(value)
                else f"{segment_index + 1}\n{value:.1f}%"
            )
            red, green, blue = to_rgb(color)
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            axis.text(
                text_radius * np.cos(angle),
                text_radius * np.sin(angle),
                label,
                ha="center",
                va="center",
                fontsize=9,
                color="black" if luminance > 0.55 else "white",
            )

    if include_apex:
        apex_value = percentages[16] if measured_apex else np.nan
        apex_color = (
            "white" if not measured_apex
            else "#d7d7d7" if not np.isfinite(apex_value)
            else color_map(normalization(apex_value))
        )
        axis.add_patch(
            Circle((0.0, 0.0), 1.0, facecolor=apex_color, edgecolor="none")
        )
    draw_aha17_boundaries(axis, show_numbers=False, include_apex=include_apex)
    if include_apex:
        apex_label = (
            "N/A" if not measured_apex
            else "17\nN/A" if not np.isfinite(apex_value)
            else f"17\n{apex_value:.1f}%"
        )
        red, green, blue = to_rgb(apex_color)
        luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
        axis.text(
            0.0, 0.0, apex_label, ha="center", va="center", fontsize=11,
            color="black" if luminance > 0.55 else "white",
        )
    maximum_coronary_radius = _draw_projected_coronaries(
        axis,
        coronary_paths,
        coronary_label_colors=coronary_label_colors,
        coronary_label_names=coronary_label_names,
        coronary_default_color=coronary_default_color,
        coronary_linewidth=coronary_linewidth,
    )
    view_limit = max(4.45, maximum_coronary_radius * 1.05 + 0.10)
    axis.set_xlim(-view_limit, view_limit)
    axis.set_ylim(-view_limit, view_limit)
    axis.set_title(
        f"AHA-{percentages.size} infarct scar burden" if title is None else title,
        pad=16,
    )
    colorbar = figure.colorbar(
        ScalarMappable(norm=normalization, cmap=color_map),
        ax=axis,
        fraction=0.046,
        pad=0.04,
    )
    colorbar.set_label("Scar burden (%)")

    if output_path is not None:
        _save_figure_overwriting(figure, output_path)
    if show:
        plt.show()
    else:
        plt.close(figure)
    return figure, axis
