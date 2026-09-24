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
from matplotlib.patches import Circle
from matplotlib.patches import Wedge

from ..territory.aha17 import (
    FOUR_SECTOR_CENTERS_DEG,
    SIX_SECTOR_CENTERS_DEG,
    draw_aha17_boundaries,
)
from .mapping import LgeBullseyeResult
from .scar import Aha16ScarSummary


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


def plot_lge_bullseye(
    result: LgeBullseyeResult,
    output_path: str | Path | None = None,
    *,
    title: str = "Continuous LGE intensity bullseye",
    colormap: str = "gray",
    show_numbers: bool = True,
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
    radial_edges[0], radial_edges[-1] = 1.0, 4.0
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
    if result.include_apex and np.isfinite(result.apex_intensity):
        fraction = np.clip((result.apex_intensity - vmin) / (vmax - vmin), 0.0, 1.0)
        apex_color = plt.get_cmap(colormap)(fraction)
    elif result.include_apex:
        apex_color = "#d0d0d0"
    if result.include_apex:
        axis.add_patch(
            Circle((0.0, 0.0), 1.0, facecolor=apex_color, edgecolor="none")
        )
    draw_aha17_boundaries(
        axis, show_numbers=show_numbers, include_apex=result.include_apex
    )
    colorbar = figure.colorbar(mesh, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("LGE intensity (native units)")
    axis.set_title(title, pad=16)

    if output_path is not None:
        _save_figure_overwriting(figure, output_path)
    if show:
        plt.show()
    else:
        plt.close(figure)
    return figure, axis


def plot_lge_scar_bullseye(
    summary: Aha16ScarSummary,
    output_path: str | Path | None = None,
    *,
    title: str = "AHA-16 infarct scar burden",
    colormap: str | Colormap | None = None,
    include_apex: bool = True,
    show: bool = True,
):
    """Render native-voxel scar percentage for AHA segments 1 through 16."""

    percentages = np.asarray(summary.scar_percentages, dtype=float)
    if percentages.shape != (16,):
        raise ValueError("scar_percentages must contain AHA segments 1 through 16")
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
        axis.add_patch(
            Circle((0.0, 0.0), 1.0, facecolor="white", edgecolor="none")
        )
    draw_aha17_boundaries(axis, show_numbers=False, include_apex=include_apex)
    if include_apex:
        axis.text(0.0, 0.0, "N/A", ha="center", va="center", fontsize=11)
    axis.set_title(title, pad=16)
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
