"""Plot a coronary-distance heatmap on the exact territory bullseye grid."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from ..territory.aha17 import draw_aha17_boundaries

from .mapping import DistanceBullseyeResult


def _centers_to_edges(values: np.ndarray) -> np.ndarray:
    midpoints = (values[:-1] + values[1:]) / 2.0
    return np.concatenate(
        [
            [values[0] - (midpoints[0] - values[0])],
            midpoints,
            [values[-1] + (values[-1] - midpoints[-1])],
        ]
    )


def plot_distance_bullseye(
    result: DistanceBullseyeResult,
    output_path: str | Path | None = None,
    *,
    title: str = "Distance to nearest projected coronary",
    colormap: str = "turbo",
    value_range_mm: tuple[float, float] | None = None,
    show_numbers: bool = True,
    show: bool = True,
):
    """Draw the distance heatmap; every cell matches the territory result."""

    values = np.asarray(result.distance_mm, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Distance bullseye contains no finite values")
    if value_range_mm is None:
        vmin = 0.0
        vmax = float(np.nanpercentile(finite, 99.0))
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = max(float(np.nanmax(finite)), 1.0)
    else:
        vmin, vmax = map(float, value_range_mm)
        if not np.isfinite([vmin, vmax]).all() or vmax <= vmin:
            raise ValueError("value_range_mm must be two finite increasing values")

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

    apex_value = result.apex_distance_mm
    apex_color = plt.get_cmap(colormap)(
        np.clip((apex_value - vmin) / (vmax - vmin), 0.0, 1.0)
    ) if np.isfinite(apex_value) else "#d0d0d0"
    axis.add_patch(Circle((0.0, 0.0), 1.0, facecolor=apex_color, edgecolor="none"))
    draw_aha17_boundaries(axis, show_numbers=show_numbers)
    colorbar = figure.colorbar(mesh, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Distance to nearest coronary (mm)")
    axis.set_title(title, pad=16)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(figure)
    return figure, axis
