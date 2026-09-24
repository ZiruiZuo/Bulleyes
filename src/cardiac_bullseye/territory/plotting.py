from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch

from ..coronary.projection import ProjectedCoronaryPath
from .aha17 import draw_aha17_boundaries
from .mapping import TERRITORY_COLORS, TERRITORY_NAMES
from .models import BullseyeResult


CORONARY_DEFAULT_LABEL_COLORS = {
    1: "#d62728",  # LAD: red
    2: "#ff7f0e",  # LCX: orange; distinct from blue LCX territory
    3: "#9467bd",  # RCA: purple
}

CORONARY_AUTO_COLORS = (
    "#d62728",  # red
    "#ff7f0e",  # orange
    "#9467bd",  # purple
    "#17becf",  # cyan
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
)


def _centers_to_edges(values: np.ndarray) -> np.ndarray:
    if values.size < 2:
        raise ValueError("At least two centers are required")
    midpoints = (values[:-1] + values[1:]) / 2.0
    first = values[0] - (midpoints[0] - values[0])
    last = values[-1] + (values[-1] - midpoints[-1])
    return np.concatenate([[first], midpoints, [last]])


def plot_territory_bullseye(
    result: BullseyeResult,
    output_path: str | Path | None = None,
    *,
    title: str = "Patient-specific territory with AHA-17 overlay",
    show_numbers: bool = True,
    show_landmark_markers: bool = True,
    coronary_paths: list[ProjectedCoronaryPath] | None = None,
    coronary_label_colors: dict[int, str] | None = None,
    coronary_label_names: dict[int, str] | None = None,
    coronary_default_color: str | None = None,
    coronary_linewidth: float = 1.8,
    show: bool = True,
):
    """Plot a categorical territory map under canonical AHA-17 boundaries."""

    figure, axis = plt.subplots(figsize=(8, 8))
    radial_edges = _centers_to_edges(np.asarray(result.radial_positions))
    radial_edges[0] = 1.0
    radial_edges[-1] = 4.0

    angle_step = 2.0 * np.pi / result.angles.size
    canonical_angle_edges = np.arange(result.angles.size + 1) * angle_step
    plot_angle_edges = canonical_angle_edges + np.pi
    angle_grid, radius_grid = np.meshgrid(plot_angle_edges, radial_edges)
    x_edges = radius_grid * np.cos(angle_grid)
    y_edges = radius_grid * np.sin(angle_grid)

    cmap = ListedColormap(TERRITORY_COLORS)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    axis.pcolormesh(
        x_edges,
        y_edges,
        result.territory_labels,
        cmap=cmap,
        norm=norm,
        shading="flat",
        antialiased=False,
        rasterized=True,
    )

    center_color = TERRITORY_COLORS[result.apex_territory]
    axis.add_patch(Circle((0.0, 0.0), 1.0, facecolor=center_color, edgecolor="none"))

    # Patient-specific territory borders. Drawing one binary contour per label
    # avoids treating category numbers as continuous measurements.
    plot_angles = result.angles + np.pi
    angle_centers, radius_centers = np.meshgrid(
        plot_angles, result.radial_positions
    )
    x_centers = radius_centers * np.cos(angle_centers)
    y_centers = radius_centers * np.sin(angle_centers)
    for territory_id in (1, 2, 3):
        binary = (result.territory_labels == territory_id).astype(float)
        if binary.min() != binary.max():
            axis.contour(
                x_centers,
                y_centers,
                binary,
                levels=[0.5],
                colors=["#333333"],
                linewidths=1.1,
            )

    draw_aha17_boundaries(axis, show_numbers=show_numbers)

    coronary_paths = coronary_paths or []
    coronary_label_colors = coronary_label_colors or {}
    coronary_label_names = coronary_label_names or {}
    coronary_labels = sorted({path.label for path in coronary_paths})
    resolved_coronary_colors = {
        label: coronary_label_colors.get(
            label,
            coronary_default_color
            if coronary_default_color is not None
            else CORONARY_DEFAULT_LABEL_COLORS.get(
                label,
                CORONARY_AUTO_COLORS[index % len(CORONARY_AUTO_COLORS)],
            ),
        )
        for index, label in enumerate(coronary_labels)
    }
    for path in coronary_paths:
        color = resolved_coronary_colors[path.label]
        axis.plot(
            path.xy[:, 0],
            path.xy[:, 1],
            color=color,
            linewidth=coronary_linewidth,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=20,
        )

    # AHA drawing fixes the view to +/-4.2. Expand it when a projected vessel
    # lies above the basal plane (radius > 4), so the complete path is visible.
    finite_radii = [
        np.linalg.norm(path.xy[np.isfinite(path.xy).all(axis=1)], axis=1)
        for path in coronary_paths
        if np.isfinite(path.xy).all(axis=1).any()
    ]
    maximum_coronary_radius = max(
        (float(values.max()) for values in finite_radii if values.size),
        default=0.0,
    )
    view_limit = max(4.45, maximum_coronary_radius * 1.05 + 0.10)
    axis.set_xlim(-view_limit, view_limit)
    axis.set_ylim(-view_limit, view_limit)

    if show_landmark_markers:
        for angle_deg, label in ((120.0, "AIP"), (240.0, "IIP")):
            angle = np.deg2rad(angle_deg)
            axis.plot(
                4.0 * np.cos(angle),
                4.0 * np.sin(angle),
                "o",
                markerfacecolor="#ff4fd8",
                markeredgecolor="black",
                markersize=9,
                zorder=10,
            )
            axis.text(
                4.18 * np.cos(angle),
                4.18 * np.sin(angle),
                label,
                ha="center",
                va="center",
                fontsize=9,
            )

    legend_handles = [
        Patch(facecolor=TERRITORY_COLORS[index], edgecolor="black", label=name)
        for index, name in ((1, "LAD"), (2, "LCX"), (3, "RCA"))
    ]
    if result.apex_territory == 0:
        legend_handles.append(
            Patch(
                facecolor=TERRITORY_COLORS[0],
                edgecolor="black",
                label=TERRITORY_NAMES[0],
            )
        )
    for label in coronary_labels:
        color = resolved_coronary_colors[label]
        name = coronary_label_names.get(
            label,
            "Coronary" if len(coronary_labels) == 1 else f"Coronary {label}",
        )
        legend_handles.append(
            Line2D([0], [0], color=color, linewidth=coronary_linewidth, label=name)
        )
    axis.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
    )
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


def save_bullseye_figure(
    result: BullseyeResult,
    output_path: str | Path,
    *,
    title: str = "Patient-specific territory with AHA-17 overlay",
    coronary_paths: list[ProjectedCoronaryPath] | None = None,
    coronary_label_colors: dict[int, str] | None = None,
    coronary_label_names: dict[int, str] | None = None,
    coronary_default_color: str | None = None,
    coronary_linewidth: float = 1.8,
    show: bool = False,
) -> Path:
    """Save only the final bullseye PNG."""

    output_path = Path(output_path)
    if output_path.suffix.lower() != ".png":
        output_path = output_path.with_suffix(".png")
    plot_territory_bullseye(
        result,
        output_path,
        title=title,
        coronary_paths=coronary_paths,
        coronary_label_colors=coronary_label_colors,
        coronary_label_names=coronary_label_names,
        coronary_default_color=coronary_default_color,
        coronary_linewidth=coronary_linewidth,
        show=show,
    )
    return output_path
