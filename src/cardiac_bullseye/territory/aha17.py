from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import ArrayLike, NDArray


SIX_SECTOR_CENTERS_DEG = np.array([90, 150, 210, 270, 330, 30], dtype=float)
FOUR_SECTOR_CENTERS_DEG = np.array([90, 180, 270, 0], dtype=float)

AHA_SEGMENT_NAMES = {
    1: "basal anterior",
    2: "basal anteroseptal",
    3: "basal inferoseptal",
    4: "basal inferior",
    5: "basal inferolateral",
    6: "basal anterolateral",
    7: "mid anterior",
    8: "mid anteroseptal",
    9: "mid inferoseptal",
    10: "mid inferior",
    11: "mid inferolateral",
    12: "mid anterolateral",
    13: "apical anterior",
    14: "apical septal",
    15: "apical inferior",
    16: "apical lateral",
    17: "apex",
}


def aha17_segment_id_map(
    radial_positions: ArrayLike,
    canonical_angles: ArrayLike,
) -> NDArray[np.uint8]:
    """Return AHA segment IDs for the annular portion of a bullseye.

    Radii 1-2 are apical, 2-3 mid-cavity, and 3-4 basal. Canonical angles
    receive the same 180-degree display rotation used by the MATLAB reference.
    Segment 17 is the separate center cap and is not present in this array.
    """

    radius = np.asarray(radial_positions, dtype=float)
    angle = np.asarray(canonical_angles, dtype=float)
    if radius.ndim != 1 or angle.ndim != 1:
        raise ValueError("radial_positions and canonical_angles must be 1-D")

    plot_degrees = np.mod(np.rad2deg(angle + np.pi), 360.0)
    six_index = np.floor(np.mod(plot_degrees - 60.0, 360.0) / 60.0).astype(int)
    four_index = np.floor(np.mod(plot_degrees - 45.0, 360.0) / 90.0).astype(int)

    result = np.zeros((radius.size, angle.size), dtype=np.uint8)
    apical = (radius >= 1.0) & (radius < 2.0)
    middle = (radius >= 2.0) & (radius < 3.0)
    basal = (radius >= 3.0) & (radius <= 4.0)
    result[apical, :] = (13 + four_index)[None, :]
    result[middle, :] = (7 + six_index)[None, :]
    result[basal, :] = (1 + six_index)[None, :]
    return result


def _draw_radial_line(
    axis: Axes,
    angle_deg: float,
    inner_radius: float,
    outer_radius: float,
) -> None:
    angle = np.deg2rad(angle_deg)
    axis.plot(
        [inner_radius * np.cos(angle), outer_radius * np.cos(angle)],
        [inner_radius * np.sin(angle), outer_radius * np.sin(angle)],
        "k--",
        linewidth=1.25,
    )


def draw_aha17_boundaries(axis: Axes, *, show_numbers: bool = True) -> None:
    """Draw the canonical 6+6+4+1 AHA-17 geometry on an axis."""

    circle_angle = np.linspace(0.0, 2.0 * np.pi, 721)
    for radius in (1.0, 2.0, 3.0, 4.0):
        axis.plot(
            radius * np.cos(circle_angle),
            radius * np.sin(circle_angle),
            "k-",
            linewidth=1.5,
        )

    for angle in (60, 120, 180, 240, 300, 360):
        _draw_radial_line(axis, angle, 2.0, 4.0)
    for angle in (45, 135, 225, 315):
        _draw_radial_line(axis, angle, 1.0, 2.0)

    if show_numbers:
        _draw_segment_numbers(axis)

    axis.set_aspect("equal")
    axis.set_xlim(-4.2, 4.2)
    axis.set_ylim(-4.2, 4.2)
    axis.set_axis_off()


def _draw_segment_numbers(axis: Axes) -> None:
    for radius, segment_start in ((3.5, 1), (2.5, 7)):
        for offset, angle_deg in enumerate(SIX_SECTOR_CENTERS_DEG):
            angle = np.deg2rad(angle_deg)
            axis.text(
                radius * np.cos(angle),
                radius * np.sin(angle),
                str(segment_start + offset),
                ha="center",
                va="center",
                fontsize=11,
            )

    for offset, angle_deg in enumerate(FOUR_SECTOR_CENTERS_DEG):
        angle = np.deg2rad(angle_deg)
        axis.text(
            1.5 * np.cos(angle),
            1.5 * np.sin(angle),
            str(13 + offset),
            ha="center",
            va="center",
            fontsize=11,
        )

    axis.text(0.0, 0.0, "17", ha="center", va="center", fontsize=11)


def plot_aha17_template(
    output_path: str | Path | None = None,
    *,
    show: bool = True,
) -> None:
    """Plot and optionally save an empty AHA-17 bullseye template."""

    figure, axis = plt.subplots(figsize=(7, 7))
    axis.set_facecolor("#f7f7f7")
    draw_aha17_boundaries(axis)
    axis.set_title("AHA 17-segment bullseye")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=180, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(figure)
