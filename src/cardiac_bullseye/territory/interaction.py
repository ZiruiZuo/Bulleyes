"""Shared image-navigation calculations for interactive annotations."""

from __future__ import annotations

import numpy as np


def zoom_limits_about_point(
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    point_xy: tuple[float, float],
    *,
    scale: float,
    image_shape: tuple[int, int],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Scale axes limits about a fixed image coordinate, preserving y direction."""

    values = np.asarray((*xlim, *ylim, *point_xy, scale), dtype=float)
    if not np.isfinite(values).all() or scale <= 0.0:
        raise ValueError("Zoom limits, point, and scale must be finite and valid")
    if len(image_shape) != 2 or any(size <= 0 for size in image_shape):
        raise ValueError("image_shape must contain two positive dimensions")

    def zoom_axis(
        limits: tuple[float, float], point: float, full_span: float
    ) -> tuple[float, float]:
        first, second = limits
        if abs(second - first) * scale >= full_span:
            return (
                (-0.5, full_span - 0.5) if second > first else (full_span - 0.5, -0.5)
            )
        minimum_span = min(2.0, full_span)
        effective_scale = max(scale, minimum_span / abs(second - first))
        return (
            point + (first - point) * effective_scale,
            point + (second - point) * effective_scale,
        )

    return (
        zoom_axis(xlim, float(point_xy[0]), float(image_shape[1])),
        zoom_axis(ylim, float(point_xy[1]), float(image_shape[0])),
    )
