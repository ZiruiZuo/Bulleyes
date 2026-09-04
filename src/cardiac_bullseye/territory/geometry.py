from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage


def largest_connected_component(mask: ArrayLike) -> NDArray[np.bool_]:
    """Return the largest 2-D connected component of a binary mask."""

    binary = np.asarray(mask, dtype=bool)
    labels, count = ndimage.label(binary)
    if count == 0:
        return np.zeros_like(binary)

    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def infer_lv_cavity_center(myocardium_mask: ArrayLike) -> tuple[float, float]:
    """Infer the LV cavity centroid from a closed ring-like myocardium mask.

    Returns ``(x, y)``. For clinical data, a dedicated LV cavity mask is more
    reliable than this convenience method.
    """

    myocardium = largest_connected_component(myocardium_mask)
    if not myocardium.any():
        raise ValueError("The myocardium mask is empty")

    filled = ndimage.binary_fill_holes(myocardium)
    cavity = largest_connected_component(filled & ~myocardium)
    if not cavity.any():
        raise ValueError(
            "No enclosed LV cavity was found; provide a closed myocardium mask "
            "or a dedicated LV cavity mask"
        )

    y, x = ndimage.center_of_mass(cavity)
    return float(x), float(y)


def estimate_lv_center(myocardium_mask: ArrayLike) -> tuple[float, float]:
    """Estimate the LV center, with a centroid fallback for an open ring."""

    myocardium = largest_connected_component(myocardium_mask)
    if not myocardium.any():
        raise ValueError("The myocardium mask is empty")
    try:
        return infer_lv_cavity_center(myocardium)
    except ValueError:
        y, x = ndimage.center_of_mass(myocardium)
        return float(x), float(y)


def aligned_angles(
    x: ArrayLike,
    y: ArrayLike,
    center_xy: tuple[float, float],
    aip_xy: tuple[float, float],
    iip_xy: tuple[float, float],
) -> NDArray[np.float64]:
    """Rotate/reflect pixel angles so IIP is on the positive-angle side.

    Zero is the midpoint of the shorter AIP-IIP arc. The result is in
    ``[0, 2*pi)``. Six-sector anatomical warping is intentionally left for the
    next teaching step.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    cx, cy = center_xy

    aip_angle = np.arctan2(aip_xy[1] - cy, aip_xy[0] - cx)
    iip_angle = np.arctan2(iip_xy[1] - cy, iip_xy[0] - cx)
    difference = np.arctan2(
        np.sin(iip_angle - aip_angle), np.cos(iip_angle - aip_angle)
    )
    if np.isclose(difference, 0.0):
        raise ValueError("AIP and IIP have indistinguishable angles")

    zero_angle = aip_angle + difference / 2.0
    result = np.mod(np.arctan2(y - cy, x - cx) - zero_angle, 2.0 * np.pi)
    if difference < 0:
        result = np.mod(-result, 2.0 * np.pi)
    return result


def normalize_six_segment_angles(
    relative_angles: ArrayLike,
    center_xy: tuple[float, float],
    aip_xy: tuple[float, float],
    iip_xy: tuple[float, float],
) -> NDArray[np.float64]:
    """Warp six anatomical intervals to six equal 60-degree intervals.

    ``relative_angles`` must already be returned by :func:`aligned_angles`.
    The transformation mirrors the MATLAB reference implementation and fixes
    the normalized IIP and AIP positions at 60 and 300 degrees respectively.
    """

    cx, cy = center_xy
    aip_angle = np.arctan2(aip_xy[1] - cy, aip_xy[0] - cx)
    iip_angle = np.arctan2(iip_xy[1] - cy, iip_xy[0] - cx)
    difference = np.arctan2(
        np.sin(iip_angle - aip_angle), np.cos(iip_angle - aip_angle)
    )
    separation = min(abs(float(difference)), np.pi - np.finfo(float).eps)
    if separation <= np.finfo(float).eps:
        raise ValueError("AIP and IIP have indistinguishable angles")

    return normalize_six_segment_relative_angles(relative_angles, separation)


def normalize_six_segment_relative_angles(
    relative_angles: ArrayLike,
    separations: ArrayLike,
) -> NDArray[np.float64]:
    """Vectorised six-interval warp for already aligned relative angles."""

    angles, separation = np.broadcast_arrays(
        np.mod(np.asarray(relative_angles, dtype=float), 2.0 * np.pi),
        np.asarray(separations, dtype=float),
    )
    separation = np.clip(separation, np.finfo(float).eps, np.pi - np.finfo(float).eps)
    boundaries = (
        np.zeros_like(separation),
        separation / 2.0,
        np.pi - separation / 2.0,
        np.full_like(separation, np.pi),
        np.pi + separation / 2.0,
        2.0 * np.pi - separation / 2.0,
        np.full_like(separation, 2.0 * np.pi),
    )
    output = np.empty_like(angles)
    for segment in range(6):
        lower = boundaries[segment]
        upper = boundaries[segment + 1]
        selected = (angles >= lower) & (
            (angles < upper) if segment < 5 else (angles <= upper)
        )
        fraction = np.divide(
            angles - lower,
            upper - lower,
            out=np.zeros_like(angles),
            where=(upper - lower) > 0,
        )
        output[selected] = (segment + fraction[selected]) * np.pi / 3.0
    return output
