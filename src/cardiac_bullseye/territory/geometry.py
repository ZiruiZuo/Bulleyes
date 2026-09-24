from __future__ import annotations

from collections.abc import Iterable

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


def estimate_lv_centers(
    myocardium_masks: ArrayLike,
    *,
    center_overrides: Iterable[tuple[float, float] | None] | None = None,
) -> NDArray[np.float64]:
    """Estimate one LV cavity center per nonempty short-axis slice.

    A slice whose myocardium encloses a cavity uses its own cavity centroid.
    An open slice copies the center of the nearest closed slice.  Equal-distance
    ties are resolved toward the larger (more basal) slice index.  Empty slices
    are returned as ``(nan, nan)`` and do not act as reference slices.
    """

    masks = np.asarray(myocardium_masks, dtype=bool)
    if masks.ndim != 3:
        raise ValueError("myocardium_masks must have shape (slice, height, width)")

    centers = np.full((masks.shape[0], 2), np.nan, dtype=float)
    nonempty_indices: list[int] = []
    closed_indices: list[int] = []
    for slice_index, myocardium in enumerate(masks):
        if not myocardium.any():
            continue
        nonempty_indices.append(slice_index)
        try:
            centers[slice_index] = infer_lv_cavity_center(myocardium)
        except ValueError:
            continue
        closed_indices.append(slice_index)

    if nonempty_indices and not closed_indices:
        raise ValueError(
            "No myocardial slice contains a closed LV cavity; "
            "at least one closed LV cavity is required"
        )

    for slice_index in nonempty_indices:
        if np.isfinite(centers[slice_index]).all():
            continue
        reference_index = min(
            closed_indices,
            key=lambda candidate: (abs(candidate - slice_index), -candidate),
        )
        centers[slice_index] = centers[reference_index]

    if center_overrides is not None:
        overrides = list(center_overrides)
        if len(overrides) != masks.shape[0]:
            raise ValueError("center_overrides must have one entry per slice")
        height, width = masks.shape[1:]
        for slice_index, override in enumerate(overrides):
            if override is None:
                continue
            if not masks[slice_index].any():
                raise ValueError(
                    f"Center override on slice {slice_index + 1} requires a "
                    "nonempty myocardium mask"
                )
            value = np.asarray(override, dtype=float)
            if value.shape != (2,) or not np.isfinite(value).all():
                raise ValueError(
                    "Each center override must contain two finite values"
                )
            x, y = map(float, value)
            if not (0.0 <= x < width and 0.0 <= y < height):
                raise ValueError(
                    f"Center override on slice {slice_index + 1} is outside "
                    "the image"
                )
            centers[slice_index] = (x, y)
    return centers


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
