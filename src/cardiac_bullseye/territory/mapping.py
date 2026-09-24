from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage

from .aha17 import aha17_segment_id_map
from .geometry import (
    aligned_angles,
    estimate_lv_centers,
    normalize_six_segment_angles,
)
from .models import BullseyeResult, LandmarkPair, StudyData


TERRITORY_NAMES = {0: "Unknown", 1: "LAD", 2: "LCX", 3: "RCA"}
TERRITORY_COLORS = np.asarray(
    [
        [0.82, 0.82, 0.82],  # unknown: gray
        [0.20, 0.75, 0.30],  # LAD: green
        [0.35, 0.60, 0.90],  # LCX: blue
        [1.00, 0.92, 0.30],  # RCA: yellow
    ],
    dtype=float,
)


def _periodic_fill_missing(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Fill missing values in a 1-D circular signal by linear interpolation."""

    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    if not valid.any():
        return values.copy()
    if valid.sum() == 1:
        return np.full_like(values, values[valid][0])

    indices = np.arange(values.size)
    valid_indices = indices[valid]
    valid_values = values[valid]
    extended_indices = np.concatenate(
        [valid_indices - values.size, valid_indices, valid_indices + values.size]
    )
    extended_values = np.tile(valid_values, 3)
    return np.interp(indices, extended_indices, extended_values)


def _normalize_probabilities(
    probabilities: NDArray[np.float64],
) -> NDArray[np.float64]:
    probabilities = np.clip(probabilities, 0.0, None)
    denominator = np.sum(probabilities, axis=-1, keepdims=True)
    return np.divide(
        probabilities,
        denominator,
        out=np.full_like(probabilities, np.nan),
        where=denominator > 0,
    )


def compute_slice_territory_probabilities(
    study: StudyData,
    landmarks: list[LandmarkPair | None],
    *,
    angle_count: int = 360,
    angular_smoothing_sigma_deg: float = 1.5,
    six_segment_warp: bool = True,
    center_overrides: Iterable[tuple[float, float] | None] | None = None,
) -> NDArray[np.float64]:
    """Convert territory masks to per-slice circular probabilities.

    The result has shape ``(slice, angle, 3)`` in LAD, LCX, RCA order. Missing
    landmarks leave an entire slice as NaN so later radial interpolation can
    safely ignore it.
    """

    if len(landmarks) != study.slice_count:
        raise ValueError("There must be one landmark entry per slice")
    if angle_count < 4:
        raise ValueError("angle_count must be at least 4")
    if angular_smoothing_sigma_deg < 0:
        raise ValueError("angular_smoothing_sigma_deg cannot be negative")

    result = np.full((study.slice_count, angle_count, 3), np.nan, dtype=float)
    sigma_bins = angular_smoothing_sigma_deg / 360.0 * angle_count
    centers = estimate_lv_centers(
        study.territory_mask > 0,
        center_overrides=center_overrides,
    )

    for slice_index, pair in enumerate(landmarks):
        if pair is None:
            continue

        labels_2d = study.territory_mask[slice_index]
        myocardium = labels_2d > 0
        if not myocardium.any():
            continue

        center = tuple(centers[slice_index])
        y, x = np.nonzero(myocardium)
        labels = labels_2d[y, x].astype(int)
        angles = aligned_angles(x, y, center, pair.aip, pair.iip)
        if six_segment_warp:
            angles = normalize_six_segment_angles(
                angles, center, pair.aip, pair.iip
            )

        angle_bins = np.floor(angles / (2.0 * np.pi) * angle_count).astype(int)
        angle_bins = np.clip(angle_bins, 0, angle_count - 1)

        counts = np.zeros((angle_count, 3), dtype=float)
        for territory_id in (1, 2, 3):
            np.add.at(
                counts[:, territory_id - 1],
                angle_bins[labels == territory_id],
                1.0,
            )

        total = counts.sum(axis=1)
        probabilities = np.divide(
            counts,
            total[:, None],
            out=np.full_like(counts, np.nan),
            where=total[:, None] > 0,
        )
        for territory_index in range(3):
            probabilities[:, territory_index] = _periodic_fill_missing(
                probabilities[:, territory_index]
            )

        if sigma_bins > 0:
            probabilities = ndimage.gaussian_filter1d(
                probabilities,
                sigma=sigma_bins,
                axis=0,
                mode="wrap",
            )
        result[slice_index] = _normalize_probabilities(probabilities)

    return result


def _interpolate_slice_probabilities(
    slice_probabilities: NDArray[np.float64],
    source_radii: NDArray[np.float64],
    target_radii: NDArray[np.float64],
) -> NDArray[np.float64]:
    angle_count = slice_probabilities.shape[1]
    output = np.full((target_radii.size, angle_count, 3), np.nan, dtype=float)

    for angle_index in range(angle_count):
        for territory_index in range(3):
            values = slice_probabilities[:, angle_index, territory_index]
            valid = np.isfinite(values)
            if valid.sum() == 1:
                output[:, angle_index, territory_index] = values[valid][0]
            elif valid.sum() >= 2:
                output[:, angle_index, territory_index] = np.interp(
                    target_radii,
                    source_radii[valid],
                    values[valid],
                )
    return output


def remove_deleted_slices_from_radial_axis(
    values: ArrayLike,
    source_radii: ArrayLike,
    deleted_slice_indices: Iterable[int],
    *,
    rescale_retained_base: bool = True,
) -> tuple[NDArray, NDArray[np.float64]]:
    """Remove deleted slices and optionally map the retained base to radius 4."""

    data = np.asarray(values)
    source = np.asarray(source_radii, dtype=float)
    if data.ndim == 0 or data.shape[0] != source.size:
        raise ValueError("values and source_radii must have the same slice count")
    deleted = tuple(int(index) for index in deleted_slice_indices)
    retained_basal_index = last_retained_slice_index(source.size, deleted)
    if retained_basal_index == source.size - 1:
        return data, source

    retained = slice(0, retained_basal_index + 1)
    retained_source = source[retained].copy()
    if rescale_retained_base:
        basal_radius = float(retained_source[-1])
        if basal_radius > 0.0:
            retained_source *= 4.0 / basal_radius
        else:
            retained_source[-1] = 4.0
    return data[retained], retained_source


def last_retained_slice_index(
    slice_count: int,
    deleted_slice_indices: Iterable[int],
) -> int:
    """Validate a contiguous basal deletion and return its retained boundary."""

    if slice_count < 1:
        raise ValueError("slice_count must be positive")
    deleted = {int(index) for index in deleted_slice_indices}
    invalid = sorted(index for index in deleted if index < 0 or index >= slice_count)
    if invalid:
        raise ValueError(f"Deleted slice indices are out of range: {invalid}")
    if not deleted:
        return slice_count - 1
    first_deleted = min(deleted)
    if deleted != set(range(first_deleted, slice_count)):
        raise ValueError("Deleted slices must form one contiguous basal tail")
    if first_deleted == 0:
        raise ValueError("At least one slice must remain after deletion")
    return first_deleted - 1


def infer_apex_territory(
    probabilities: ArrayLike,
    radial_positions: ArrayLike,
) -> int:
    """Infer segment 17 from aggregate probabilities in AHA segments 13-16."""

    values = np.asarray(probabilities, dtype=float)
    radii = np.asarray(radial_positions, dtype=float)
    if values.ndim != 3 or values.shape[0] != radii.size or values.shape[2] != 3:
        raise ValueError(
            "probabilities and radial_positions have incompatible shapes"
        )
    selected = values[(radii >= 1.0) & (radii < 2.0)]
    finite_counts = np.isfinite(selected).sum(axis=(0, 1))
    scores = np.divide(
        np.nansum(selected, axis=(0, 1)),
        finite_counts,
        out=np.full(3, np.nan, dtype=float),
        where=finite_counts > 0,
    )
    if not np.isfinite(scores).any():
        return 0
    return int(np.nanargmax(scores)) + 1


def build_bullseye(
    slice_probabilities: ArrayLike,
    slice_positions: ArrayLike,
    *,
    source_radii: ArrayLike | None = None,
    radial_count: int = 300,
    radial_smoothing_sigma: float = 1.5,
    apex_territory: int | None = None,
    deleted_slice_indices: Iterable[int] = (),
    rescale_deleted_base: bool = True,
) -> BullseyeResult:
    """Interpolate slice probabilities into the annular AHA bullseye domain."""

    probabilities = np.asarray(slice_probabilities, dtype=float)
    positions = np.asarray(slice_positions, dtype=float)
    if probabilities.ndim != 3 or probabilities.shape[2] != 3:
        raise ValueError(
            "slice_probabilities must have shape (slice, angle, territory=3)"
        )
    if positions.shape != (probabilities.shape[0],):
        raise ValueError("slice_positions must have one value per slice")
    if not np.all(np.diff(positions) > 0):
        raise ValueError("slice_positions must increase from apex to base")
    if radial_count < 3:
        raise ValueError("radial_count must be at least 3")
    if radial_smoothing_sigma < 0:
        raise ValueError("radial_smoothing_sigma cannot be negative")
    if not np.isfinite(probabilities).any():
        raise ValueError("No valid slice probabilities are available")

    if source_radii is not None:
        source_radii_array = np.asarray(source_radii, dtype=float)
        if source_radii_array.shape != positions.shape:
            raise ValueError("source_radii must have one value per slice")
        if not np.isfinite(source_radii_array).all():
            raise ValueError("source_radii must contain only finite values")
        if source_radii_array.size > 1 and not np.all(
            np.diff(source_radii_array) > 0
        ):
            raise ValueError("source_radii must increase from apex to base")
    elif positions.size == 1 or np.isclose(positions[-1], positions[0]):
        source_radii_array = np.asarray([2.5])
    else:
        normalized_positions = (positions - positions[0]) / (
            positions[-1] - positions[0]
        )
        source_radii_array = 1.0 + 3.0 * normalized_positions

    radial_positions = np.linspace(1.0, 4.0, radial_count)
    radial_probabilities, radial_source_radii = (
        remove_deleted_slices_from_radial_axis(
            probabilities,
            source_radii_array,
            deleted_slice_indices,
            rescale_retained_base=rescale_deleted_base,
        )
    )
    if not np.isfinite(radial_probabilities).any():
        raise ValueError("No valid slice probabilities remain after deletion")
    dense_probabilities = _interpolate_slice_probabilities(
        radial_probabilities, radial_source_radii, radial_positions
    )
    if radial_smoothing_sigma > 0:
        dense_probabilities = ndimage.gaussian_filter1d(
            dense_probabilities,
            sigma=radial_smoothing_sigma,
            axis=0,
            mode="nearest",
        )
    dense_probabilities = _normalize_probabilities(dense_probabilities)

    valid = np.isfinite(dense_probabilities).any(axis=2)
    labels = np.zeros(valid.shape, dtype=np.uint8)
    labels[valid] = np.nanargmax(dense_probabilities[valid], axis=1) + 1

    angle_count = probabilities.shape[1]
    angles = (np.arange(angle_count, dtype=float) + 0.5) * 2.0 * np.pi / angle_count
    segment_ids = aha17_segment_id_map(radial_positions, angles)

    if apex_territory is None:
        apex_territory = infer_apex_territory(
            dense_probabilities, radial_positions
        )
    if apex_territory not in {0, 1, 2, 3}:
        raise ValueError("apex_territory must be 0, 1, 2, or 3")

    return BullseyeResult(
        probabilities=dense_probabilities,
        territory_labels=labels,
        aha_segment_ids=segment_ids,
        radial_positions=radial_positions,
        angles=angles,
        apex_territory=int(apex_territory),
    )


def compute_bullseye(
    study: StudyData,
    landmarks: list[LandmarkPair | None],
    *,
    angle_count: int = 360,
    radial_count: int = 300,
    angular_smoothing_sigma_deg: float = 1.5,
    radial_smoothing_sigma: float = 1.5,
    six_segment_warp: bool = True,
    apex_territory: int | None = None,
    source_radii: ArrayLike | None = None,
    deleted_slice_indices: Iterable[int] = (),
    rescale_deleted_base: bool = True,
    center_overrides: Iterable[tuple[float, float] | None] | None = None,
) -> BullseyeResult:
    """Run the complete numerical pipeline without plotting or file output."""

    slice_probabilities = compute_slice_territory_probabilities(
        study,
        landmarks,
        angle_count=angle_count,
        angular_smoothing_sigma_deg=angular_smoothing_sigma_deg,
        six_segment_warp=six_segment_warp,
        center_overrides=center_overrides,
    )
    return build_bullseye(
        slice_probabilities,
        study.slice_positions,
        source_radii=source_radii,
        radial_count=radial_count,
        radial_smoothing_sigma=radial_smoothing_sigma,
        apex_territory=apex_territory,
        deleted_slice_indices=deleted_slice_indices,
        rescale_deleted_base=rescale_deleted_base,
    )


def compute_aha17_territory_percentages(
    result: BullseyeResult,
) -> NDArray[np.float64]:
    """Return a ``(17, 3)`` table of LAD/LCX/RCA percentages by AHA segment."""

    table = np.full((17, 3), np.nan, dtype=float)
    for segment_id in range(1, 17):
        selected = result.aha_segment_ids == segment_id
        if selected.any():
            table[segment_id - 1] = (
                np.nanmean(result.probabilities[selected], axis=0) * 100.0
            )

    if result.apex_territory > 0:
        table[16] = 0.0
        table[16, result.apex_territory - 1] = 100.0
    return table
