"""Voxel-wise n-SD infarct scar quantification on short-axis LGE."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..territory.aha17 import aha17_segment_id_map
from ..territory.geometry import (
    aligned_angles,
    estimate_lv_centers,
    normalize_six_segment_angles,
)
from ..territory.mapping import last_retained_slice_index
from ..territory.models import LandmarkPair, StudyData


@dataclass(frozen=True)
class LgeScarResult:
    """Per-slice reference statistics and the resulting myocardial scar mask."""

    scar_mask: NDArray[np.bool_]
    reference_means: NDArray[np.float64]
    reference_stds: NDArray[np.float64]
    thresholds: NDArray[np.float64]
    reference_voxel_counts: NDArray[np.int64]


@dataclass(frozen=True)
class Aha16ScarSummary:
    """Native-voxel scar burden for AHA segments 1 through 16."""

    scar_percentages: NDArray[np.float64]
    scar_voxel_counts: NDArray[np.int64]
    myocardial_voxel_counts: NDArray[np.int64]


def _interpolate_missing_landmarks(
    landmarks: list[LandmarkPair | None],
    retained_slice_count: int,
) -> list[LandmarkPair]:
    valid_indices = np.asarray(
        [index for index in range(retained_slice_count) if landmarks[index] is not None],
        dtype=float,
    )
    if valid_indices.size == 0:
        raise ValueError("At least one retained slice must have AIP/IIP landmarks")
    coordinates = np.asarray(
        [landmarks[int(index)].as_array() for index in valid_indices],
        dtype=float,
    )
    targets = np.arange(retained_slice_count, dtype=float)
    interpolated = np.empty((retained_slice_count, 2, 2), dtype=float)
    for point_index in range(2):
        for coordinate_index in range(2):
            interpolated[:, point_index, coordinate_index] = np.interp(
                targets,
                valid_indices,
                coordinates[:, point_index, coordinate_index],
            )
    return [
        LandmarkPair(aip=tuple(points[0]), iip=tuple(points[1]))
        for points in interpolated
    ]


def compute_reference_statistics(
    image_slice: ArrayLike,
    myocardium_mask: ArrayLike,
    reference_roi_mask: ArrayLike,
    *,
    sd_multiplier: float = 5.0,
    min_roi_voxels: int = 20,
) -> tuple[int, float, float, float]:
    """Return count, mean, sample SD, and threshold for one manual ROI."""

    image = np.asarray(image_slice)
    myocardium = np.asarray(myocardium_mask, dtype=bool)
    roi = np.asarray(reference_roi_mask, dtype=bool)
    if image.shape != myocardium.shape or image.shape != roi.shape:
        raise ValueError("image, myocardium, and reference ROI must share a shape")
    if not np.isfinite(sd_multiplier) or sd_multiplier <= 0.0:
        raise ValueError("sd_multiplier must be positive and finite")
    if (
        isinstance(min_roi_voxels, bool)
        or not isinstance(min_roi_voxels, int)
        or min_roi_voxels < 2
    ):
        raise ValueError("min_roi_voxels must be an integer of at least 2")
    selected = roi & myocardium & np.isfinite(image)
    count = int(np.count_nonzero(selected))
    if count < min_roi_voxels:
        raise ValueError(
            f"Reference ROI contains {count} valid myocardial voxels; "
            f"at least {min_roi_voxels} are required"
        )
    values = np.asarray(image[selected], dtype=float)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    return count, mean, std, mean + float(sd_multiplier) * std


def compute_aha16_segment_ids(
    study: StudyData,
    landmarks: list[LandmarkPair | None],
    *,
    six_segment_warp: bool = True,
    deleted_slice_indices: Iterable[int] = (),
    center_overrides: Iterable[tuple[float, float] | None] | None = None,
) -> NDArray[np.uint8]:
    """Assign native myocardial voxels to AHA segments 1 through 16."""

    if len(landmarks) != study.slice_count:
        raise ValueError("There must be one landmark entry per slice")
    retained_basal = last_retained_slice_index(
        study.slice_count, deleted_slice_indices
    )
    mapped_landmarks = _interpolate_missing_landmarks(
        landmarks, retained_basal + 1
    )
    first_position = float(study.slice_positions[0])
    last_position = float(study.slice_positions[retained_basal])
    segment_ids = np.zeros(study.mr.shape, dtype=np.uint8)
    if center_overrides is None:
        retained_center_overrides = None
    else:
        all_center_overrides = list(center_overrides)
        if len(all_center_overrides) != study.slice_count:
            raise ValueError("center_overrides must have one entry per slice")
        retained_center_overrides = all_center_overrides[: retained_basal + 1]
    centers = estimate_lv_centers(
        study.territory_mask[: retained_basal + 1] > 0,
        center_overrides=retained_center_overrides,
    )

    for slice_index in range(retained_basal + 1):
        pair = mapped_landmarks[slice_index]
        myocardium = study.territory_mask[slice_index] > 0
        if not myocardium.any():
            continue
        if retained_basal == 0 or np.isclose(last_position, first_position):
            longitudinal_fraction = 0.0
        else:
            longitudinal_fraction = (
                float(study.slice_positions[slice_index]) - first_position
            ) / (last_position - first_position)
        if longitudinal_fraction < 1.0 / 3.0:
            representative_radius = 1.5
        elif longitudinal_fraction < 2.0 / 3.0:
            representative_radius = 2.5
        else:
            representative_radius = 3.5

        center = tuple(centers[slice_index])
        y, x = np.nonzero(myocardium)
        angles = aligned_angles(x, y, center, pair.aip, pair.iip)
        if six_segment_warp:
            angles = normalize_six_segment_angles(
                angles, center, pair.aip, pair.iip
            )
        voxel_segments = aha17_segment_id_map(
            [representative_radius], angles
        )[0]
        segment_ids[slice_index, y, x] = voxel_segments
    return segment_ids


def compute_aha16_scar_percentages(
    scar_mask: ArrayLike,
    segment_ids: ArrayLike,
) -> Aha16ScarSummary:
    """Count scar and myocardium voxels in each AHA segment 1 through 16."""

    scar = np.asarray(scar_mask, dtype=bool)
    segments = np.asarray(segment_ids)
    if scar.shape != segments.shape:
        raise ValueError("scar_mask and segment_ids must have the same shape")
    present = set(np.unique(segments).tolist())
    if not present.issubset(set(range(17))):
        raise ValueError("segment_ids may only contain AHA segment IDs 0 through 16")

    myocardium_counts = np.zeros(16, dtype=np.int64)
    scar_counts = np.zeros(16, dtype=np.int64)
    percentages = np.full(16, np.nan, dtype=float)
    for segment_id in range(1, 17):
        selected = segments == segment_id
        myocardium_count = int(np.count_nonzero(selected))
        scar_count = int(np.count_nonzero(scar & selected))
        myocardium_counts[segment_id - 1] = myocardium_count
        scar_counts[segment_id - 1] = scar_count
        if myocardium_count:
            percentages[segment_id - 1] = scar_count / myocardium_count * 100.0
    return Aha16ScarSummary(
        scar_percentages=percentages,
        scar_voxel_counts=scar_counts,
        myocardial_voxel_counts=myocardium_counts,
    )


def compute_lge_scar(
    study: StudyData,
    landmarks: list[LandmarkPair | None],
    reference_roi_masks: ArrayLike,
    *,
    sd_multiplier: float = 5.0,
    min_roi_voxels: int = 20,
    deleted_slice_indices: Iterable[int] = (),
) -> LgeScarResult:
    """Threshold every retained LGE slice from its manually selected ROI."""

    if len(landmarks) != study.slice_count:
        raise ValueError("There must be one landmark entry per slice")
    roi_masks = np.asarray(reference_roi_masks, dtype=bool)
    if roi_masks.shape != study.mr.shape:
        raise ValueError("reference_roi_masks must have the same shape as LGE")
    means = np.full(study.slice_count, np.nan, dtype=float)
    stds = np.full(study.slice_count, np.nan, dtype=float)
    thresholds = np.full(study.slice_count, np.nan, dtype=float)
    counts = np.zeros(study.slice_count, dtype=np.int64)
    scar = np.zeros(study.mr.shape, dtype=bool)

    retained_basal = last_retained_slice_index(
        study.slice_count, deleted_slice_indices
    )
    for slice_index in range(retained_basal + 1):
        myocardium = study.territory_mask[slice_index] > 0
        if not myocardium.any():
            continue
        finite = np.isfinite(study.mr[slice_index])
        try:
            count, mean, std, threshold = compute_reference_statistics(
                study.mr[slice_index],
                myocardium,
                roi_masks[slice_index],
                sd_multiplier=sd_multiplier,
                min_roi_voxels=min_roi_voxels,
            )
        except ValueError as error:
            raise ValueError(f"Slice {slice_index + 1}: {error}") from error
        means[slice_index] = mean
        stds[slice_index] = std
        thresholds[slice_index] = threshold
        counts[slice_index] = count
        scar[slice_index] = myocardium & finite & (
            study.mr[slice_index] > threshold
        )

    return LgeScarResult(
        scar_mask=scar,
        reference_means=means,
        reference_stds=stds,
        thresholds=thresholds,
        reference_voxel_counts=counts,
    )
