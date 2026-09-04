"""Map a short-axis myocardial distance volume onto the territory bullseye grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage

from ..territory.geometry import (
    aligned_angles,
    estimate_lv_center,
    normalize_six_segment_angles,
)
from ..territory.models import BullseyeResult, LandmarkPair, StudyData


@dataclass(frozen=True)
class DistanceBullseyeResult:
    distance_mm: NDArray[np.float64]
    radial_positions: NDArray[np.float64]
    angles: NDArray[np.float64]
    apex_distance_mm: float

    def __post_init__(self) -> None:
        expected = (len(self.radial_positions), len(self.angles))
        if self.distance_mm.shape != expected:
            raise ValueError("distance_mm does not match the bullseye coordinate grid")


def _periodic_fill(values: NDArray[np.float64]) -> NDArray[np.float64]:
    valid = np.isfinite(values)
    if not valid.any():
        return values.copy()
    if valid.sum() == 1:
        return np.full_like(values, values[valid][0])
    indices = np.arange(len(values))
    valid_indices = indices[valid]
    return np.interp(
        indices,
        np.concatenate([valid_indices - len(values), valid_indices, valid_indices + len(values)]),
        np.tile(values[valid], 3),
    )


def compute_slice_distance_profiles(
    study: StudyData,
    short_axis_distance_mm: ArrayLike,
    landmarks: list[LandmarkPair | None],
    *,
    angle_count: int = 360,
    angular_smoothing_sigma_deg: float = 1.5,
    six_segment_warp: bool = True,
) -> NDArray[np.float64]:
    """Average myocardial distance values in the same angular bins as territory."""

    distance = np.asarray(short_axis_distance_mm, dtype=float)
    if distance.shape != study.mr.shape:
        raise ValueError("short_axis_distance_mm must match the internal study shape")
    if len(landmarks) != study.slice_count:
        raise ValueError("There must be one landmark entry per slice")

    profiles = np.full((study.slice_count, angle_count), np.nan, dtype=float)
    sigma_bins = angular_smoothing_sigma_deg / 360.0 * angle_count
    for slice_index, pair in enumerate(landmarks):
        if pair is None:
            continue
        myocardial = (study.territory_mask[slice_index] > 0) & np.isfinite(
            distance[slice_index]
        )
        if not myocardial.any():
            continue
        center = estimate_lv_center(study.territory_mask[slice_index] > 0)
        y, x = np.nonzero(myocardial)
        values = distance[slice_index, y, x]
        angles = aligned_angles(x, y, center, pair.aip, pair.iip)
        if six_segment_warp:
            angles = normalize_six_segment_angles(
                angles, center, pair.aip, pair.iip
            )
        bins = np.floor(angles / (2.0 * np.pi) * angle_count).astype(int)
        bins = np.clip(bins, 0, angle_count - 1)
        sums = np.bincount(bins, weights=values, minlength=angle_count)
        counts = np.bincount(bins, minlength=angle_count)
        profile = np.divide(
            sums,
            counts,
            out=np.full(angle_count, np.nan, dtype=float),
            where=counts > 0,
        )
        profile = _periodic_fill(profile)
        if sigma_bins > 0:
            profile = ndimage.gaussian_filter1d(
                profile, sigma=sigma_bins, mode="wrap"
            )
        profiles[slice_index] = profile
    if not np.isfinite(profiles).any():
        raise ValueError("No valid distance values can be mapped to the bullseye")
    return profiles


def _radial_interpolate(
    profiles: NDArray[np.float64],
    source_radii: NDArray[np.float64],
    target_radii: NDArray[np.float64],
) -> NDArray[np.float64]:
    output = np.full((len(target_radii), profiles.shape[1]), np.nan, dtype=float)
    for angle_index in range(profiles.shape[1]):
        values = profiles[:, angle_index]
        valid = np.isfinite(values)
        if valid.sum() == 1:
            output[:, angle_index] = values[valid][0]
        elif valid.sum() >= 2:
            output[:, angle_index] = np.interp(
                target_radii, source_radii[valid], values[valid]
            )
    return output


def compute_distance_bullseye(
    study: StudyData,
    short_axis_distance_mm: ArrayLike,
    landmarks: list[LandmarkPair | None],
    reference_bullseye: BullseyeResult,
    source_radii: ArrayLike,
    *,
    angular_smoothing_sigma_deg: float = 1.5,
    radial_smoothing_sigma: float = 1.5,
    six_segment_warp: bool = True,
) -> DistanceBullseyeResult:
    """Create a distance heatmap exactly aligned with a territory bullseye."""

    source_radii_array = np.asarray(source_radii, dtype=float)
    if source_radii_array.shape != (study.slice_count,):
        raise ValueError("source_radii must have one value per slice")
    if not np.all(np.diff(source_radii_array) > 0):
        raise ValueError("source_radii must increase from apex to base")

    profiles = compute_slice_distance_profiles(
        study,
        short_axis_distance_mm,
        landmarks,
        angle_count=len(reference_bullseye.angles),
        angular_smoothing_sigma_deg=angular_smoothing_sigma_deg,
        six_segment_warp=six_segment_warp,
    )
    dense = _radial_interpolate(
        profiles,
        source_radii_array,
        np.asarray(reference_bullseye.radial_positions, dtype=float),
    )
    if radial_smoothing_sigma > 0:
        dense = ndimage.gaussian_filter1d(
            dense, sigma=radial_smoothing_sigma, axis=0, mode="nearest"
        )

    apex_distance = np.nan
    for profile in profiles:
        if np.isfinite(profile).any():
            apex_distance = float(np.nanmean(profile))
            break
    return DistanceBullseyeResult(
        distance_mm=dense,
        radial_positions=np.asarray(reference_bullseye.radial_positions, dtype=float),
        angles=np.asarray(reference_bullseye.angles, dtype=float),
        apex_distance_mm=apex_distance,
    )
