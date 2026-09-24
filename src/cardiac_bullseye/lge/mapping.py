"""Map raw short-axis LGE intensity onto a continuous AHA bullseye."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from ..territory.geometry import (
    aligned_angles,
    estimate_lv_centers,
    normalize_six_segment_angles,
)
from ..territory.mapping import remove_deleted_slices_from_radial_axis
from ..territory.models import LandmarkPair, StudyData


@dataclass(frozen=True)
class LgeBullseyeResult:
    """Continuous, unnormalised LGE intensity on the AHA bullseye grid."""

    intensity: NDArray[np.float64]
    radial_positions: NDArray[np.float64]
    angles: NDArray[np.float64]
    apex_intensity: float
    include_apex: bool = True

    def __post_init__(self) -> None:
        expected = (len(self.radial_positions), len(self.angles))
        if self.intensity.shape != expected:
            raise ValueError("intensity does not match the bullseye coordinate grid")


def _periodic_fill(values: NDArray[np.float64]) -> NDArray[np.float64]:
    valid = np.isfinite(values)
    if not valid.any():
        return values.copy()
    if valid.sum() == 1:
        return np.full_like(values, values[valid][0])
    indices = np.arange(values.size)
    valid_indices = indices[valid]
    return np.interp(
        indices,
        np.concatenate(
            [valid_indices - values.size, valid_indices, valid_indices + values.size]
        ),
        np.tile(values[valid], 3),
    )


def compute_slice_lge_profiles(
    study: StudyData,
    landmarks: list[LandmarkPair | None],
    *,
    angle_count: int = 360,
    angular_smoothing_sigma_deg: float = 1.5,
    six_segment_warp: bool = True,
    center_overrides: Iterable[tuple[float, float] | None] | None = None,
) -> NDArray[np.float64]:
    """Average raw myocardial LGE intensity within each angular bin."""

    if len(landmarks) != study.slice_count:
        raise ValueError("There must be one landmark entry per slice")
    if angle_count < 4:
        raise ValueError("angle_count must be at least 4")
    if angular_smoothing_sigma_deg < 0.0:
        raise ValueError("angular_smoothing_sigma_deg cannot be negative")

    profiles = np.full((study.slice_count, angle_count), np.nan, dtype=float)
    sigma_bins = angular_smoothing_sigma_deg / 360.0 * angle_count
    centers = estimate_lv_centers(
        study.territory_mask > 0,
        center_overrides=center_overrides,
    )
    for slice_index, pair in enumerate(landmarks):
        if pair is None:
            continue
        mask = (study.territory_mask[slice_index] > 0) & np.isfinite(
            study.mr[slice_index]
        )
        if not mask.any():
            continue
        center = tuple(centers[slice_index])
        y, x = np.nonzero(mask)
        values = np.asarray(study.mr[slice_index, y, x], dtype=float)
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
        if sigma_bins > 0.0:
            profile = ndimage.gaussian_filter1d(
                profile, sigma=sigma_bins, mode="wrap"
            )
        profiles[slice_index] = profile

    if not np.isfinite(profiles).any():
        raise ValueError("No valid LGE intensities can be mapped to the bullseye")
    return profiles


def _radial_interpolate(
    profiles: NDArray[np.float64],
    source_radii: NDArray[np.float64],
    target_radii: NDArray[np.float64],
) -> NDArray[np.float64]:
    output = np.full((target_radii.size, profiles.shape[1]), np.nan, dtype=float)
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


def compute_lge_bullseye(
    study: StudyData,
    landmarks: list[LandmarkPair | None],
    *,
    angle_count: int = 360,
    radial_count: int = 300,
    angular_smoothing_sigma_deg: float = 1.5,
    radial_smoothing_sigma: float = 1.5,
    six_segment_warp: bool = True,
    deleted_slice_indices: Iterable[int] = (),
    include_apex: bool = True,
    center_overrides: Iterable[tuple[float, float] | None] | None = None,
) -> LgeBullseyeResult:
    """Create a continuous bullseye while retaining native LGE intensities."""

    if radial_count < 3:
        raise ValueError("radial_count must be at least 3")
    if radial_smoothing_sigma < 0.0:
        raise ValueError("radial_smoothing_sigma cannot be negative")

    profiles = compute_slice_lge_profiles(
        study,
        landmarks,
        angle_count=angle_count,
        angular_smoothing_sigma_deg=angular_smoothing_sigma_deg,
        six_segment_warp=six_segment_warp,
        center_overrides=center_overrides,
    )
    if study.slice_count == 1:
        source_radii = np.asarray([0.0])
    else:
        positions = np.asarray(study.slice_positions, dtype=float)
        source_radii = 4.0 * (positions - positions[0]) / (
            positions[-1] - positions[0]
        )
    retained_profiles, retained_radii = remove_deleted_slices_from_radial_axis(
        profiles, source_radii, deleted_slice_indices
    )
    if not np.isfinite(retained_profiles).any():
        raise ValueError("No valid LGE intensities remain after deletion")

    radial_positions = np.linspace(1.0, 4.0, radial_count)
    intensity = _radial_interpolate(
        retained_profiles, retained_radii, radial_positions
    )
    if radial_smoothing_sigma > 0.0:
        intensity = ndimage.gaussian_filter1d(
            intensity,
            sigma=radial_smoothing_sigma,
            axis=0,
            mode="nearest",
        )

    apex_intensity = np.nan
    if include_apex:
        for profile in retained_profiles:
            if np.isfinite(profile).any():
                apex_intensity = float(np.nanmean(profile))
                break
    angles = (
        np.arange(angle_count, dtype=float) + 0.5
    ) * 2.0 * np.pi / angle_count
    return LgeBullseyeResult(
        intensity=intensity,
        radial_positions=radial_positions,
        angles=angles,
        apex_intensity=apex_intensity,
        include_apex=include_apex,
    )
