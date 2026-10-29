"""Shared annotation and mask-anchored radial inputs for bullseye workflows."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from ..territory.io import load_center_overrides_json, load_deleted_slice_indices
from ..territory.landmarks import (
    load_saved_landmarks,
    select_landmarks_interactively,
    select_single_reference_interactively,
)
from ..territory.models import LandmarkPair, StudyData
from ..territory.radial import (
    MaskAnchoredRadialFrame,
    build_mask_anchored_radial_frame,
    projection_minimum_radius,
)


@dataclass(frozen=True)
class BullseyeFrameInputs:
    """Validated annotation data and the radial frame shared by all mappings."""

    landmarks: list[LandmarkPair | None]
    deleted_slice_indices: set[int]
    center_overrides: list[tuple[float, float] | None]
    radial_frame: MaskAnchoredRadialFrame


def prepare_bullseye_frame(
    study: StudyData,
    landmark_path: str | Path,
    *,
    reuse_saved_points: bool,
    landmark_mode: str,
    projection: str,
    slice_positions: ArrayLike,
    transpose_for_display: bool,
    legacy_lge_default: bool = False,
    expected_common_basename: str | None = None,
    cancel_is_error: bool = False,
) -> BullseyeFrameInputs:
    """Load or freshly annotate a study, then validate its complete plot frame."""

    if landmark_mode not in {"per-slice", "single-reference"}:
        raise ValueError("landmark_mode must be 'per-slice' or 'single-reference'")
    minimum_radius = projection_minimum_radius(projection)
    if np.count_nonzero(np.any(study.territory_mask != 0, axis=(1, 2))) < 2:
        raise ValueError("mask-anchored mapping requires at least two retained slices containing myocardium")
    landmark_path = Path(landmark_path)
    if reuse_saved_points and landmark_path.is_file():
        landmarks = load_saved_landmarks(
            study, landmark_path, legacy_default_lge=legacy_lge_default,
            expected_common_basename=expected_common_basename,
        )
    else:
        select = (
            select_single_reference_interactively
            if landmark_mode == "single-reference"
            else select_landmarks_interactively
        )
        landmarks = select(
            study,
            landmark_path,
            resume_existing=False,
            transpose_for_display=transpose_for_display,
            cancel_is_error=cancel_is_error,
        )

    deleted = load_deleted_slice_indices(landmark_path, study.slice_count)
    centers = load_center_overrides_json(
        landmark_path,
        study.slice_count,
        expected_image_shape=study.mr.shape,
        legacy_default_lge=legacy_lge_default,
        expected_common_basename=expected_common_basename,
    )
    if not any(pair is not None for pair in landmarks):
        raise RuntimeError("No slice has a valid AIP/IIP pair")
    radial_frame = build_mask_anchored_radial_frame(
        study.territory_mask,
        slice_positions,
        deleted,
        minimum_radius=minimum_radius,
    )
    return BullseyeFrameInputs(landmarks, deleted, centers, radial_frame)
