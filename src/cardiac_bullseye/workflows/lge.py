"""Unified workflow for native LGE intensity and AHA scar burden."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import AppConfig, validate_config
from ..lge import (
    compute_lge_scar,
    load_lge_study,
    load_lge_study_with_scar,
    save_aha16_scar_csv,
    save_lge_bullseye_npz,
    save_lge_scar_mask,
    save_reference_roi_masks,
    select_reference_rois_interactively,
)
from ..territory.models import StudyData
from ..territory.radial import projection_minimum_radius
from .bullseye_frame import BullseyeFrameInputs, prepare_bullseye_frame
from .lge_common import compute_lge_bullseye_products, save_lge_bullseye_figures


@dataclass(frozen=True)
class LgeBullseyeOutputs:
    landmarks: Path
    numeric_data: Path | None
    figure: Path
    reference_rois: Path | None
    scar_mask: Path | None
    scar_segments: Path | None
    scar_figure: Path


def generate_lge_bullseye(
    config: AppConfig,
    *,
    landmark_mode: str = "per-slice",
    projection: str = "1-4",
) -> LgeBullseyeOutputs:
    """Generate LGE bullseyes using normal ROIs or a configured scar mask."""

    if landmark_mode not in {"per-slice", "single-reference"}:
        raise ValueError("landmark_mode must be 'per-slice' or 'single-reference'")
    projection_minimum_radius(projection)
    validate_config(config, "lge-bullseye")
    lge_path = config.lge_bullseye.lge
    mask_path = config.lge_bullseye.mask
    if lge_path is None or mask_path is None:
        raise ValueError("LGE bullseye generation requires LGE and mask inputs")
    output_directory = config.case.output_directory
    output_directory.mkdir(parents=True, exist_ok=True)
    landmarks_path = (
        config.lge_bullseye.landmarks_path
        or config.case.output_path("lge_landmarks.json")
    )
    numeric_path = config.case.output_path("lge_intensity_bullseye.npz")
    figure_path = config.case.output_path("lge_intensity_bullseye.png")
    reference_rois_path = config.case.output_path("lge_normal_rois.json")
    scar_mask_path = config.case.output_path("lge_scar_mask.nii.gz")
    scar_segments_path = config.case.output_path("lge_scar_segments.csv")
    scar_figure_path = config.case.output_path("lge_scar_bullseye.png")

    if config.lge_bullseye.lge_scar_mask is not None:
        return _generate_from_external_scar(
            config,
            landmark_mode=landmark_mode,
            projection=projection,
            landmarks_path=landmarks_path,
            figure_path=figure_path,
            scar_figure_path=scar_figure_path,
        )

    print("1/7 Loading LGE and checking the myocardium mask grid...")
    study = load_lge_study(lge_path, mask_path)
    print(
        f"    Using {study.slice_count} apex-to-base slices with in-plane "
        f"shape {study.mr.shape[1:]}"
    )

    print("2/7 Preparing shared AIP/IIP points and radial frame...")
    frame = prepare_bullseye_frame(
        study,
        landmarks_path,
        reuse_saved_points=config.lge_bullseye.reuse_saved_points,
        landmark_mode=landmark_mode,
        projection=projection,
        slice_positions=study.slice_positions,
        transpose_for_display=True,
        legacy_lge_default=config.lge_bullseye.landmarks_path is None,
        expected_common_basename=config.case.output_path("landmarks.json").name,
    )
    return _generate_from_reference_rois(
        config,
        study,
        frame,
        projection=projection,
        landmarks_path=landmarks_path,
        numeric_path=numeric_path,
        figure_path=figure_path,
        reference_rois_path=reference_rois_path,
        scar_mask_path=scar_mask_path,
        scar_segments_path=scar_segments_path,
        scar_figure_path=scar_figure_path,
    )


def _generate_from_reference_rois(
    config: AppConfig,
    study: StudyData,
    frame: BullseyeFrameInputs,
    *,
    projection: str,
    landmarks_path: Path,
    numeric_path: Path,
    figure_path: Path,
    reference_rois_path: Path,
    scar_mask_path: Path,
    scar_segments_path: Path,
    scar_figure_path: Path,
) -> LgeBullseyeOutputs:
    """Estimate scar from fresh normal ROIs and save all seven outputs."""

    lge_path = config.lge_bullseye.lge
    mask_path = config.lge_bullseye.mask
    minimum_radius = projection_minimum_radius(projection)
    landmarks = frame.landmarks
    deleted = frame.deleted_slice_indices
    center_overrides = frame.center_overrides

    print("3/7 Draw a normal-myocardium ROI on every retained slice...")
    reference_roi_slices = select_reference_rois_interactively(
        study,
        landmarks,
        reference_rois_path,
        config.lge_bullseye.normal_roi_min_voxels,
        sd_multiplier=config.lge_bullseye.scar_sd_multiplier,
        deleted_slice_indices=deleted,
        resume_existing=False,
    )
    reference_masks = np.zeros(study.mr.shape, dtype=bool)
    for slice_index, reference_mask in enumerate(reference_roi_slices):
        if reference_mask is not None:
            reference_masks[slice_index] = reference_mask

    print(
        "4/7 Applying the per-slice "
        f"{config.lge_bullseye.scar_sd_multiplier:g}-SD scar threshold..."
    )
    scar_result = compute_lge_scar(
        study,
        landmarks,
        reference_masks,
        sd_multiplier=config.lge_bullseye.scar_sd_multiplier,
        min_roi_voxels=config.lge_bullseye.normal_roi_min_voxels,
        deleted_slice_indices=deleted,
    )
    save_reference_roi_masks(
        reference_rois_path,
        reference_roi_slices,
        scar_result=scar_result,
        sd_multiplier=config.lge_bullseye.scar_sd_multiplier,
        image_shape=study.mr.shape[1:],
    )
    saved_scar_mask = save_lge_scar_mask(
        scar_mask_path, lge_path, mask_path, scar_result.scar_mask
    )

    segment_count = 17 if minimum_radius == 0.0 else 16
    print(f"5/7 Mapping native LGE intensity and AHA-{segment_count} scar burden...")
    products = compute_lge_bullseye_products(
        study,
        landmarks,
        scar_result.scar_mask,
        angle_count=config.lge_bullseye.angle_count,
        radial_count=config.lge_bullseye.radial_count,
        angular_smoothing_sigma_deg=(config.lge_bullseye.angular_smoothing_sigma_deg),
        radial_smoothing_sigma=config.lge_bullseye.radial_smoothing_sigma,
        six_segment_warp=config.lge_bullseye.six_segment_warp,
        deleted_slice_indices=deleted,
        center_overrides=center_overrides,
        source_radii=frame.radial_frame.source_radii,
        minimum_radius=minimum_radius,
    )
    saved_numeric = save_lge_bullseye_npz(numeric_path, products.intensity)
    saved_scar_segments = save_aha16_scar_csv(scar_segments_path, products.scar)

    print(f"6/7 Saving the native-intensity and AHA-{segment_count} scar-burden figures...")
    save_lge_bullseye_figures(products, figure_path, scar_figure_path)
    print("7/7 Finalizing LGE bullseye outputs...")
    print(f"    Landmarks: {landmarks_path}")
    print(f"    Numeric data: {saved_numeric}")
    print(f"    Figure: {figure_path}")
    print(f"    Reference ROIs: {reference_rois_path}")
    print(f"    Scar mask: {saved_scar_mask}")
    print(f"    Scar segment table: {saved_scar_segments}")
    print(f"    Scar figure: {scar_figure_path}")
    return LgeBullseyeOutputs(
        landmarks_path,
        saved_numeric,
        figure_path,
        reference_rois_path,
        saved_scar_mask,
        saved_scar_segments,
        scar_figure_path,
    )


def _generate_from_external_scar(
    config: AppConfig,
    *,
    landmark_mode: str,
    projection: str,
    landmarks_path: Path,
    figure_path: Path,
    scar_figure_path: Path,
) -> LgeBullseyeOutputs:
    """Validate an external scar before annotation and save only two figures."""

    options = config.lge_bullseye
    minimum_radius = projection_minimum_radius(projection)
    print("1/3 Loading LGE, myocardium, and external scar mask...")
    loaded = load_lge_study_with_scar(
        options.lge,
        options.mask,
        options.lge_scar_mask,
    )
    print(
        f"    Using {loaded.study.slice_count} apex-to-base slices with in-plane "
        f"shape {loaded.study.mr.shape[1:]}"
    )

    print("2/3 Preparing shared AIP/IIP points and radial frame...")
    frame = prepare_bullseye_frame(
        loaded.study,
        landmarks_path,
        reuse_saved_points=options.reuse_saved_points,
        landmark_mode=landmark_mode,
        projection=projection,
        slice_positions=loaded.study.slice_positions,
        transpose_for_display=True,
        legacy_lge_default=options.landmarks_path is None,
        expected_common_basename=config.case.output_path("landmarks.json").name,
        cancel_is_error=True,
    )

    segment_count = 17 if minimum_radius == 0.0 else 16
    print(f"3/3 Mapping native LGE intensity and AHA-{segment_count} scar burden...")
    products = compute_lge_bullseye_products(
        loaded.study,
        frame.landmarks,
        loaded.scar_mask,
        angle_count=options.angle_count,
        radial_count=options.radial_count,
        angular_smoothing_sigma_deg=options.angular_smoothing_sigma_deg,
        radial_smoothing_sigma=options.radial_smoothing_sigma,
        six_segment_warp=options.six_segment_warp,
        deleted_slice_indices=frame.deleted_slice_indices,
        include_apex=minimum_radius == 0.0,
        center_overrides=frame.center_overrides,
        source_radii=frame.radial_frame.source_radii,
        minimum_radius=minimum_radius,
    )
    save_lge_bullseye_figures(products, figure_path, scar_figure_path)
    print(f"    Landmarks: {landmarks_path}")
    print(f"    Intensity figure: {figure_path}")
    print(f"    Scar figure: {scar_figure_path}")
    return LgeBullseyeOutputs(
        landmarks=landmarks_path,
        numeric_data=None,
        figure=figure_path,
        reference_rois=None,
        scar_mask=None,
        scar_segments=None,
        scar_figure=scar_figure_path,
    )
