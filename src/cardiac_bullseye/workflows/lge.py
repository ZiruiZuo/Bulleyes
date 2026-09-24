"""Interactive workflow for native LGE intensity and AHA-16 scar burden."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import AppConfig, validate_config
from ..lge import (
    compute_lge_scar,
    load_lge_study,
    save_aha16_scar_csv,
    save_lge_bullseye_npz,
    save_lge_scar_mask,
    save_reference_roi_masks,
    select_reference_rois_interactively,
)
from ..territory.io import load_center_overrides_json, load_deleted_slice_indices
from ..territory.landmarks import load_saved_landmarks, select_landmarks_interactively
from .lge_common import compute_lge_bullseye_products, save_lge_bullseye_figures


@dataclass(frozen=True)
class LgeBullseyeOutputs:
    landmarks: Path
    numeric_data: Path
    figure: Path
    reference_rois: Path
    scar_mask: Path
    scar_segments: Path
    scar_figure: Path


def generate_lge_bullseye(
    config: AppConfig,
) -> LgeBullseyeOutputs:
    """Generate configured intensity and infarct-scar bullseyes from SAX LGE."""

    validate_config(config, "lge-bullseye")
    lge_path = config.lge_bullseye.lge
    mask_path = config.lge_bullseye.mask
    if lge_path is None or mask_path is None:
        raise ValueError("LGE bullseye generation requires LGE and mask inputs")
    output_directory = config.case.output_directory
    output_directory.mkdir(parents=True, exist_ok=True)
    landmarks_path = config.case.output_path("lge_landmarks.json")
    numeric_path = config.case.output_path("lge_intensity_bullseye.npz")
    figure_path = config.case.output_path("lge_intensity_bullseye.png")
    reference_rois_path = config.case.output_path("lge_normal_rois.json")
    scar_mask_path = config.case.output_path("lge_scar_mask.nii.gz")
    scar_segments_path = config.case.output_path("lge_scar_segments.csv")
    scar_figure_path = config.case.output_path("lge_scar_bullseye.png")

    print("1/7 Loading LGE and checking the myocardium mask grid...")
    study = load_lge_study(lge_path, mask_path)
    print(
        f"    Using {study.slice_count} apex-to-base slices with in-plane "
        f"shape {study.mr.shape[1:]}"
    )

    print(
        "2/7 Load saved AIP/IIP points..."
        if config.lge_bullseye.reuse_saved_points
        else "2/7 Select AIP followed by IIP on each retained slice..."
    )
    if config.lge_bullseye.reuse_saved_points:
        landmarks = load_saved_landmarks(study, landmarks_path)
    else:
        landmarks = select_landmarks_interactively(
            study, landmarks_path, resume_existing=False
        )
    deleted = load_deleted_slice_indices(landmarks_path, study.slice_count)
    center_overrides = load_center_overrides_json(
        landmarks_path,
        study.slice_count,
        expected_image_shape=study.mr.shape,
    )
    if not any(pair is not None for pair in landmarks):
        raise RuntimeError("No slice has a valid AIP/IIP pair")

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

    print("5/7 Mapping native LGE intensity and AHA-16 scar burden...")
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
    )
    saved_numeric = save_lge_bullseye_npz(numeric_path, products.intensity)
    saved_scar_segments = save_aha16_scar_csv(scar_segments_path, products.scar)

    print("6/7 Saving the native-intensity and AHA-16 scar-burden figures...")
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
