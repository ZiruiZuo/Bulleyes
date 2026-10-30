"""Unified workflow for native-voxel LGE scar burden."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from ..config import AppConfig, validate_config
from ..coronary import (
    CardiacGeometry,
    load_cardiac_geometry,
    load_coronary_paths_from_nifti,
    project_coronary_paths,
    validate_geometry_against_short_axis_nifti,
)
from ..lge import (
    compute_lge_scar,
    load_lge_study_with_extent,
    load_lge_study_with_scar,
    save_aha16_scar_csv,
    save_lge_scar_mask,
    save_reference_roi_masks,
    select_reference_rois_interactively,
)
from ..territory.models import StudyData
from ..territory.radial import projection_minimum_radius
from .bullseye_frame import BullseyeFrameInputs, prepare_bullseye_frame
from .lge_common import compute_lge_scar_summary, save_lge_scar_figure


@dataclass(frozen=True)
class LgeBullseyeOutputs:
    landmarks: Path
    reference_rois: Path | None
    scar_mask: Path | None
    scar_segments: Path | None
    scar_figure: Path


def _crop_geometry_to_lge_extent(
    geometry: CardiacGeometry,
    *,
    first_slice: int,
    last_slice: int,
    full_slice_count: int,
) -> CardiacGeometry:
    """Align full-stack geometry with the mask-cropped LGE study."""

    if geometry.short_axis_shape[2] != full_slice_count:
        raise ValueError(
            "Geometry JSON and the complete LGE stack have different slice counts"
        )
    if not (0 <= first_slice < last_slice <= full_slice_count):
        raise ValueError("Invalid cropped LGE slice extent")
    if first_slice == 0 and last_slice == full_slice_count:
        return geometry

    def shifted(affine: np.ndarray) -> np.ndarray:
        result = np.asarray(affine, dtype=float).copy()
        result[:3, 3] = (
            np.asarray(affine, dtype=float)
            @ np.asarray([0.0, 0.0, float(first_slice), 1.0])
        )[:3]
        return result

    return replace(
        geometry,
        short_axis_affine=shifted(geometry.short_axis_affine),
        short_axis_sampling_affine=shifted(geometry.short_axis_sampling_affine),
        short_axis_shape=(
            geometry.short_axis_shape[0],
            geometry.short_axis_shape[1],
            last_slice - first_slice,
        ),
        slice_positions_mm=geometry.slice_positions_mm[first_slice:last_slice],
    )


def _load_lge_coronary_geometry(
    config: AppConfig,
    lge_path: Path | None,
    *,
    first_slice: int,
    last_slice: int,
    full_slice_count: int,
) -> CardiacGeometry | None:
    if not config.bullseye.intensity_coronary_overlay:
        return None
    geometry_path = config.bullseye.geometry_json
    if lge_path is None or geometry_path is None:
        raise ValueError("LGE coronary overlay requires LGE and geometry inputs")
    print("    Loading geometry for the optional coronary overlay...")
    geometry = load_cardiac_geometry(geometry_path)
    validate_geometry_against_short_axis_nifti(geometry, lge_path)
    return _crop_geometry_to_lge_extent(
        geometry,
        first_slice=first_slice,
        last_slice=last_slice,
        full_slice_count=full_slice_count,
    )


def _project_lge_coronaries(
    config: AppConfig,
    study: StudyData,
    frame: BullseyeFrameInputs,
    geometry: CardiacGeometry | None,
):
    if geometry is None:
        return []
    coronary_mask = config.bullseye.coronary_mask
    if coronary_mask is None:
        raise ValueError("LGE coronary overlay requires a coronary mask")
    print("    Extracting and projecting the registered coronary centreline...")
    coronary_paths = load_coronary_paths_from_nifti(
        coronary_mask,
        mask_is_centerline=config.bullseye.coronary_mask_is_centerline,
        min_path_length_mm=config.bullseye.coronary_min_path_length_mm,
        resample_step_mm=config.bullseye.coronary_resample_step_mm,
        smoothing_sigma_voxels=config.bullseye.coronary_smoothing_sigma_voxels,
    )
    projected = project_coronary_paths(
        coronary_paths,
        geometry,
        study,
        frame.landmarks,
        six_segment_warp=config.lge_bullseye.six_segment_warp,
        deleted_slice_indices=frame.deleted_slice_indices,
        radial_frame=frame.radial_frame,
    )
    print(
        f"    Extracted {len(coronary_paths)} 3-D path(s); "
        f"projected {len(projected)} visible path(s)"
    )
    return projected


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
            scar_figure_path=scar_figure_path,
        )

    print("1/6 Loading LGE and checking the myocardium mask grid...")
    loaded = load_lge_study_with_extent(lge_path, mask_path)
    study = loaded.study
    print(
        f"    Using {study.slice_count} apex-to-base slices with in-plane "
        f"shape {study.mr.shape[1:]}"
    )

    cardiac_geometry = _load_lge_coronary_geometry(
        config,
        lge_path,
        first_slice=loaded.first_slice,
        last_slice=loaded.last_slice,
        full_slice_count=loaded.full_slice_count,
    )
    radial_positions = (
        cardiac_geometry.slice_positions_mm
        if cardiac_geometry is not None
        else study.slice_positions
    )
    print("2/6 Preparing shared AIP/IIP points and radial frame...")
    frame = prepare_bullseye_frame(
        study,
        landmarks_path,
        reuse_saved_points=config.lge_bullseye.reuse_saved_points,
        landmark_mode=landmark_mode,
        projection=projection,
        slice_positions=radial_positions,
        transpose_for_display=True,
        legacy_lge_default=config.lge_bullseye.landmarks_path is None,
        expected_common_basename=config.case.output_path("landmarks.json").name,
    )
    return _generate_from_reference_rois(
        config,
        study,
        frame,
        projection=projection,
        cardiac_geometry=cardiac_geometry,
        landmarks_path=landmarks_path,
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
    cardiac_geometry: CardiacGeometry | None,
    landmarks_path: Path,
    reference_rois_path: Path,
    scar_mask_path: Path,
    scar_segments_path: Path,
    scar_figure_path: Path,
) -> LgeBullseyeOutputs:
    """Estimate scar from fresh normal ROIs and save scar-only products."""

    lge_path = config.lge_bullseye.lge
    mask_path = config.lge_bullseye.mask
    minimum_radius = projection_minimum_radius(projection)
    landmarks = frame.landmarks
    deleted = frame.deleted_slice_indices
    center_overrides = frame.center_overrides

    print("3/6 Draw a normal-myocardium ROI on every retained slice...")
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
        "4/6 Applying the per-slice "
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
    print(f"5/6 Mapping native-voxel AHA-{segment_count} scar burden...")
    scar_summary = compute_lge_scar_summary(
        study,
        landmarks,
        scar_result.scar_mask,
        six_segment_warp=config.lge_bullseye.six_segment_warp,
        deleted_slice_indices=deleted,
        center_overrides=center_overrides,
        source_radii=frame.radial_frame.source_radii,
        minimum_radius=minimum_radius,
    )
    saved_scar_segments = save_aha16_scar_csv(scar_segments_path, scar_summary)
    projected_coronaries = _project_lge_coronaries(
        config, study, frame, cardiac_geometry
    )

    print(f"6/6 Saving the AHA-{segment_count} scar-burden figure...")
    save_lge_scar_figure(
        scar_summary,
        scar_figure_path,
        include_apex=minimum_radius == 0.0,
        coronary_paths=projected_coronaries,
        coronary_label_colors=config.bullseye.coronary_label_colors,
        coronary_label_names=config.bullseye.coronary_label_names,
        coronary_default_color=config.bullseye.coronary_default_color,
        coronary_linewidth=config.bullseye.coronary_linewidth,
        show=False,
    )
    print(f"    Landmarks: {landmarks_path}")
    print(f"    Reference ROIs: {reference_rois_path}")
    print(f"    Scar mask: {saved_scar_mask}")
    print(f"    Scar segment table: {saved_scar_segments}")
    print(f"    Scar figure: {scar_figure_path}")
    return LgeBullseyeOutputs(
        landmarks=landmarks_path,
        reference_rois=reference_rois_path,
        scar_mask=saved_scar_mask,
        scar_segments=saved_scar_segments,
        scar_figure=scar_figure_path,
    )


def _generate_from_external_scar(
    config: AppConfig,
    *,
    landmark_mode: str,
    projection: str,
    landmarks_path: Path,
    scar_figure_path: Path,
) -> LgeBullseyeOutputs:
    """Validate an external scar before annotation and save its scar figure."""

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

    cardiac_geometry = _load_lge_coronary_geometry(
        config,
        options.lge,
        first_slice=loaded.first_slice,
        last_slice=loaded.last_slice,
        full_slice_count=loaded.full_slice_count,
    )
    radial_positions = (
        cardiac_geometry.slice_positions_mm
        if cardiac_geometry is not None
        else loaded.study.slice_positions
    )

    print("2/3 Preparing shared AIP/IIP points and radial frame...")
    frame = prepare_bullseye_frame(
        loaded.study,
        landmarks_path,
        reuse_saved_points=options.reuse_saved_points,
        landmark_mode=landmark_mode,
        projection=projection,
        slice_positions=radial_positions,
        transpose_for_display=True,
        legacy_lge_default=options.landmarks_path is None,
        expected_common_basename=config.case.output_path("landmarks.json").name,
        cancel_is_error=True,
    )

    segment_count = 17 if minimum_radius == 0.0 else 16
    print(f"3/3 Mapping native-voxel AHA-{segment_count} scar burden...")
    scar_summary = compute_lge_scar_summary(
        loaded.study,
        frame.landmarks,
        loaded.scar_mask,
        six_segment_warp=options.six_segment_warp,
        deleted_slice_indices=frame.deleted_slice_indices,
        center_overrides=frame.center_overrides,
        source_radii=frame.radial_frame.source_radii,
        minimum_radius=minimum_radius,
    )
    projected_coronaries = _project_lge_coronaries(
        config, loaded.study, frame, cardiac_geometry
    )
    save_lge_scar_figure(
        scar_summary,
        scar_figure_path,
        include_apex=minimum_radius == 0.0,
        coronary_paths=projected_coronaries,
        coronary_label_colors=config.bullseye.coronary_label_colors,
        coronary_label_names=config.bullseye.coronary_label_names,
        coronary_default_color=config.bullseye.coronary_default_color,
        coronary_linewidth=config.bullseye.coronary_linewidth,
        show=False,
    )
    print(f"    Landmarks: {landmarks_path}")
    print(f"    Scar figure: {scar_figure_path}")
    return LgeBullseyeOutputs(
        landmarks=landmarks_path,
        reference_rois=None,
        scar_mask=None,
        scar_segments=None,
        scar_figure=scar_figure_path,
    )
