"""Continuous image-intensity bullseye workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import AppConfig, validate_config
from ..coronary import (
    load_cardiac_geometry,
    load_coronary_paths_from_nifti,
    project_coronary_paths,
    validate_geometry_against_short_axis_nifti,
)
from ..intensity import load_intensity_study_nifti
from ..lge import compute_lge_bullseye, plot_lge_bullseye
from ..territory.io import load_center_overrides_json, load_deleted_slice_indices
from ..territory.landmarks import (
    load_saved_landmarks,
    select_landmarks_interactively,
    select_single_reference_interactively,
)
from ..territory.radial import (
    build_mask_anchored_radial_frame,
    projection_minimum_radius,
)


@dataclass(frozen=True)
class IntensityBullseyeOutputs:
    landmarks: Path
    figure: Path


def generate_intensity_bullseye(
    config: AppConfig,
    *,
    landmark_mode: str = "per-slice",
    projection: str = "1-4",
) -> IntensityBullseyeOutputs:
    """Map myocardial image intensity onto the shared patient bullseye frame."""

    if landmark_mode not in {"per-slice", "single-reference"}:
        raise ValueError("landmark_mode must be 'per-slice' or 'single-reference'")
    minimum_radius = projection_minimum_radius(projection)
    validate_config(config, "intensity-bullseye")
    image_path = config.bullseye.short_axis_mr
    mask_path = config.bullseye.short_axis_mask
    if image_path is None or mask_path is None:
        raise ValueError(
            "Intensity bullseye generation requires short-axis image and mask inputs"
        )

    print("1/4 Loading the short-axis image and myocardium mask...")
    study = load_intensity_study_nifti(image_path, mask_path)
    print(f"    Loaded {study.slice_count} slices with shape {study.mr.shape[1:]}")

    cardiac_geometry = None
    radial_slice_positions = study.slice_positions
    if config.bullseye.intensity_coronary_overlay:
        geometry_path = config.bullseye.geometry_json
        coronary_mask = config.bullseye.coronary_mask
        if geometry_path is None or coronary_mask is None:
            raise ValueError(
                "Intensity coronary overlay requires geometry and coronary inputs"
            )
        print("    Loading geometry for the optional coronary overlay...")
        cardiac_geometry = load_cardiac_geometry(geometry_path)
        validate_geometry_against_short_axis_nifti(cardiac_geometry, image_path)
        if len(cardiac_geometry.slice_positions_mm) != study.slice_count:
            raise ValueError(
                "Geometry JSON and short-axis intensity study have different "
                "slice counts"
            )
        radial_slice_positions = cardiac_geometry.slice_positions_mm

    landmark_path = config.case.output_path("landmarks.json")
    can_reuse = config.bullseye.reuse_saved_points and landmark_path.is_file()
    if can_reuse:
        print("2/4 Loading shared AIP/IIP points...")
        landmarks = load_saved_landmarks(study, landmark_path)
    elif landmark_mode == "single-reference":
        if config.bullseye.reuse_saved_points:
            print("2/4 Shared landmark file is missing; select one reference slice...")
        else:
            print("2/4 Select AIP/IIP on one reference slice...")
        landmarks = select_single_reference_interactively(
            study,
            landmark_path,
            resume_existing=False,
            transpose_for_display=True,
        )
    else:
        if config.bullseye.reuse_saved_points:
            print("2/4 Shared landmark file is missing; annotate each retained slice...")
        else:
            print("2/4 Select AIP followed by IIP on each retained slice...")
        landmarks = select_landmarks_interactively(
            study,
            landmark_path,
            resume_existing=False,
            transpose_for_display=True,
        )

    deleted = load_deleted_slice_indices(landmark_path, study.slice_count)
    center_overrides = load_center_overrides_json(
        landmark_path,
        study.slice_count,
        expected_image_shape=study.mr.shape,
    )
    if not any(pair is not None for pair in landmarks):
        raise RuntimeError("No slice has a valid AIP/IIP pair")

    print("3/4 Computing angular means and radial interpolation...")
    radial_frame = build_mask_anchored_radial_frame(
        study.territory_mask,
        radial_slice_positions,
        deleted,
        minimum_radius=minimum_radius,
    )
    result = compute_lge_bullseye(
        study,
        landmarks,
        angle_count=config.bullseye.angle_count,
        radial_count=config.bullseye.radial_count,
        angular_smoothing_sigma_deg=(
            config.bullseye.angular_smoothing_sigma_deg
        ),
        radial_smoothing_sigma=config.bullseye.radial_smoothing_sigma,
        six_segment_warp=config.bullseye.six_segment_warp,
        deleted_slice_indices=deleted,
        include_apex=True,
        center_overrides=center_overrides,
        source_radii=radial_frame.source_radii,
        minimum_radius=minimum_radius,
        center_blend_radius=1.0 if minimum_radius == 0.0 else None,
    )

    projected_coronaries = []
    if config.bullseye.intensity_coronary_overlay:
        assert cardiac_geometry is not None
        assert config.bullseye.coronary_mask is not None
        print("    Extracting and projecting the registered coronary centreline...")
        coronary_paths = load_coronary_paths_from_nifti(
            config.bullseye.coronary_mask,
            mask_is_centerline=config.bullseye.coronary_mask_is_centerline,
            min_path_length_mm=config.bullseye.coronary_min_path_length_mm,
            resample_step_mm=config.bullseye.coronary_resample_step_mm,
            smoothing_sigma_voxels=(
                config.bullseye.coronary_smoothing_sigma_voxels
            ),
        )
        projected_coronaries = project_coronary_paths(
            coronary_paths,
            cardiac_geometry,
            study,
            landmarks,
            six_segment_warp=config.bullseye.six_segment_warp,
            deleted_slice_indices=deleted,
            radial_frame=radial_frame,
        )
        print(
            f"    Extracted {len(coronary_paths)} 3-D path(s); "
            f"projected {len(projected_coronaries)} visible path(s)"
        )

    print("4/4 Saving the continuous intensity bullseye...")
    figure_path = config.case.output_path("intensity_bullseye.png")
    plot_lge_bullseye(
        result,
        figure_path,
        title="Continuous myocardial intensity bullseye",
        colormap="CMRmap",
        colorbar_label="Image intensity (native units)",
        coronary_paths=projected_coronaries,
        coronary_label_colors=config.bullseye.coronary_label_colors,
        coronary_label_names=config.bullseye.coronary_label_names,
        coronary_default_color=config.bullseye.coronary_default_color,
        coronary_linewidth=config.bullseye.coronary_linewidth,
        show=config.bullseye.show_final_figure,
    )
    print(f"    Landmarks: {landmark_path}")
    print(f"    Figure: {figure_path}")
    return IntensityBullseyeOutputs(landmarks=landmark_path, figure=figure_path)
