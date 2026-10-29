"""Patient-specific territory and coronary-distance bullseye workflow."""

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

from ..config import AppConfig, validate_config
from ..coronary import (
    load_cardiac_geometry,
    load_coronary_paths_from_nifti,
    project_coronary_paths,
    validate_geometry_against_short_axis_nifti,
)
from ..distance import (
    compute_distance_bullseye,
    compute_distance_volume_nifti,
    plot_distance_bullseye,
    resample_distance_to_short_axis,
    save_distance_nifti,
)
from ..territory.io import load_study_nifti
from ..territory.mapping import compute_bullseye
from ..territory.plotting import save_bullseye_figure
from ..territory.radial import projection_minimum_radius
from .bullseye_frame import prepare_bullseye_frame
from .short_axis import ShortAxisOutputs


@dataclass(frozen=True)
class BullseyeOutputs:
    territory_figure: Path
    distance_figure: Path
    world_distance: Path
    short_axis_distance: Path


def generate_bullseye(
    config: AppConfig,
    short_axis_outputs: ShortAxisOutputs | None = None,
    *,
    landmark_mode: str = "per-slice",
    projection: str = "1-4",
) -> BullseyeOutputs:
    """Generate territory and distance bullseyes for one configured case."""
    if landmark_mode not in {"per-slice", "single-reference"}:
        raise ValueError(
            "landmark_mode must be 'per-slice' or 'single-reference'"
        )
    minimum_radius = projection_minimum_radius(projection)
    if short_axis_outputs is None:
        validate_config(config, "bullseye")
        mr_path = config.bullseye.short_axis_mr
        mask_path = config.bullseye.short_axis_mask
        geometry_path = config.bullseye.geometry_json
    else:
        validate_config(config, "pipeline")
        mr_path = short_axis_outputs.mr
        mask_path = short_axis_outputs.mask
        geometry_path = short_axis_outputs.geometry
        missing = [
            path
            for path in (mr_path, mask_path, geometry_path)
            if path is None or not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Short-axis workflow did not produce all bullseye inputs: "
                + ", ".join(
                    "<missing mask>" if path is None else str(path) for path in missing
                )
            )

    if mask_path is None:
        raise ValueError("Bullseye generation requires a short-axis territory mask")
    if mr_path is None or geometry_path is None:
        raise ValueError(
            "Bullseye generation requires short-axis MR and geometry inputs"
        )

    coronary_mask = config.bullseye.coronary_mask
    world_myocardium_mask = config.bullseye.world_myocardium_mask
    if coronary_mask is None or world_myocardium_mask is None:
        raise ValueError("Bullseye generation requires coronary and myocardium masks")

    territory_figure = config.case.output_path("bullseye.png")
    distance_figure = config.case.output_path("distance_bullseye.png")
    world_distance = config.case.output_path("distance_world.nii.gz")
    short_axis_distance_output = config.case.output_path("distance_short_axis.nii.gz")

    print("1/8 Loading geometry and validating the short-axis NIfTI...")
    cardiac_geometry = load_cardiac_geometry(geometry_path)
    validate_geometry_against_short_axis_nifti(cardiac_geometry, mr_path)

    print("2/8 Loading MR and territory NIfTI data...")
    study = load_study_nifti(
        mr_path,
        mask_path,
        slice_axis=2,
        reverse_slice_order=False,
        territory_label_map=config.bullseye.territory_label_map,
    )
    print(f"    Loaded {study.slice_count} slices with shape {study.mr.shape[1:]}")
    if len(cardiac_geometry.slice_positions_mm) != study.slice_count:
        raise ValueError(
            "Geometry JSON and short-axis study have different slice counts"
        )

    if config.bullseye.reuse_saved_points:
        print("3/8 Load saved AIP/IIP points...")
    elif landmark_mode == "single-reference":
        print("3/8 Select AIP/IIP on one reference slice...")
    else:
        print("3/8 Select AIP followed by IIP on each slice...")
    landmark_path = (
        config.bullseye.landmarks_path or config.case.output_path("landmarks.json")
    )
    frame = prepare_bullseye_frame(
        study,
        landmark_path,
        reuse_saved_points=config.bullseye.reuse_saved_points,
        landmark_mode=landmark_mode,
        projection=projection,
        slice_positions=cardiac_geometry.slice_positions_mm,
        transpose_for_display=True,
    )
    landmarks = frame.landmarks
    deleted_slice_indices = frame.deleted_slice_indices
    center_overrides = frame.center_overrides
    completed = sum(pair is not None for pair in landmarks)
    print(f"    Valid landmarks: {completed}/{study.slice_count}")
    if deleted_slice_indices:
        deleted_numbers = ", ".join(
            str(index + 1) for index in sorted(deleted_slice_indices)
        )
        print(f"    Deleted slices: {deleted_numbers}")
    radial_frame = frame.radial_frame
    source_radii = radial_frame.source_radii
    center_blend_radius = None
    if minimum_radius == 0.0:
        positive_valid_radii = [
            float(source_radii[index])
            for index, pair in enumerate(landmarks)
            if pair is not None
            and index not in deleted_slice_indices
            and source_radii[index] > 0.0
            and np.any(study.territory_mask[index] != 0)
        ]
        center_blend_radius = (
            positive_valid_radii[0] if positive_valid_radii else 1.0
        )
    print(
        "    Radial anchors: "
        f"slice {radial_frame.apical_index + 1} "
        f"({radial_frame.apical_position:.2f} mm) -> "
        f"r={radial_frame.minimum_radius:g}; "
        f"slice {radial_frame.basal_index + 1} "
        f"({radial_frame.basal_position:.2f} mm) -> r=4"
    )

    print("4/8 Extracting and projecting the registered coronary centreline...")
    coronary_paths = load_coronary_paths_from_nifti(
        coronary_mask,
        mask_is_centerline=config.bullseye.coronary_mask_is_centerline,
        min_path_length_mm=config.bullseye.coronary_min_path_length_mm,
        resample_step_mm=config.bullseye.coronary_resample_step_mm,
        smoothing_sigma_voxels=config.bullseye.coronary_smoothing_sigma_voxels,
    )
    projected_coronaries = project_coronary_paths(
        coronary_paths,
        cardiac_geometry,
        study,
        landmarks,
        six_segment_warp=config.bullseye.six_segment_warp,
        deleted_slice_indices=deleted_slice_indices,
        radial_frame=radial_frame,
    )
    print(
        f"    Extracted {len(coronary_paths)} 3-D path(s); "
        f"projected {len(projected_coronaries)} visible path(s)"
    )

    print("5/8 Computing territory probabilities and AHA-17 mapping...")
    result = compute_bullseye(
        study,
        landmarks,
        angle_count=config.bullseye.angle_count,
        radial_count=config.bullseye.radial_count,
        angular_smoothing_sigma_deg=config.bullseye.angular_smoothing_sigma_deg,
        radial_smoothing_sigma=config.bullseye.radial_smoothing_sigma,
        center_blend_radius=center_blend_radius,
        six_segment_warp=config.bullseye.six_segment_warp,
        apex_territory=(
            config.bullseye.apex_territory if minimum_radius == 1.0 else None
        ),
        source_radii=source_radii,
        minimum_radius=minimum_radius,
        deleted_slice_indices=deleted_slice_indices,
        rescale_deleted_base=False,
        center_overrides=center_overrides,
    )

    print("6/8 Saving territory bullseye PNG with coronary overlay...")
    saved_territory_figure = save_bullseye_figure(
        result,
        territory_figure,
        title="Patient-specific territory, AHA-17 and coronary overlay",
        coronary_paths=projected_coronaries,
        coronary_label_colors=config.bullseye.coronary_label_colors,
        coronary_label_names=config.bullseye.coronary_label_names,
        coronary_default_color=config.bullseye.coronary_default_color,
        coronary_linewidth=config.bullseye.coronary_linewidth,
        show=config.bullseye.show_final_figure,
    )
    print(f"    Bullseye: {saved_territory_figure}")

    print("7/8 Computing world-space distance mask and resampling to SAX...")
    distance_volume = compute_distance_volume_nifti(
        coronary_mask,
        world_myocardium_mask,
        vessel_labels=config.bullseye.distance_vessel_labels,
        coronary_is_centerline=config.bullseye.coronary_mask_is_centerline,
    )
    saved_world_distance = save_distance_nifti(
        world_distance,
        distance_volume.distance_mm,
        distance_volume.affine,
        distance_volume.header,
    )

    short_axis_image = nib.load(str(mr_path))
    short_axis_territory_image = nib.load(str(mask_path))
    short_axis_myocardium = np.asarray(short_axis_territory_image.dataobj) > 0
    short_axis_distance = resample_distance_to_short_axis(
        distance_volume.distance_mm,
        distance_volume.affine,
        tuple(int(value) for value in short_axis_image.shape),
        np.asarray(cardiac_geometry.short_axis_sampling_affine, dtype=float),
        target_myocardium=short_axis_myocardium,
    )
    saved_short_axis_distance = save_distance_nifti(
        short_axis_distance_output,
        short_axis_distance,
        np.asarray(short_axis_image.affine, dtype=float),
        short_axis_image.header,
    )
    print(f"    World distance mask: {saved_world_distance}")
    print(f"    SAX distance mask: {saved_short_axis_distance}")

    print("8/8 Mapping the aligned distance heatmap to the bullseye...")
    internal_short_axis_distance = np.moveaxis(short_axis_distance, 2, 0)
    distance_bullseye = compute_distance_bullseye(
        study,
        internal_short_axis_distance,
        landmarks,
        result,
        source_radii,
        angular_smoothing_sigma_deg=config.bullseye.angular_smoothing_sigma_deg,
        radial_smoothing_sigma=config.bullseye.radial_smoothing_sigma,
        six_segment_warp=config.bullseye.six_segment_warp,
        deleted_slice_indices=deleted_slice_indices,
        rescale_deleted_base=False,
        center_overrides=center_overrides,
    )
    plot_distance_bullseye(
        distance_bullseye,
        distance_figure,
        title="Distance to nearest projected coronary",
        colormap=config.bullseye.distance_colormap,
        value_range_mm=config.bullseye.distance_value_range_mm,
        show=config.bullseye.show_final_figure,
    )
    print(f"    Distance bullseye: {distance_figure}")

    return BullseyeOutputs(
        territory_figure=territory_figure,
        distance_figure=distance_figure,
        world_distance=world_distance,
        short_axis_distance=short_axis_distance_output,
    )
