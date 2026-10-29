"""External-scar LGE intensity and AHA scar bullseye workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import AppConfig, validate_config
from ..lge import load_lge_study_with_scar
from ..territory.radial import projection_minimum_radius
from .bullseye_frame import prepare_bullseye_frame
from .lge_common import compute_lge_bullseye_products, save_lge_bullseye_figures


@dataclass(frozen=True)
class LgeMaskBullseyeOutputs:
    """The three files written by the external-scar LGE workflow."""

    landmarks: Path
    intensity_figure: Path
    scar_figure: Path


def generate_lge_mask_bullseye(
    config: AppConfig,
    *,
    landmark_mode: str = "per-slice",
    projection: str = "1-4",
) -> LgeMaskBullseyeOutputs:
    """Render LGE intensity and scar figures from a validated external mask."""

    if landmark_mode not in {"per-slice", "single-reference"}:
        raise ValueError("landmark_mode must be 'per-slice' or 'single-reference'")
    minimum_radius = projection_minimum_radius(projection)
    validate_config(config, "lge-mask-bullseye")
    options = config.lge_mask_bullseye
    if options.lge is None or options.mask is None or options.lge_scar_mask is None:
        raise ValueError(
            "External-scar bullseye generation requires LGE, myocardium mask, "
            "and LGE scar mask inputs"
        )

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

    output_directory = config.case.output_directory
    output_directory.mkdir(parents=True, exist_ok=True)
    landmarks_path = (
        options.landmarks_path or config.case.output_path("lge_landmarks.json")
    )
    intensity_path = config.case.output_path("lge_intensity_bullseye.png")
    scar_path = config.case.output_path("lge_scar_bullseye.png")

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
    save_lge_bullseye_figures(products, intensity_path, scar_path)
    print(f"    Landmarks: {landmarks_path}")
    print(f"    Intensity figure: {intensity_path}")
    print(f"    Scar figure: {scar_path}")
    return LgeMaskBullseyeOutputs(
        landmarks=landmarks_path,
        intensity_figure=intensity_path,
        scar_figure=scar_path,
    )
