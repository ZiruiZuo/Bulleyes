"""External-scar LGE intensity and AHA-16 bullseye workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import AppConfig, validate_config
from ..lge import load_lge_study_with_scar
from ..territory.io import load_center_overrides_json, load_deleted_slice_indices
from ..territory.landmarks import load_saved_landmarks, select_landmarks_interactively
from .lge_common import compute_lge_bullseye_products, save_lge_bullseye_figures


@dataclass(frozen=True)
class LgeMaskBullseyeOutputs:
    """The three files written by the external-scar LGE workflow."""

    landmarks: Path
    intensity_figure: Path
    scar_figure: Path


def generate_lge_mask_bullseye(config: AppConfig) -> LgeMaskBullseyeOutputs:
    """Render LGE intensity and scar figures from a validated external mask."""

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
    landmarks_path = config.case.output_path("lge_landmarks.json")
    intensity_path = config.case.output_path("lge_intensity_bullseye.png")
    scar_path = config.case.output_path("lge_scar_bullseye.png")

    print(
        "2/3 Load saved AIP/IIP points..."
        if options.reuse_saved_points
        else "2/3 Select AIP followed by IIP on each retained slice..."
    )
    if options.reuse_saved_points:
        landmarks = load_saved_landmarks(loaded.study, landmarks_path)
    else:
        landmarks = select_landmarks_interactively(
            loaded.study,
            landmarks_path,
            resume_existing=False,
            cancel_is_error=True,
        )
    deleted = load_deleted_slice_indices(landmarks_path, loaded.study.slice_count)
    center_overrides = load_center_overrides_json(
        landmarks_path,
        loaded.study.slice_count,
        expected_image_shape=loaded.study.mr.shape,
    )
    if not any(pair is not None for pair in landmarks):
        raise RuntimeError("No slice has a valid AIP/IIP pair")

    print("3/3 Mapping native LGE intensity and AHA-16 scar burden...")
    products = compute_lge_bullseye_products(
        loaded.study,
        landmarks,
        loaded.scar_mask,
        angle_count=options.angle_count,
        radial_count=options.radial_count,
        angular_smoothing_sigma_deg=options.angular_smoothing_sigma_deg,
        radial_smoothing_sigma=options.radial_smoothing_sigma,
        six_segment_warp=options.six_segment_warp,
        deleted_slice_indices=deleted,
        include_apex=False,
        center_overrides=center_overrides,
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
