"""Shared, non-interactive LGE intensity and scar bullseye products."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from ..lge import (
    Aha16ScarSummary,
    LgeBullseyeResult,
    compute_aha16_scar_percentages,
    compute_aha16_segment_ids,
    compute_aha_scar_percentages,
    compute_aha_segment_ids,
    compute_lge_bullseye,
    plot_lge_bullseye,
    plot_lge_scar_bullseye,
)
from ..territory.models import LandmarkPair, StudyData


@dataclass(frozen=True)
class LgeBullseyeProducts:
    """Continuous intensity and native AHA scar products from one LGE study."""

    intensity: LgeBullseyeResult
    scar: Aha16ScarSummary


def compute_lge_bullseye_products(
    study: StudyData,
    landmarks: list[LandmarkPair | None],
    scar_mask: ArrayLike,
    *,
    angle_count: int,
    radial_count: int,
    angular_smoothing_sigma_deg: float,
    radial_smoothing_sigma: float,
    six_segment_warp: bool,
    deleted_slice_indices: Iterable[int] = (),
    include_apex: bool = True,
    center_overrides: Iterable[tuple[float, float] | None] | None = None,
    source_radii: Iterable[float] | None = None,
    minimum_radius: float = 1.0,
) -> LgeBullseyeProducts:
    """Compute LGE intensity and scar burden without workflow interaction."""

    scar = np.asarray(scar_mask, dtype=bool)
    if scar.shape != study.mr.shape:
        raise ValueError("scar_mask must have the same shape as LGE")
    if np.any(scar & ~(study.territory_mask > 0)):
        raise ValueError("scar_mask must be contained within myocardium")

    deleted = tuple(deleted_slice_indices)
    centers = None if center_overrides is None else tuple(center_overrides)
    radii = None if source_radii is None else tuple(source_radii)
    if radii is None:
        # Preserve the external-scar workflow's original frame until it opts in.
        segment_ids = compute_aha16_segment_ids(
            study,
            landmarks,
            six_segment_warp=six_segment_warp,
            deleted_slice_indices=deleted,
            center_overrides=centers,
        )
        scar_summary = compute_aha16_scar_percentages(scar, segment_ids)
    else:
        include_segment_17 = minimum_radius < 1.0
        segment_ids = compute_aha_segment_ids(
            study,
            landmarks,
            radii,
            include_segment_17=include_segment_17,
            six_segment_warp=six_segment_warp,
            deleted_slice_indices=deleted,
            center_overrides=centers,
        )
        scar_summary = compute_aha_scar_percentages(
            scar, segment_ids, segment_count=17 if include_segment_17 else 16
        )
    intensity = compute_lge_bullseye(
        study,
        landmarks,
        angle_count=angle_count,
        radial_count=radial_count,
        angular_smoothing_sigma_deg=angular_smoothing_sigma_deg,
        radial_smoothing_sigma=radial_smoothing_sigma,
        six_segment_warp=six_segment_warp,
        deleted_slice_indices=deleted,
        include_apex=include_apex,
        center_overrides=centers,
        source_radii=radii,
        minimum_radius=minimum_radius,
        center_blend_radius=1.0 if minimum_radius == 0.0 else None,
    )
    return LgeBullseyeProducts(intensity=intensity, scar=scar_summary)


def save_lge_bullseye_figures(
    products: LgeBullseyeProducts,
    intensity_path: str | Path,
    scar_path: str | Path,
) -> tuple[Path, Path]:
    """Save intensity and scar products through the atomic plotting APIs."""

    saved_intensity = Path(intensity_path)
    saved_scar = Path(scar_path)
    plot_lge_bullseye(products.intensity, saved_intensity, show=False)
    plot_lge_scar_bullseye(
        products.scar,
        saved_scar,
        include_apex=products.intensity.include_apex,
        show=False,
    )
    return saved_intensity, saved_scar
