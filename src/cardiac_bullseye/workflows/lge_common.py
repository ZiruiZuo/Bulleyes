"""Shared, non-interactive LGE scar-burden computation and plotting."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from ..coronary import ProjectedCoronaryPath
from ..lge import (
    Aha16ScarSummary,
    compute_aha16_scar_percentages,
    compute_aha16_segment_ids,
    compute_aha_scar_percentages,
    compute_aha_segment_ids,
    plot_lge_scar_bullseye,
)
from ..territory.models import LandmarkPair, StudyData


def compute_lge_scar_summary(
    study: StudyData,
    landmarks: list[LandmarkPair | None],
    scar_mask: ArrayLike,
    *,
    six_segment_warp: bool,
    deleted_slice_indices: Iterable[int] = (),
    center_overrides: Iterable[tuple[float, float] | None] | None = None,
    source_radii: Iterable[float] | None = None,
    minimum_radius: float = 1.0,
) -> Aha16ScarSummary:
    """Compute native-voxel AHA scar burden without an intensity bullseye."""

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
        return compute_aha16_scar_percentages(scar, segment_ids)

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
    return compute_aha_scar_percentages(
        scar, segment_ids, segment_count=17 if include_segment_17 else 16
    )


def save_lge_scar_figure(
    summary: Aha16ScarSummary,
    scar_path: str | Path,
    *,
    include_apex: bool,
    coronary_paths: list[ProjectedCoronaryPath] | None = None,
    coronary_label_colors: dict[int, str] | None = None,
    coronary_label_names: dict[int, str] | None = None,
    coronary_default_color: str | None = None,
    coronary_linewidth: float = 1.8,
    show: bool = False,
) -> Path:
    """Save the scar-burden PNG, optionally with projected coronaries."""

    saved_scar = Path(scar_path)
    plot_lge_scar_bullseye(
        summary,
        saved_scar,
        include_apex=include_apex,
        coronary_paths=coronary_paths,
        coronary_label_colors=coronary_label_colors,
        coronary_label_names=coronary_label_names,
        coronary_default_color=coronary_default_color,
        coronary_linewidth=coronary_linewidth,
        show=show,
    )
    return saved_scar
