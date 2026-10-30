from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from cardiac_bullseye.territory.models import LandmarkPair, StudyData
from cardiac_bullseye.workflows.lge_common import (
    compute_lge_scar_summary,
    save_lge_scar_figure,
)


def _three_slice_ring_study() -> tuple[StudyData, list[LandmarkPair], np.ndarray]:
    """Build an apex-to-base ring stack with one myocardial scar per level."""

    size = 25
    y, x = np.ogrid[:size, :size]
    ring = ((x - 12) ** 2 + (y - 12) ** 2 >= 25) & (
        (x - 12) ** 2 + (y - 12) ** 2 <= 81
    )
    mr = np.zeros((3, size, size), dtype=float)
    for slice_index, value in enumerate((10.0, 20.0, 30.0)):
        mr[slice_index, ring] = value
    territory = np.repeat(ring[None, :, :], 3, axis=0).astype(np.uint8)
    scar = np.zeros_like(territory, dtype=bool)
    scar[0, 12, 20] = True
    scar[1, 20, 12] = True
    scar[2, 12, 4] = True
    study = StudyData(
        mr=mr,
        territory_mask=territory,
        slice_positions=np.asarray((0.0, 6.0, 12.0)),
    )
    landmarks = [LandmarkPair(aip=(12.0, 5.0), iip=(12.0, 19.0))] * 3
    return study, landmarks, scar


def _summary():
    study, landmarks, scar = _three_slice_ring_study()
    return compute_lge_scar_summary(
        study,
        landmarks,
        scar,
        six_segment_warp=False,
    )


def test_compute_lge_scar_summary_returns_native_segment_counts():

    study, _, scar = _three_slice_ring_study()

    summary = _summary()

    assert summary.scar_percentages.shape == (16,)
    assert summary.scar_voxel_counts.sum() == np.count_nonzero(scar)
    assert np.all(summary.scar_voxel_counts >= 0)
    assert study.mr.shape == scar.shape


def test_save_lge_scar_figure_writes_atomic_png(tmp_path):
    scar_path = tmp_path / "scar.png"

    saved_scar = save_lge_scar_figure(
        _summary(), scar_path, include_apex=False
    )

    assert saved_scar == scar_path
    assert scar_path.stat().st_size > 0


def test_compute_lge_scar_summary_rejects_scar_shape_mismatch():
    """Accepting a differently shaped scar array would misalign native voxels."""

    study, landmarks, _ = _three_slice_ring_study()

    with pytest.raises(ValueError, match="scar_mask must have the same shape as LGE"):
        compute_lge_scar_summary(
            study,
            landmarks,
            np.zeros((2, 25, 25), dtype=bool),
            six_segment_warp=False,
        )


def test_compute_lge_scar_summary_rejects_scar_outside_myocardium():
    """Allowing background scar voxels would inflate AHA scar burden."""

    study, landmarks, scar = _three_slice_ring_study()
    scar[0, 0, 0] = True

    with pytest.raises(ValueError, match="scar_mask must be contained within myocardium"):
        compute_lge_scar_summary(
            study,
            landmarks,
            scar,
            six_segment_warp=False,
        )


def test_compute_lge_scar_summary_excludes_deleted_basal_data():

    study, landmarks, scar = _three_slice_ring_study()

    summary = compute_lge_scar_summary(
        study,
        landmarks,
        scar,
        six_segment_warp=False,
        deleted_slice_indices=(2,),
    )

    assert summary.scar_voxel_counts.sum() == 2


def test_compute_lge_scar_summary_uses_manual_center_overrides():
    study, landmarks, scar = _three_slice_ring_study()
    x_gradient = np.arange(study.mr.shape[2], dtype=float)[None, None, :]
    study = StudyData(
        mr=study.mr + x_gradient,
        territory_mask=study.territory_mask,
        slice_positions=study.slice_positions,
    )
    options = dict(six_segment_warp=False)

    automatic = compute_lge_scar_summary(
        study,
        landmarks,
        scar,
        **options,
    )
    adjusted = compute_lge_scar_summary(
        study,
        landmarks,
        scar,
        center_overrides=[(17.0, 12.0)] * 3,
        **options,
    )

    assert not np.array_equal(
        automatic.scar_voxel_counts,
        adjusted.scar_voxel_counts,
    )


def test_compute_lge_scar_summary_accepts_a_single_pass_center_iterable():
    study, landmarks, scar = _three_slice_ring_study()

    summary = compute_lge_scar_summary(
        study,
        landmarks,
        scar,
        six_segment_warp=False,
        center_overrides=(center for center in [(14.0, 12.0), None, None]),
    )

    assert summary.scar_percentages.shape == (16,)
