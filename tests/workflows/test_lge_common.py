from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from cardiac_bullseye.territory.models import LandmarkPair, StudyData
from cardiac_bullseye.workflows.lge_common import (
    compute_lge_bullseye_products,
    save_lge_bullseye_figures,
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


def _products():
    study, landmarks, scar = _three_slice_ring_study()
    return compute_lge_bullseye_products(
        study,
        landmarks,
        scar,
        angle_count=24,
        radial_count=9,
        angular_smoothing_sigma_deg=0.0,
        radial_smoothing_sigma=0.0,
        six_segment_warp=False,
    )


def test_compute_lge_bullseye_products_returns_intensity_and_scar_products():
    """Dropping either downstream product must fail this shared-core contract."""

    study, _, scar = _three_slice_ring_study()

    products = _products()

    assert products.intensity.intensity.shape == (9, 24)
    assert products.scar.scar_percentages.shape == (16,)
    assert products.scar.scar_voxel_counts.sum() == np.count_nonzero(scar)
    assert np.all(products.scar.scar_voxel_counts >= 0)
    assert study.mr.shape == scar.shape


def test_save_lge_bullseye_figures_writes_both_atomic_pngs(tmp_path):
    """Omitting either plot save must leave one requested image absent."""

    intensity_path = tmp_path / "intensity.png"
    scar_path = tmp_path / "scar.png"

    saved_intensity, saved_scar = save_lge_bullseye_figures(
        _products(), intensity_path, scar_path
    )

    assert saved_intensity == intensity_path
    assert saved_scar == scar_path
    assert intensity_path.stat().st_size > 0
    assert scar_path.stat().st_size > 0


def test_compute_lge_bullseye_products_rejects_scar_shape_mismatch():
    """Accepting a differently shaped scar array would misalign native voxels."""

    study, landmarks, _ = _three_slice_ring_study()

    with pytest.raises(ValueError, match="scar_mask must have the same shape as LGE"):
        compute_lge_bullseye_products(
            study,
            landmarks,
            np.zeros((2, 25, 25), dtype=bool),
            angle_count=24,
            radial_count=9,
            angular_smoothing_sigma_deg=0.0,
            radial_smoothing_sigma=0.0,
            six_segment_warp=False,
        )


def test_compute_lge_bullseye_products_rejects_scar_outside_myocardium():
    """Allowing background scar voxels would inflate AHA scar burden."""

    study, landmarks, scar = _three_slice_ring_study()
    scar[0, 0, 0] = True

    with pytest.raises(ValueError, match="scar_mask must be contained within myocardium"):
        compute_lge_bullseye_products(
            study,
            landmarks,
            scar,
            angle_count=24,
            radial_count=9,
            angular_smoothing_sigma_deg=0.0,
            radial_smoothing_sigma=0.0,
            six_segment_warp=False,
        )


def test_compute_lge_bullseye_products_excludes_deleted_basal_data():
    """Ignoring deleted indices would retain basal scar and intensity data."""

    study, landmarks, scar = _three_slice_ring_study()

    products = compute_lge_bullseye_products(
        study,
        landmarks,
        scar,
        angle_count=24,
        radial_count=9,
        angular_smoothing_sigma_deg=0.0,
        radial_smoothing_sigma=0.0,
        six_segment_warp=False,
        deleted_slice_indices=(2,),
    )

    assert products.scar.scar_voxel_counts.sum() == 2
    assert np.allclose(products.intensity.intensity[-1], 20.0)


def test_compute_lge_products_use_manual_center_overrides_for_both_outputs():
    study, landmarks, scar = _three_slice_ring_study()
    x_gradient = np.arange(study.mr.shape[2], dtype=float)[None, None, :]
    study = StudyData(
        mr=study.mr + x_gradient,
        territory_mask=study.territory_mask,
        slice_positions=study.slice_positions,
    )
    options = dict(
        angle_count=24,
        radial_count=9,
        angular_smoothing_sigma_deg=0.0,
        radial_smoothing_sigma=0.0,
        six_segment_warp=False,
    )

    automatic = compute_lge_bullseye_products(
        study,
        landmarks,
        scar,
        **options,
    )
    adjusted = compute_lge_bullseye_products(
        study,
        landmarks,
        scar,
        center_overrides=[(17.0, 12.0)] * 3,
        **options,
    )

    assert not np.allclose(
        automatic.intensity.intensity,
        adjusted.intensity.intensity,
        equal_nan=True,
    )
    assert not np.array_equal(
        automatic.scar.scar_voxel_counts,
        adjusted.scar.scar_voxel_counts,
    )


def test_compute_lge_products_accepts_a_single_pass_center_iterable():
    study, landmarks, scar = _three_slice_ring_study()

    products = compute_lge_bullseye_products(
        study,
        landmarks,
        scar,
        angle_count=24,
        radial_count=9,
        angular_smoothing_sigma_deg=0.0,
        radial_smoothing_sigma=0.0,
        six_segment_warp=False,
        center_overrides=(center for center in [(14.0, 12.0), None, None]),
    )

    assert products.intensity.intensity.shape == (9, 24)
