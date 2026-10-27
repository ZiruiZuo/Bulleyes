from __future__ import annotations

import matplotlib
import nibabel as nib
import numpy as np

matplotlib.use("Agg")

from cardiac_bullseye.distance import (
    DistanceBullseyeResult,
    compute_distance_bullseye,
    compute_distance_volume_nifti,
    plot_distance_bullseye,
    resample_distance_to_short_axis,
    save_distance_nifti,
)
from cardiac_bullseye.distance.mapping import _mean_apical_segment_distances
from cardiac_bullseye.territory.mapping import build_bullseye
from cardiac_bullseye.territory.models import LandmarkPair, StudyData


def test_apex_distance_is_equal_mean_of_segments_13_to_16():
    segment_ids = np.asarray([[13, 13, 14, 15, 16, 16]])
    distance = np.asarray([[1.0, 3.0, 10.0, 20.0, 30.0, 50.0]])

    result = _mean_apical_segment_distances(distance, segment_ids)

    assert result == 18.0


def test_zero_to_four_distance_plot_does_not_cover_real_center_data():
    result = DistanceBullseyeResult(
        distance_mm=np.tile(np.linspace(1.0, 5.0, 5)[:, None], (1, 8)),
        radial_positions=np.linspace(0.0, 4.0, 5),
        angles=(np.arange(8, dtype=float) + 0.5) * 2.0 * np.pi / 8,
        apex_distance_mm=99.0,
    )

    _figure, axis = plot_distance_bullseye(result, show=False)

    assert len(axis.patches) == 0


def test_world_distance_is_minimum_of_three_projected_vessels(tmp_path):
    shape = (7, 7, 3)
    myocardium = np.zeros(shape, dtype=np.uint8)
    myocardium[1:6, 3, 1] = 1
    coronary = np.zeros(shape, dtype=np.uint8)
    coronary[1, 2, 1] = 2
    coronary[5, 4, 1] = 3
    coronary[3, 3, 2] = 4
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    myocardium_path = tmp_path / "myocardium.nii.gz"
    coronary_path = tmp_path / "coronary.nii.gz"
    nib.save(nib.Nifti1Image(myocardium, affine), myocardium_path)
    nib.save(nib.Nifti1Image(coronary, affine), coronary_path)

    result = compute_distance_volume_nifti(
        coronary_path,
        myocardium_path,
        vessel_labels=(2, 3, 4),
        coronary_is_centerline=True,
    )
    np.testing.assert_allclose(
        result.distance_mm[1:6, 3, 1], [0.0, 2.0, 0.0, 2.0, 0.0]
    )
    assert result.vessel_labels == (2, 3, 4)
    assert set(result.nearest_vessel_label[1:6, 3, 1]) <= {2, 3, 4}
    assert np.isnan(result.distance_mm[0, 0, 0])


def test_resampling_preserves_mm_and_masks_background():
    source = np.full((3, 3, 3), np.nan, dtype=float)
    source[1, 1, 1] = 7.5
    target_mask = np.zeros_like(source, dtype=bool)
    target_mask[1, 1, 1] = True
    sampled = resample_distance_to_short_axis(
        source,
        np.eye(4),
        source.shape,
        np.eye(4),
        target_myocardium=target_mask,
    )
    assert sampled[1, 1, 1] == 7.5
    assert np.isnan(sampled[0, 0, 0])


def test_saved_distance_nifti_remains_float_with_nan(tmp_path):
    data = np.full((2, 2, 2), np.nan, dtype=np.float32)
    data[1, 1, 1] = 3.25
    integer_header = nib.Nifti1Header()
    integer_header.set_data_dtype(np.uint8)
    path = save_distance_nifti(
        tmp_path / "distance.nii.gz", data, np.eye(4), integer_header
    )
    loaded = nib.load(path)
    assert loaded.get_data_dtype() == np.dtype(np.float32)
    loaded_data = loaded.get_fdata()
    assert loaded_data[1, 1, 1] == 3.25
    assert np.isnan(loaded_data[0, 0, 0])


def test_distance_bullseye_uses_exact_reference_grid(tmp_path):
    slices, size = 4, 25
    y, x = np.ogrid[:size, :size]
    ring = ((x - 12) ** 2 + (y - 12) ** 2 >= 25) & (
        (x - 12) ** 2 + (y - 12) ** 2 <= 81
    )
    mask = np.repeat(ring[None, :, :], slices, axis=0).astype(np.uint8)
    study = StudyData(
        mr=np.zeros_like(mask, dtype=float),
        territory_mask=mask,
        slice_positions=np.arange(slices, dtype=float),
    )
    landmarks = [LandmarkPair(aip=(12.0, 5.0), iip=(12.0, 19.0))] * slices
    probabilities = np.zeros((slices, 48, 3), dtype=float)
    probabilities[:, :, 0] = 1.0
    reference = build_bullseye(
        probabilities,
        np.arange(slices),
        source_radii=np.linspace(0.0, 4.0, slices),
        radial_count=31,
        radial_smoothing_sigma=0.0,
    )
    sax_distance = np.where(mask > 0, 12.5, np.nan)
    distance_result = compute_distance_bullseye(
        study,
        sax_distance,
        landmarks,
        reference,
        np.linspace(0.0, 4.0, slices),
        angular_smoothing_sigma_deg=0.0,
        radial_smoothing_sigma=0.0,
    )
    assert distance_result.distance_mm.shape == reference.territory_labels.shape
    np.testing.assert_array_equal(
        distance_result.radial_positions, reference.radial_positions
    )
    np.testing.assert_array_equal(distance_result.angles, reference.angles)
    np.testing.assert_allclose(distance_result.distance_mm, 12.5)
    output = tmp_path / "distance.png"
    plot_distance_bullseye(distance_result, output, show=False)
    assert output.is_file() and output.stat().st_size > 1000

    deleted_reference = build_bullseye(
        probabilities,
        np.arange(slices),
        source_radii=np.linspace(0.0, 4.0, slices),
        radial_count=31,
        radial_smoothing_sigma=0.0,
        deleted_slice_indices={3},
    )
    slice_distances = np.arange(1.0, slices + 1.0)[:, None, None]
    varying_sax_distance = np.where(mask > 0, slice_distances, np.nan)
    deleted_distance = compute_distance_bullseye(
        study,
        varying_sax_distance,
        landmarks,
        deleted_reference,
        np.linspace(0.0, 4.0, slices),
        angular_smoothing_sigma_deg=0.0,
        radial_smoothing_sigma=0.0,
        deleted_slice_indices={3},
    )
    assert np.isfinite(deleted_distance.distance_mm).all()
    radius_two = np.flatnonzero(
        np.isclose(deleted_distance.radial_positions, 2.0)
    )[0]
    np.testing.assert_allclose(deleted_distance.distance_mm[radius_two], 2.0)
    np.testing.assert_allclose(deleted_distance.distance_mm[-1], 3.0)

    anchored_radii = np.asarray([1.0, 2.0, 4.0, 5.0])
    anchored_reference = build_bullseye(
        probabilities,
        np.arange(slices),
        source_radii=anchored_radii,
        radial_count=31,
        radial_smoothing_sigma=0.0,
        deleted_slice_indices={3},
        rescale_deleted_base=False,
    )
    anchored_distance = compute_distance_bullseye(
        study,
        varying_sax_distance,
        landmarks,
        anchored_reference,
        anchored_radii,
        angular_smoothing_sigma_deg=0.0,
        radial_smoothing_sigma=0.0,
        deleted_slice_indices={3},
        rescale_deleted_base=False,
    )
    np.testing.assert_allclose(anchored_distance.distance_mm[-1], 3.0)
