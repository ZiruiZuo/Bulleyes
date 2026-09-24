from __future__ import annotations

import matplotlib
import numpy as np
import pytest
import SimpleITK as sitk

matplotlib.use("Agg")

from cardiac_bullseye.lge import (
    compute_lge_bullseye,
    load_lge_study,
    load_lge_study_with_scar,
    plot_lge_bullseye,
    save_lge_bullseye_npz,
)
from cardiac_bullseye.territory.models import LandmarkPair, StudyData


def _write_image(
    path,
    array,
    *,
    origin=(1.0, 2.0, 3.0),
    spacing=(0.8, 0.9, 6.0),
    direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    is_vector=False,
):
    image = sitk.GetImageFromArray(np.asarray(array), isVector=is_vector)
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    image.SetDirection(direction)
    sitk.WriteImage(image, str(path))


def _constant_slice_study(values=(10.0, 20.0, 30.0)):
    size = 25
    y, x = np.ogrid[:size, :size]
    ring = ((x - 12) ** 2 + (y - 12) ** 2 >= 25) & (
        (x - 12) ** 2 + (y - 12) ** 2 <= 81
    )
    mask = np.repeat(ring[None, :, :], len(values), axis=0).astype(np.uint8)
    mr = np.zeros_like(mask, dtype=float)
    for index, value in enumerate(values):
        mr[index, ring] = value
    study = StudyData(mr, mask, np.arange(len(values), dtype=float) * 6.0)
    landmark = LandmarkPair(aip=(12.0, 5.0), iip=(12.0, 19.0))
    return study, [landmark] * len(values)


def test_loader_checks_same_grid_and_crops_to_myocardial_extent(tmp_path):
    lge = np.arange(5 * 9 * 9, dtype=np.float32).reshape(5, 9, 9)
    mask = np.zeros_like(lge, dtype=np.uint8)
    mask[1:4, 2:7, 2:7] = 1
    lge_path = tmp_path / "lge.nii.gz"
    mask_path = tmp_path / "myocardium.seg.nrrd"
    _write_image(lge_path, lge)
    _write_image(mask_path, mask)

    study = load_lge_study(lge_path, mask_path)

    assert study.mr.shape == (3, 9, 9)
    np.testing.assert_array_equal(study.mr, lge[1:4])
    np.testing.assert_array_equal(study.territory_mask, mask[1:4])
    np.testing.assert_allclose(study.slice_positions, [0.0, 6.0, 12.0])


def test_loader_rejects_geometry_mismatch_without_resampling(tmp_path):
    volume = np.zeros((3, 9, 9), dtype=np.float32)
    volume[:, 2:7, 2:7] = 1
    lge_path = tmp_path / "lge.nii.gz"
    mask_path = tmp_path / "myocardium.nrrd"
    _write_image(lge_path, volume)
    _write_image(mask_path, volume, origin=(2.0, 2.0, 3.0))

    with pytest.raises(ValueError, match="origin"):
        load_lge_study(lge_path, mask_path)


def test_scar_loader_crops_all_inputs_to_myocardial_extent(tmp_path):
    lge = np.arange(5 * 9 * 9, dtype=np.float32).reshape(5, 9, 9)
    myocardium = np.zeros_like(lge, dtype=np.uint8)
    myocardium[1:4, 2:7, 2:7] = 1
    scar = np.zeros_like(lge, dtype=np.uint8)
    scar[2, 4, 4] = 9
    lge_path = tmp_path / "lge.nii.gz"
    myocardium_path = tmp_path / "myocardium.seg.nrrd"
    scar_path = tmp_path / "scar.nrrd"
    _write_image(lge_path, lge)
    _write_image(myocardium_path, myocardium)
    _write_image(scar_path, scar)

    loaded = load_lge_study_with_scar(lge_path, myocardium_path, scar_path)

    assert loaded.study.mr.shape == (3, 9, 9)
    assert loaded.scar_mask.shape == (3, 9, 9)
    np.testing.assert_array_equal(loaded.study.mr, lge[1:4])
    np.testing.assert_array_equal(loaded.study.territory_mask, myocardium[1:4])
    np.testing.assert_array_equal(loaded.scar_mask, scar[1:4] != 0)


def test_scar_loader_accepts_empty_scar_mask(tmp_path):
    lge = np.ones((5, 9, 9), dtype=np.float32)
    myocardium = np.zeros_like(lge, dtype=np.uint8)
    myocardium[1:4, 2:7, 2:7] = 1
    scar = np.zeros_like(lge, dtype=np.uint8)
    lge_path = tmp_path / "lge.nii.gz"
    myocardium_path = tmp_path / "myocardium.nrrd"
    scar_path = tmp_path / "scar.nrrd"
    _write_image(lge_path, lge)
    _write_image(myocardium_path, myocardium)
    _write_image(scar_path, scar)

    loaded = load_lge_study_with_scar(lge_path, myocardium_path, scar_path)

    assert not np.any(loaded.scar_mask)


def test_scar_loader_rejects_scar_voxels_outside_myocardium(tmp_path):
    lge = np.ones((5, 9, 9), dtype=np.float32)
    myocardium = np.zeros_like(lge, dtype=np.uint8)
    myocardium[1:4, 2:7, 2:7] = 1
    scar = np.zeros_like(lge, dtype=np.uint8)
    scar[0, 0, 0] = 1
    scar[2, 1, 1] = 1
    lge_path = tmp_path / "lge.nii.gz"
    myocardium_path = tmp_path / "myocardium.nrrd"
    scar_path = tmp_path / "scar.nrrd"
    _write_image(lge_path, lge)
    _write_image(myocardium_path, myocardium)
    _write_image(scar_path, scar)

    with pytest.raises(ValueError, match=r"2.*outside.*myocardium"):
        load_lge_study_with_scar(lge_path, myocardium_path, scar_path)


@pytest.mark.parametrize(
    ("name", "array", "metadata"),
    [
        ("size", np.zeros((5, 9, 8), dtype=np.uint8), {}),
        ("spacing", np.zeros((5, 9, 9), dtype=np.uint8), {"spacing": (0.8, 0.9, 6.1)}),
        ("origin", np.zeros((5, 9, 9), dtype=np.uint8), {"origin": (1.1, 2.0, 3.0)}),
        (
            "direction",
            np.zeros((5, 9, 9), dtype=np.uint8),
            {"direction": (-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)},
        ),
    ],
)
def test_scar_loader_rejects_scar_physical_grid_mismatches(
    tmp_path, name, array, metadata
):
    lge = np.ones((5, 9, 9), dtype=np.float32)
    myocardium = np.zeros_like(lge, dtype=np.uint8)
    myocardium[1:4, 2:7, 2:7] = 1
    lge_path = tmp_path / "lge.nii.gz"
    myocardium_path = tmp_path / "myocardium.nrrd"
    scar_path = tmp_path / "scar.nrrd"
    _write_image(lge_path, lge)
    _write_image(myocardium_path, myocardium)
    _write_image(scar_path, array, **metadata)

    with pytest.raises(ValueError, match=rf"scar mask.*{name}"):
        load_lge_study_with_scar(lge_path, myocardium_path, scar_path)


def test_scar_loader_rejects_vector_scar_mask(tmp_path):
    lge = np.ones((5, 9, 9), dtype=np.float32)
    myocardium = np.zeros_like(lge, dtype=np.uint8)
    myocardium[1:4, 2:7, 2:7] = 1
    scar = np.zeros((5, 9, 9, 2), dtype=np.uint8)
    lge_path = tmp_path / "lge.nii.gz"
    myocardium_path = tmp_path / "myocardium.nrrd"
    scar_path = tmp_path / "scar.nrrd"
    _write_image(lge_path, lge)
    _write_image(myocardium_path, myocardium)
    _write_image(scar_path, scar, is_vector=True)

    with pytest.raises(ValueError, match="scar mask must be a scalar"):
        load_lge_study_with_scar(lge_path, myocardium_path, scar_path)


def test_scar_loader_uses_union_of_component_myocardium_segmentation(tmp_path):
    lge = np.ones((5, 9, 9), dtype=np.float32)
    myocardium = np.zeros((5, 9, 9, 2), dtype=np.uint8)
    myocardium[1:4, 2:7, 2:7, 0] = 1
    myocardium[2, 1, 1, 1] = 1
    scar = np.zeros_like(lge, dtype=np.uint8)
    scar[2, 1, 1] = 1
    lge_path = tmp_path / "lge.nii.gz"
    myocardium_path = tmp_path / "myocardium.seg.nrrd"
    scar_path = tmp_path / "scar.nrrd"
    _write_image(lge_path, lge)
    _write_image(myocardium_path, myocardium, is_vector=True)
    _write_image(scar_path, scar)

    loaded = load_lge_study_with_scar(lge_path, myocardium_path, scar_path)

    np.testing.assert_array_equal(
        loaded.study.territory_mask, np.any(myocardium != 0, axis=-1)[1:4]
    )
    np.testing.assert_array_equal(loaded.scar_mask, scar[1:4] != 0)
    assert loaded.study.territory_mask[1, 1, 1]
    assert loaded.scar_mask[1, 1, 1]


def test_bullseye_preserves_raw_lge_intensity_and_interpolates_skips():
    study, landmarks = _constant_slice_study()
    landmarks[1] = None

    result = compute_lge_bullseye(
        study,
        landmarks,
        angle_count=24,
        radial_count=4,
        angular_smoothing_sigma_deg=0.0,
        radial_smoothing_sigma=0.0,
        six_segment_warp=False,
    )

    np.testing.assert_allclose(result.radial_positions, [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(result.intensity[:, 0], [15.0, 20.0, 25.0, 30.0])
    np.testing.assert_allclose(
        result.intensity,
        np.repeat(result.intensity[:, :1], result.intensity.shape[1], axis=1),
    )
    assert result.apex_intensity == pytest.approx(10.0)
    assert np.nanmax(result.intensity) == pytest.approx(30.0)


def test_deleted_basal_slice_promotes_last_retained_slice_to_outer_ring():
    study, landmarks = _constant_slice_study()

    result = compute_lge_bullseye(
        study,
        landmarks,
        angle_count=24,
        radial_count=4,
        angular_smoothing_sigma_deg=0.0,
        radial_smoothing_sigma=0.0,
        six_segment_warp=False,
        deleted_slice_indices={2},
    )

    np.testing.assert_allclose(result.intensity[:, 0], [12.5, 15.0, 17.5, 20.0])
    np.testing.assert_allclose(result.intensity[-1], 20.0)


def test_lge_outputs_include_numeric_values_and_png(tmp_path):
    study, landmarks = _constant_slice_study()
    result = compute_lge_bullseye(
        study,
        landmarks,
        angle_count=24,
        radial_count=9,
        angular_smoothing_sigma_deg=0.0,
        radial_smoothing_sigma=0.0,
        six_segment_warp=False,
    )
    npz_path = save_lge_bullseye_npz(tmp_path / "lge_bullseye.npz", result)
    png_path = tmp_path / "lge_bullseye.png"
    figure, axis = plot_lge_bullseye(result, png_path, show=False)
    figure.canvas.draw()
    assert {text.get_text() for text in axis.texts} == {
        str(segment) for segment in range(1, 18)
    }
    assert len(axis.patches) == 1

    with np.load(npz_path) as payload:
        np.testing.assert_array_equal(payload["intensity"], result.intensity)
        assert float(payload["apex_intensity"]) == pytest.approx(10.0)
    assert png_path.is_file() and png_path.stat().st_size > 1000
