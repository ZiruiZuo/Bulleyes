from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest

from cardiac_bullseye.intensity import load_intensity_study_nifti


def test_intensity_loader_accepts_any_nonzero_mask_labels_without_cropping(tmp_path):
    image = np.arange(5 * 6 * 3, dtype=np.float32).reshape(5, 6, 3)
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[1:4, 2:5, 0] = 1
    mask[1:4, 2:5, 2] = 9
    affine = np.diag([0.8, 0.9, 6.0, 1.0])
    image_path = tmp_path / "image.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(image, affine), image_path)
    nib.save(nib.Nifti1Image(mask, affine), mask_path)

    study = load_intensity_study_nifti(image_path, mask_path)

    assert study.mr.shape == (3, 5, 6)
    np.testing.assert_array_equal(study.mr, np.moveaxis(image, 2, 0))
    np.testing.assert_array_equal(
        study.territory_mask, np.moveaxis(mask != 0, 2, 0)
    )
    np.testing.assert_allclose(study.slice_positions, [0.0, 6.0, 12.0])


def test_intensity_loader_rejects_different_image_and_mask_grids(tmp_path):
    image = np.zeros((5, 6, 3), dtype=np.float32)
    mask = np.ones_like(image, dtype=np.uint8)
    image_path = tmp_path / "image.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(image, np.eye(4)), image_path)
    shifted = np.eye(4)
    shifted[0, 3] = 1.0
    nib.save(nib.Nifti1Image(mask, shifted), mask_path)

    with pytest.raises(ValueError, match="affine"):
        load_intensity_study_nifti(image_path, mask_path)
