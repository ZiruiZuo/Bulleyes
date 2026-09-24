from __future__ import annotations

import nibabel as nib
import numpy as np

import cardiac_bullseye.short_axis as short_axis
from cardiac_bullseye.short_axis.geometry import LongAxisPlan, generate_short_axis_stack


def _plan() -> LongAxisPlan:
    apex = np.asarray([8.0, 8.0, 2.0])
    long_axis = np.asarray([0.0, 0.0, 1.0])
    return LongAxisPlan(
        vla_x_axis=np.asarray([1.0, 0.0, 0.0]),
        vla_y_axis=np.asarray([0.0, 0.0, 1.0]),
        vla_normal=np.asarray([0.0, -1.0, 0.0]),
        hla_x_axis=np.asarray([0.0, -1.0, 0.0]),
        hla_y_axis=long_axis,
        hla_normal=np.asarray([-1.0, 0.0, 0.0]),
        apex=apex,
        valve=apex + 10.0 * long_axis,
        long_axis=long_axis,
    )


def test_saved_short_axis_points_regenerate_mr_and_changed_mask_without_ui(tmp_path):
    assert hasattr(short_axis, "save_short_axis_points")
    assert hasattr(short_axis, "convert_to_short_axis_from_saved_points")
    data = np.arange(16**3, dtype=np.float32).reshape((16, 16, 16))
    changed_mask = np.zeros_like(data, dtype=np.uint8)
    changed_mask[5:11, 5:11, 3:14] = 7
    image_path = tmp_path / "source.nii.gz"
    mask_path = tmp_path / "changed_labels.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), image_path)
    nib.save(nib.Nifti1Image(changed_mask, np.eye(4)), mask_path)
    plan = _plan()
    sampled = generate_short_axis_stack(
        data,
        np.eye(4),
        plan,
        0.0,
        10.0,
        5.0,
        1.0,
        (10.0, 10.0),
    )
    points_path = tmp_path / "case_short_axis_points.json"
    short_axis.save_short_axis_points(
        points_path,
        plan=plan,
        apex_t_mm=0.0,
        base_t_mm=10.0,
        slice_spacing_mm=5.0,
        pixel_spacing_mm=1.0,
        fov_mm=(10.0, 10.0),
        source_shape=data.shape,
        source_affine=np.eye(4),
        reference_x_axis=None,
        display_affine=sampled.affine,
    )
    output_path = tmp_path / "short_axis.nii.gz"
    mask_output_path = tmp_path / "short_axis_mask.nii.gz"
    geometry_path = tmp_path / "geometry.json"

    short_axis.convert_to_short_axis_from_saved_points(
        image_path=image_path,
        output_path=output_path,
        points_path=points_path,
        mask_path=mask_path,
        mask_output_path=mask_output_path,
        geometry_output_path=geometry_path,
    )

    output = nib.load(output_path)
    mask_output = nib.load(mask_output_path)
    assert output.shape == (11, 11, 3)
    assert mask_output.shape == output.shape
    np.testing.assert_allclose(mask_output.affine, output.affine)
    assert set(np.unique(np.asarray(mask_output.dataobj))) <= {0.0, 7.0}
    assert geometry_path.is_file()


def test_saved_short_axis_points_reject_a_different_source_grid(tmp_path):
    assert hasattr(short_axis, "save_short_axis_points")
    assert hasattr(short_axis, "load_short_axis_points")
    plan = _plan()
    points_path = tmp_path / "case_short_axis_points.json"
    short_axis.save_short_axis_points(
        points_path,
        plan=plan,
        apex_t_mm=0.0,
        base_t_mm=10.0,
        slice_spacing_mm=5.0,
        pixel_spacing_mm=1.0,
        fov_mm=(10.0, 10.0),
        source_shape=(16, 16, 16),
        source_affine=np.eye(4),
        reference_x_axis=None,
        display_affine=np.eye(4),
    )

    try:
        short_axis.load_short_axis_points(
            points_path,
            source_shape=(16, 16, 16),
            source_affine=np.diag([2.0, 1.0, 1.0, 1.0]),
        )
    except ValueError as error:
        assert "source image grid" in str(error)
    else:
        raise AssertionError("a different source affine must be rejected")
