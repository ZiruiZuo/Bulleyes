import numpy as np
import nibabel as nib

from cardiac_bullseye.short_axis.geometry import (
    LongAxisPlan,
    generate_short_axis_stack,
    undirected_axis_angle_deg,
)
from cardiac_bullseye.short_axis.nifti import (
    constrained_reference_x_axis,
    recenter_affine_to_reference,
    voxel_grid_center_world,
)


def _plan() -> LongAxisPlan:
    return LongAxisPlan(
        vla_x_axis=np.array([0.0, 0.0, 1.0]),
        vla_y_axis=np.array([1.0, 0.0, 0.0]),
        vla_normal=np.array([0.0, 1.0, 0.0]),
        hla_x_axis=np.array([0.0, 1.0, 0.0]),
        hla_y_axis=np.array([0.0, 0.0, 1.0]),
        hla_normal=np.array([1.0, 0.0, 0.0]),
        apex=np.array([8.0, 8.0, 2.0]),
        valve=np.array([8.0, 8.0, 12.0]),
        long_axis=np.array([0.0, 0.0, 1.0]),
    )


def test_reference_axis_controls_only_inplane_orientation():
    volume = np.zeros((16, 16, 16), dtype=np.float32)
    reference_x = np.array([1.0, 1.0, 0.2])

    result = generate_short_axis_stack(
        volume,
        np.eye(4),
        _plan(),
        0.0,
        10.0,
        5.0,
        1.0,
        fov_mm=(12.0, 12.0),
        reference_x_axis=reference_x,
    )

    axes = result.affine[:3, :3] / np.linalg.norm(result.affine[:3, :3], axis=0)
    expected_x = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    np.testing.assert_allclose(axes[:, 0], expected_x, atol=1e-7)
    np.testing.assert_allclose(axes[:, 2], (0.0, 0.0, 1.0), atol=1e-7)
    np.testing.assert_allclose(axes.T @ axes, np.eye(3), atol=1e-7)


def test_reference_axis_parallel_to_long_axis_is_rejected():
    volume = np.zeros((16, 16, 16), dtype=np.float32)

    try:
        generate_short_axis_stack(
            volume,
            np.eye(4),
            _plan(),
            0.0,
            10.0,
            5.0,
            1.0,
            reference_x_axis=(0.0, 0.0, 1.0),
        )
    except ValueError as error:
        assert "reference" in str(error)
    else:
        raise AssertionError("parallel reference axis should be rejected")


def test_constrained_reference_axis_keeps_long_axis_and_matches_slicer_codes():
    """A new Apex/Base selection must not move the LGE views to other panes."""

    reference_affine = np.eye(4)
    reference_affine[:3, :3] = np.asarray(
        [
            [-0.859565973, 1.30360317, 3.38281775],
            [-1.36231744, 0.16511856, -4.12719393],
            [-0.989798248, -1.35934448, 2.74277449],
        ]
    )
    selected_long_axis = np.asarray(
        [0.5654110233, -0.6225232479, 0.5410870361]
    )

    x_axis = constrained_reference_x_axis(reference_affine, selected_long_axis)
    z_axis = selected_long_axis / np.linalg.norm(selected_long_axis)
    y_axis = np.cross(z_axis, x_axis)
    candidate_affine = np.eye(4)
    candidate_affine[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))

    assert nib.aff2axcodes(candidate_affine) == nib.aff2axcodes(reference_affine)
    np.testing.assert_allclose(candidate_affine[:3, 2], selected_long_axis, atol=1e-9)
    np.testing.assert_allclose(
        candidate_affine[:3, :3].T @ candidate_affine[:3, :3],
        np.eye(3),
        atol=1e-9,
    )


def test_recenter_affine_moves_grid_center_to_reference_without_changing_axes():
    """Every Save must retain LGE centering without changing sampled data axes."""

    source_affine = np.asarray(
        [
            [0.0, -2.0, 0.0, 30.0],
            [2.0, 0.0, 0.0, 40.0],
            [0.0, 0.0, 5.0, 50.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    reference_affine = np.eye(4)
    reference_affine[:3, 3] = [-100.0, 20.0, 8.0]
    source_shape = (5, 7, 3)
    reference_shape = (9, 11, 5)

    centered = recenter_affine_to_reference(
        source_affine,
        source_shape,
        reference_affine,
        reference_shape,
    )

    np.testing.assert_allclose(centered[:3, :3], source_affine[:3, :3])
    np.testing.assert_allclose(
        voxel_grid_center_world(source_shape, centered),
        voxel_grid_center_world(reference_shape, reference_affine),
        atol=1e-9,
    )


def test_long_axis_difference_ignores_reference_normal_sign():
    assert undirected_axis_angle_deg((0.0, 0.0, 1.0), (0.0, 0.0, -2.0)) == 0.0
    assert undirected_axis_angle_deg((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)) == 90.0
