import json

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from cardiac_bullseye.short_axis.geometry import LongAxisPlan, PlaneGrid
from cardiac_bullseye.short_axis.nifti import NiftiVolume, voxel_grid_center_world
from cardiac_bullseye.short_axis.planner import ShortAxisPlanner


def _oblique_plan() -> LongAxisPlan:
    long_axis = np.asarray([0.5654110233, -0.6225232479, 0.5410870361])
    long_axis /= np.linalg.norm(long_axis)
    apex = np.asarray([8.0, 8.0, 8.0])
    return LongAxisPlan(
        vla_x_axis=np.asarray([0.0, 0.0, 1.0]),
        vla_y_axis=np.asarray([1.0, 0.0, 0.0]),
        vla_normal=np.asarray([0.0, 1.0, 0.0]),
        hla_x_axis=np.asarray([0.0, 1.0, 0.0]),
        hla_y_axis=long_axis,
        hla_normal=np.asarray([1.0, 0.0, 0.0]),
        apex=apex,
        valve=apex + 10.0 * long_axis,
        long_axis=long_axis,
    )


def _reference_affine() -> np.ndarray:
    return np.asarray(
        [
            [-0.859565973, 1.30360317, 3.38281775, -141.510513],
            [-1.36231744, 0.16511856, -4.12719393, 141.861176],
            [-0.989798248, -1.35934448, 2.74277449, 218.159561],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def test_each_save_matches_reference_view_codes_center_and_mask_geometry(tmp_path):
    """Changing Apex/Base and saving again must reapply the complete rule."""

    header = nib.Nifti1Header()
    image = NiftiVolume(
        data=np.ones((16, 16, 16), dtype=np.float32),
        affine=np.eye(4),
        header=header,
        path=tmp_path / "source.nii.gz",
    )
    mask = NiftiVolume(
        data=np.ones((16, 16, 16), dtype=np.uint8),
        affine=np.eye(4),
        header=header,
        path=tmp_path / "mask.nii.gz",
    )
    reference = NiftiVolume(
        data=np.zeros((192, 192, 16), dtype=np.float32),
        affine=_reference_affine(),
        header=header,
        path=tmp_path / "lge_sax.nii.gz",
    )
    output = tmp_path / "short_axis.nii.gz"
    mask_output = tmp_path / "short_axis_mask.nii.gz"
    geometry_output = tmp_path / "geometry.json"
    planner = ShortAxisPlanner(
        image,
        output,
        mask,
        mask_output,
        geometry_output,
        slice_spacing_mm=5.0,
        pixel_spacing_mm=1.0,
        fov_mm=(12.0, 12.0),
        orientation_reference=reference,
    )
    planner.plan = _oblique_plan()
    planner.apex_t = 0.0
    planner.vla_grid = PlaneGrid(
        center=np.asarray([8.0, 8.0, 8.0]),
        x_axis=np.asarray([1.0, 0.0, 0.0]),
        y_axis=np.asarray([0.0, 1.0, 0.0]),
        width=13,
        height=13,
        pixel_spacing=1.0,
    )
    planner.hla_grid = PlaneGrid(
        center=np.asarray([8.0, 8.0, 8.0]),
        x_axis=np.asarray([1.0, 0.0, 0.0]),
        y_axis=np.asarray([0.0, 0.0, 1.0]),
        width=13,
        height=13,
        pixel_spacing=1.0,
    )

    try:
        planner._render_planning()
        assert "Long-axis difference to LGE reference: 6.10°" in planner.title.get_text()
        planner._render_planning()
        for base_t in (10.0, 5.0):
            planner.base_t = base_t
            planner._save()

            saved = nib.load(str(output))
            saved_mask = nib.load(str(mask_output))
            assert nib.aff2axcodes(saved.affine) == nib.aff2axcodes(reference.affine)
            np.testing.assert_allclose(saved_mask.affine, saved.affine, atol=1e-6)
            np.testing.assert_allclose(
                voxel_grid_center_world(saved.shape, saved.affine),
                voxel_grid_center_world(reference.data.shape, reference.affine),
                atol=1e-5,
            )
            payload = json.loads(geometry_output.read_text(encoding="utf-8"))
            np.testing.assert_allclose(payload["short_axis_affine"], saved.affine, atol=1e-5)
            assert "short_axis_sampling_affine" in payload
            assert abs(payload["reference_long_axis_difference_deg"] - 6.0992) < 1e-3
    finally:
        plt.close(planner.fig)
