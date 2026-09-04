from __future__ import annotations

import json

import numpy as np

from cardiac_bullseye.short_axis.geometry import LongAxisPlan, StackResult
from cardiac_bullseye.short_axis.metadata import default_geometry_path, save_planning_geometry


def test_geometry_json_records_true_apex_and_slice_heights(tmp_path):
    plan = LongAxisPlan(
        vla_x_axis=np.array([1.0, 0.0, 0.0]),
        vla_y_axis=np.array([0.0, 0.0, 1.0]),
        vla_normal=np.array([0.0, -1.0, 0.0]),
        hla_x_axis=np.array([0.0, -1.0, 0.0]),
        hla_y_axis=np.array([0.0, 0.0, 1.0]),
        hla_normal=np.array([-1.0, 0.0, 0.0]),
        apex=np.array([10.0, 20.0, 30.0]),
        valve=np.array([10.0, 20.0, 50.0]),
        long_axis=np.array([0.0, 0.0, 1.0]),
    )
    affine = np.eye(4)
    affine[:3, 2] = [0.0, 0.0, 5.0]
    stack = StackResult(
        data=np.zeros((8, 9, 4), dtype=np.float32),
        affine=affine,
        positions_mm=np.array([2.0, 7.0, 12.0, 17.0]),
    )
    output = tmp_path / "geometry.json"
    save_planning_geometry(
        output,
        plan=plan,
        stack=stack,
        apex_t_mm=2.0,
        base_t_mm=20.0,
        source_image_path=tmp_path / "source.nii.gz",
        short_axis_path=tmp_path / "short_axis.nii.gz",
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == "convert2shortaxis.geometry"
    assert payload["true_apex_world_mm"] == [10.0, 20.0, 30.0]
    assert payload["selected_base_center_world_mm"] == [10.0, 20.0, 50.0]
    assert payload["slice_positions_from_true_apex_mm"] == [2.0, 7.0, 12.0, 17.0]
    assert payload["short_axis_shape"] == [8, 9, 4]
    assert default_geometry_path(tmp_path / "short_axis.nii.gz").name == "short_axis_geometry.json"
