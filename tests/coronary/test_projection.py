from __future__ import annotations

import json

import matplotlib
import nibabel as nib
import numpy as np

matplotlib.use("Agg")

from cardiac_bullseye.coronary.projection import (
    CardiacGeometry,
    CoronaryPath,
    _voxel_graph_paths,
    bullseye_slice_radii,
    load_coronary_paths_from_nifti,
    project_coronary_paths,
)
from cardiac_bullseye.territory.mapping import build_bullseye
from cardiac_bullseye.territory.models import BullseyeResult, LandmarkPair, StudyData
from cardiac_bullseye.territory.plotting import (
    CORONARY_AUTO_COLORS,
    plot_territory_bullseye,
)


def _geometry() -> CardiacGeometry:
    return CardiacGeometry(
        true_apex_world=np.array([0.0, 0.0, 0.0]),
        base_center_world=np.array([0.0, 0.0, 10.0]),
        long_axis=np.array([0.0, 0.0, 1.0]),
        short_axis_x=np.array([1.0, 0.0, 0.0]),
        short_axis_y=np.array([0.0, 1.0, 0.0]),
        short_axis_affine=np.eye(4),
        short_axis_shape=(9, 9, 5),
        slice_positions_mm=np.linspace(0.0, 10.0, 5),
        base_height_mm=10.0,
    )


def _study_and_landmarks():
    mr = np.zeros((5, 9, 9), dtype=float)
    mask = np.ones_like(mr, dtype=np.uint8)
    study = StudyData(mr, mask, np.arange(5.0))
    pair = LandmarkPair(aip=(5.0, 2.0), iip=(5.0, 6.0))
    return study, [pair] * 5


def test_true_height_controls_background_and_coronary_radius():
    geometry = _geometry()
    np.testing.assert_allclose(bullseye_slice_radii(geometry), [0, 1, 2, 3, 4])

    study, landmarks = _study_and_landmarks()
    path = CoronaryPath(
        label=1,
        world_points=np.column_stack(
            [np.zeros(9), np.full(9, 2.0), np.linspace(1.0, 9.0, 9)]
        ),
    )
    projected = project_coronary_paths(
        [path], geometry, study, landmarks, six_segment_warp=True
    )
    assert len(projected) == 1
    radii = np.linalg.norm(projected[0].xy, axis=1)
    np.testing.assert_allclose(radii, np.linspace(0.4, 3.6, 9), atol=1.0e-7)
    assert np.all(projected[0].xy[:, 0] < 0)


def test_extract_ordered_centerline_from_nifti(tmp_path):
    data = np.zeros((9, 9, 11), dtype=np.uint8)
    data[4, 6, 1:10] = 1
    path = tmp_path / "coronary.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), path)
    paths = load_coronary_paths_from_nifti(
        path, min_path_length_mm=2.0, resample_step_mm=1.0
    )
    assert len(paths) == 1
    assert paths[0].label == 1
    assert np.linalg.norm(paths[0].world_points[-1] - paths[0].world_points[0]) == 8.0


def test_binary_branches_receive_different_display_labels(tmp_path):
    data = np.zeros((9, 9, 9), dtype=np.uint8)
    data[4, 4, 1:8] = 1
    data[4, 4:8, 4] = 1
    path = tmp_path / "branches.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), path)
    paths = load_coronary_paths_from_nifti(
        path, min_path_length_mm=1.0, resample_step_mm=1.0
    )
    assert len(paths) >= 2
    assert len({item.label for item in paths}) == len(paths)


def test_corner_does_not_become_a_spurious_closed_cycle():
    skeleton = np.zeros((3, 3, 1), dtype=bool)
    skeleton[0, 0, 0] = True
    skeleton[1, 0, 0] = True
    skeleton[1, 1, 0] = True
    paths = _voxel_graph_paths(skeleton)
    assert len(paths) == 1
    assert len(paths[0]) == 3
    assert not np.array_equal(paths[0][0], paths[0][-1])


def test_explicit_slice_radii_and_coronary_plot(tmp_path):
    probabilities = np.zeros((2, 12, 3), dtype=float)
    probabilities[0, :, 0] = 1.0
    probabilities[1, :, 1] = 1.0
    result = build_bullseye(
        probabilities,
        [0.0, 1.0],
        source_radii=[0.0, 4.0],
        radial_count=7,
        radial_smoothing_sigma=0.0,
    )
    # The first annular ring is only 25% of the physical apex-base height.
    assert np.all(result.territory_labels[0] == 1)
    assert np.all(result.territory_labels[-1] == 2)

    line = CoronaryPath(
        label=1,
        world_points=np.array([[0.0, 2.0, 2.5], [0.0, 2.0, 7.5]]),
    )
    geometry = _geometry()
    study, landmarks = _study_and_landmarks()
    projected = project_coronary_paths([line], geometry, study, landmarks)
    output = tmp_path / "overlay.png"
    plot_territory_bullseye(result, output, coronary_paths=projected, show=False)
    assert output.is_file() and output.stat().st_size > 1000


def test_coronary_above_base_is_drawn_outside_bullseye(tmp_path):
    probabilities = np.zeros((2, 24, 3), dtype=float)
    probabilities[:, :, 0] = 1.0
    result = build_bullseye(
        probabilities,
        [0.0, 1.0],
        source_radii=[0.0, 4.0],
        radial_count=7,
        radial_smoothing_sigma=0.0,
    )
    geometry = _geometry()
    study, landmarks = _study_and_landmarks()
    paths = [
        CoronaryPath(
            label=1,
            world_points=np.array([[0.0, 2.0, 8.0], [0.0, 2.0, 12.0]]),
        ),
        CoronaryPath(
            label=2,
            world_points=np.array([[2.0, 0.0, 8.0], [2.0, 0.0, 11.0]]),
        ),
    ]
    projected = project_coronary_paths(paths, geometry, study, landmarks)
    assert np.isclose(
        max(np.linalg.norm(item.xy, axis=1).max() for item in projected), 4.8
    )

    figure, axis = plot_territory_bullseye(
        result,
        tmp_path / "outside.png",
        coronary_paths=projected,
        coronary_label_names={1: "Coronary LAD", 2: "Coronary LCX"},
        show=False,
    )
    assert axis.get_xlim()[1] > 4.8
    coronary_lines = [line for line in axis.lines if line.get_zorder() == 20]
    assert {line.get_color() for line in coronary_lines} == set(
        CORONARY_AUTO_COLORS[:2]
    )
    legend_names = {text.get_text() for text in axis.get_legend().get_texts()}
    assert {"Coronary LAD", "Coronary LCX"}.issubset(legend_names)
