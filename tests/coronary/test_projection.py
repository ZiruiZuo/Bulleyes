from __future__ import annotations

import json
from dataclasses import replace

import matplotlib
import nibabel as nib
import numpy as np
import pytest

matplotlib.use("Agg")

from cardiac_bullseye.coronary.projection import (
    CardiacGeometry,
    CoronaryPath,
    _acyclic_voxel_graph_paths,
    _path_endpoint_degrees,
    _voxel_graph_paths,
    bullseye_slice_radii,
    load_coronary_paths_from_nifti,
    project_coronary_paths,
)
from cardiac_bullseye.territory.mapping import build_bullseye
from cardiac_bullseye.territory.models import BullseyeResult, LandmarkPair, StudyData
from cardiac_bullseye.territory.radial import build_mask_anchored_radial_frame
from cardiac_bullseye.territory.plotting import plot_territory_bullseye


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


def test_basal_delete_rescales_coronary_to_same_effective_base():
    geometry = _geometry()
    study, landmarks = _study_and_landmarks()
    landmarks[-1] = None
    path = CoronaryPath(
        label=1,
        world_points=np.asarray(
            [[0.0, 2.0, 3.75], [0.0, 2.0, 7.5], [0.0, 2.0, 9.0]]
        ),
    )

    projected = project_coronary_paths(
        [path],
        geometry,
        study,
        landmarks,
        deleted_slice_indices={4},
    )

    radii = np.linalg.norm(projected[0].xy, axis=1)
    np.testing.assert_allclose(radii, [2.0, 4.0, 4.8], atol=1.0e-7)


def test_coronary_projection_uses_mask_anchors_and_omits_radius_below_one():
    geometry = _geometry()
    study, landmarks = _study_and_landmarks()
    study.territory_mask[0] = 0
    frame = build_mask_anchored_radial_frame(
        study.territory_mask,
        geometry.slice_positions_mm,
    )
    path = CoronaryPath(
        label=1,
        world_points=np.asarray(
            [
                [0.0, 2.0, 1.0],
                [0.0, 2.0, 2.5],
                [0.0, 2.0, 5.0],
                [0.0, 2.0, 10.0],
                [0.0, 2.0, 12.5],
            ]
        ),
    )

    projected = project_coronary_paths(
        [path], geometry, study, landmarks, radial_frame=frame
    )

    assert len(projected) == 1
    radii = np.linalg.norm(projected[0].xy, axis=1)
    np.testing.assert_allclose(radii, [1.0, 2.0, 4.0, 5.0])


def test_zero_to_four_coronary_projection_keeps_apical_mask_anchor():
    geometry = _geometry()
    study, landmarks = _study_and_landmarks()
    study.territory_mask[0] = 0
    frame = build_mask_anchored_radial_frame(
        study.territory_mask,
        geometry.slice_positions_mm,
        minimum_radius=0.0,
    )
    path = CoronaryPath(
        label=1,
        world_points=np.asarray(
            [
                [0.0, 2.0, 1.0],
                [0.0, 2.0, 2.5],
                [0.0, 2.0, 5.0],
                [0.0, 2.0, 10.0],
                [0.0, 2.0, 12.5],
            ]
        ),
    )

    projected = project_coronary_paths(
        [path], geometry, study, landmarks, radial_frame=frame
    )

    assert len(projected) == 1
    radii = np.linalg.norm(projected[0].xy, axis=1)
    np.testing.assert_allclose(radii, [0.0, 4 / 3, 4.0, 16 / 3])


def test_delete_rejects_non_basal_slice():
    geometry = _geometry()
    study, landmarks = _study_and_landmarks()
    path = CoronaryPath(
        label=1,
        world_points=np.asarray([[0.0, 2.0, 2.0], [0.0, 2.0, 8.0]]),
    )

    with pytest.raises(ValueError, match="contiguous basal tail"):
        project_coronary_paths(
            [path],
            geometry,
            study,
            landmarks,
            deleted_slice_indices={2},
        )


def test_display_recentering_does_not_change_coronary_projection():
    """Display-only LGE centering must not corrupt original-world mapping."""

    geometry = _geometry()
    display_affine = geometry.short_axis_affine.copy()
    display_affine[:3, 3] = [-300.0, 120.0, 80.0]
    recentered_geometry = replace(
        geometry,
        short_axis_affine=display_affine,
        short_axis_sampling_affine=geometry.short_axis_affine,
    )
    study, landmarks = _study_and_landmarks()
    path = CoronaryPath(
        label=1,
        world_points=np.asarray([[0.0, 2.0, 2.0], [0.0, 2.0, 8.0]]),
    )

    baseline = project_coronary_paths([path], geometry, study, landmarks)
    recentered = project_coronary_paths(
        [path], recentered_geometry, study, landmarks
    )

    assert len(recentered) == len(baseline)
    np.testing.assert_allclose(recentered[0].xy, baseline[0].xy, atol=1e-9)


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


def test_multilabel_centerlines_share_the_bifurcation_node(tmp_path):
    data = np.zeros((9, 9, 9), dtype=np.uint8)
    data[4, 4, 1:8] = 1
    data[4, 5:8, 4] = 2
    path = tmp_path / "labelled_bifurcation.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), path)

    paths = load_coronary_paths_from_nifti(
        path,
        mask_is_centerline=True,
        min_path_length_mm=0.1,
        resample_step_mm=0.0,
    )

    endpoints_by_label = {
        label: {
            tuple(point)
            for item in paths
            if item.label == label
            for point in (item.world_points[0], item.world_points[-1])
        }
        for label in (1, 2)
    }
    assert endpoints_by_label[1] & endpoints_by_label[2] == {(4.0, 4.5, 4.0)}


def test_label_transition_inside_a_path_keeps_both_vessel_labels(tmp_path):
    data = np.zeros((9, 9, 9), dtype=np.uint8)
    data[4, 4, 1:4] = 1
    data[4, 4, 4:8] = 2
    path = tmp_path / "label_transition.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), path)

    paths = load_coronary_paths_from_nifti(
        path,
        mask_is_centerline=True,
        min_path_length_mm=0.1,
        resample_step_mm=0.0,
    )

    assert {item.label for item in paths} == {1, 2}
    endpoints_by_label = {
        label: {
            tuple(point)
            for item in paths
            if item.label == label
            for point in (item.world_points[0], item.world_points[-1])
        }
        for label in (1, 2)
    }
    assert endpoints_by_label[1] & endpoints_by_label[2] == {(4.0, 4.0, 3.5)}


def test_short_internal_bridge_is_not_removed_by_length_filter(tmp_path):
    data = np.zeros((9, 9, 9), dtype=np.uint8)
    data[4, 0:5, 4] = 1
    data[1:5, 3, 4] = 1
    data[4, 4:9, 4] = 1
    data[4:8, 4, 4] = 1
    path = tmp_path / "short_internal_bridge.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), path)

    paths = load_coronary_paths_from_nifti(
        path,
        mask_is_centerline=True,
        min_path_length_mm=2.0,
        resample_step_mm=0.0,
    )

    bridge_endpoints = {(4.0, 3.0, 4.0), (4.0, 4.0, 4.0)}
    assert any(
        {tuple(item.world_points[0]), tuple(item.world_points[-1])}
        == bridge_endpoints
        for item in paths
    )


def test_extracted_coronary_graph_has_no_closed_cycles(tmp_path):
    data = np.zeros((9, 9, 7), dtype=np.uint8)
    data[3, 0:9, 3] = 1
    data[3:6, 2, 3] = 1
    data[3:6, 6, 3] = 1
    data[5, 2:7, 3] = 1
    path = tmp_path / "cycle_with_two_tails.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), path)

    paths = load_coronary_paths_from_nifti(
        path,
        mask_is_centerline=True,
        min_path_length_mm=0.1,
        resample_step_mm=0.0,
    )

    nodes = {
        tuple(point)
        for item in paths
        for point in item.world_points
    }
    edges = {
        frozenset((tuple(start), tuple(end)))
        for item in paths
        for start, end in zip(item.world_points[:-1], item.world_points[1:])
    }
    assert len(edges) == len(nodes) - 1
    assert (5.0, 4.0, 3.0) in nodes
    assert (3.0, 4.0, 3.0) not in nodes


def test_cycle_removal_prunes_new_leaf_that_was_not_an_original_endpoint():
    skeleton = np.zeros((13, 15, 1), dtype=bool)
    skeleton[2, 1:14, 0] = True
    skeleton[2:7, 3, 0] = True
    skeleton[2:7, 7, 0] = True
    skeleton[2:7, 11, 0] = True
    skeleton[6, 3:12, 0] = True

    paths = _acyclic_voxel_graph_paths(skeleton, np.eye(4))

    endpoint_degrees = _path_endpoint_degrees(paths)
    leaves = {point for point, degree in endpoint_degrees.items() if degree == 1}
    assert leaves == {(2, 1, 0), (2, 13, 0)}


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
    assert coronary_lines[0].get_color() == "#d62728"
    assert coronary_lines[1].get_color() == "#ff7f0e"
    legend_names = {text.get_text() for text in axis.get_legend().get_texts()}
    assert {"Coronary LAD", "Coronary LCX"}.issubset(legend_names)


def test_lcx_default_projection_color_is_orange_when_lad_is_absent(tmp_path):
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
    path = CoronaryPath(
        label=2,
        world_points=np.array([[2.0, 0.0, 4.0], [2.0, 0.0, 8.0]]),
    )
    projected = project_coronary_paths([path], geometry, study, landmarks)

    _, axis = plot_territory_bullseye(
        result,
        tmp_path / "lcx_only.png",
        coronary_paths=projected,
        show=False,
    )

    coronary_lines = [line for line in axis.lines if line.get_zorder() == 20]
    assert len(coronary_lines) == 1
    assert coronary_lines[0].get_color() == "#ff7f0e"
