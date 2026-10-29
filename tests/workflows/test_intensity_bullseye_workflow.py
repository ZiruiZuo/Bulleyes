from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from cardiac_bullseye.coronary import ProjectedCoronaryPath
from cardiac_bullseye.territory.models import LandmarkPair, StudyData
from cardiac_bullseye.workflows import bullseye_frame as frame_module


def _study_and_landmarks():
    size = 25
    y, x = np.ogrid[:size, :size]
    ring = ((x - 12) ** 2 + (y - 12) ** 2 >= 25) & (
        (x - 12) ** 2 + (y - 12) ** 2 <= 81
    )
    mask = np.repeat(ring[None, :, :], 3, axis=0).astype(np.uint8)
    image = np.zeros_like(mask, dtype=float)
    for index, value in enumerate((10.0, 20.0, 30.0)):
        image[index, ring] = value
    study = StudyData(image, mask, np.asarray([0.0, 6.0, 12.0]))
    pair = LandmarkPair(aip=(12.0, 5.0), iip=(12.0, 19.0))
    return study, [pair, pair, pair]


@pytest.mark.parametrize("explicit_path", [False, True])
def test_intensity_workflow_uses_shared_landmarks_and_mask_anchored_radii(
    monkeypatch, app_config, explicit_path
):
    from cardiac_bullseye.workflows import intensity_bullseye as module

    study, landmarks = _study_and_landmarks()
    config = replace(
        app_config,
        bullseye=replace(
            app_config.bullseye,
            angle_count=24,
            radial_count=4,
            angular_smoothing_sigma_deg=0.0,
            radial_smoothing_sigma=0.0,
            six_segment_warp=False,
            show_final_figure=False,
        ),
    )
    selected = {}
    expected_landmarks = config.case.output_path("landmarks.json")
    if explicit_path:
        expected_landmarks = config.case.output_directory / "shared.json"
        config = replace(
            config,
            bullseye=replace(config.bullseye, landmarks_path=expected_landmarks),
        )
    plotted = {}
    monkeypatch.setattr(module, "load_intensity_study_nifti", lambda *_args: study)

    def select_single(selected_study, output_path, **kwargs):
        selected.update(study=selected_study, path=output_path, kwargs=kwargs)
        return landmarks

    monkeypatch.setattr(
        frame_module, "select_single_reference_interactively", select_single
    )
    monkeypatch.setattr(frame_module, "load_deleted_slice_indices", lambda *_args: set())
    monkeypatch.setattr(
        frame_module, "load_center_overrides_json", lambda *_args, **_kwargs: [None] * 3
    )

    def plot(result, path, **kwargs):
        plotted.update(result=result, path=path, kwargs=kwargs)

    monkeypatch.setattr(module, "plot_lge_bullseye", plot)

    outputs = module.generate_intensity_bullseye(
        config, landmark_mode="single-reference"
    )

    expected_figure = config.case.output_path("intensity_bullseye.png")
    assert selected == {
        "study": study,
        "path": expected_landmarks,
        "kwargs": {
            "resume_existing": False,
            "transpose_for_display": True,
            "cancel_is_error": False,
        },
    }
    assert outputs.landmarks == expected_landmarks
    assert outputs.figure == expected_figure
    assert plotted["path"] == expected_figure
    assert plotted["kwargs"]["colormap"] == "CMRmap"
    assert plotted["kwargs"]["colorbar_label"] == "Image intensity (native units)"
    assert plotted["result"].apex_intensity == 10.0
    np.testing.assert_allclose(
        plotted["result"].intensity[:, 0],
        [10.0, 16.6666666667, 23.3333333333, 30.0],
    )


def test_intensity_workflow_opens_annotation_when_reuse_file_is_missing(
    monkeypatch, app_config
):
    from cardiac_bullseye.workflows import intensity_bullseye as module

    study, landmarks = _study_and_landmarks()
    config = replace(
        app_config,
        bullseye=replace(app_config.bullseye, reuse_saved_points=True),
    )
    calls = []
    monkeypatch.setattr(module, "load_intensity_study_nifti", lambda *_args: study)
    monkeypatch.setattr(
        frame_module,
        "load_saved_landmarks",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("missing file must not be loaded")
        ),
    )

    def select_each(*_args, **_kwargs):
        calls.append("interactive")
        return landmarks

    monkeypatch.setattr(frame_module, "select_landmarks_interactively", select_each)
    monkeypatch.setattr(frame_module, "load_deleted_slice_indices", lambda *_args: set())
    monkeypatch.setattr(
        frame_module, "load_center_overrides_json", lambda *_args, **_kwargs: [None] * 3
    )
    monkeypatch.setattr(module, "plot_lge_bullseye", lambda *_args, **_kwargs: None)

    module.generate_intensity_bullseye(config)

    assert calls == ["interactive"]


def test_intensity_workflow_reuses_existing_shared_landmark_file(
    monkeypatch, app_config
):
    from cardiac_bullseye.workflows import intensity_bullseye as module

    study, landmarks = _study_and_landmarks()
    config = replace(
        app_config,
        bullseye=replace(app_config.bullseye, reuse_saved_points=True),
    )
    landmark_path = config.case.output_path("landmarks.json")
    landmark_path.touch()
    calls = []
    monkeypatch.setattr(module, "load_intensity_study_nifti", lambda *_args: study)

    def load_saved(selected_study, path, **_kwargs):
        calls.append((selected_study, path))
        return landmarks

    monkeypatch.setattr(frame_module, "load_saved_landmarks", load_saved)
    monkeypatch.setattr(
        frame_module,
        "select_landmarks_interactively",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("annotation UI must not open")
        ),
    )
    monkeypatch.setattr(frame_module, "load_deleted_slice_indices", lambda *_args: set())
    monkeypatch.setattr(
        frame_module, "load_center_overrides_json", lambda *_args, **_kwargs: [None] * 3
    )
    monkeypatch.setattr(module, "plot_lge_bullseye", lambda *_args, **_kwargs: None)

    module.generate_intensity_bullseye(config)

    assert calls == [(study, landmark_path)]


def test_intensity_workflow_zero_to_four_projection_uses_real_center(
    monkeypatch, app_config
):
    from cardiac_bullseye.workflows import intensity_bullseye as module

    study, landmarks = _study_and_landmarks()
    y, x = np.nonzero(study.territory_mask[0] > 0)
    study.mr[0, y, x] = x.astype(float)
    config = replace(
        app_config,
        bullseye=replace(
            app_config.bullseye,
            angle_count=24,
            radial_count=5,
            angular_smoothing_sigma_deg=0.0,
            radial_smoothing_sigma=0.0,
            six_segment_warp=False,
            show_final_figure=False,
        ),
    )
    plotted = {}
    monkeypatch.setattr(module, "load_intensity_study_nifti", lambda *_args: study)
    monkeypatch.setattr(
        frame_module, "select_landmarks_interactively", lambda *_args, **_kwargs: landmarks
    )
    monkeypatch.setattr(frame_module, "load_deleted_slice_indices", lambda *_args: set())
    monkeypatch.setattr(
        frame_module, "load_center_overrides_json", lambda *_args, **_kwargs: [None] * 3
    )
    monkeypatch.setattr(
        module,
        "plot_lge_bullseye",
        lambda result, *_args, **_kwargs: plotted.update(result=result),
    )

    module.generate_intensity_bullseye(config, projection="0-4")

    result = plotted["result"]
    assert result.radial_positions[0] == 0.0
    np.testing.assert_allclose(result.intensity[0], np.mean(result.intensity[0]))
    radius_one = np.flatnonzero(np.isclose(result.radial_positions, 1.0))[0]
    assert np.ptp(result.intensity[radius_one]) > 0.0


def test_intensity_workflow_optionally_projects_coronaries_on_shared_frame(
    monkeypatch, app_config
):
    from cardiac_bullseye.workflows import intensity_bullseye as module

    study, landmarks = _study_and_landmarks()
    config = replace(
        app_config,
        bullseye=replace(
            app_config.bullseye,
            intensity_coronary_overlay=True,
            coronary_label_colors={2: "#123456"},
            coronary_label_names={2: "Coronary LCX"},
            coronary_linewidth=2.7,
            show_final_figure=False,
        ),
    )
    geometry = type(
        "Geometry",
        (),
        {"slice_positions_mm": np.asarray([10.0, 20.0, 40.0])},
    )()
    projected = [
        ProjectedCoronaryPath(
            label=2,
            xy=np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=float),
        )
    ]
    calls = {}
    monkeypatch.setattr(module, "load_intensity_study_nifti", lambda *_args: study)
    monkeypatch.setattr(
        frame_module, "select_landmarks_interactively", lambda *_args, **_kwargs: landmarks
    )
    monkeypatch.setattr(frame_module, "load_deleted_slice_indices", lambda *_args: set())
    monkeypatch.setattr(
        frame_module, "load_center_overrides_json", lambda *_args, **_kwargs: [None] * 3
    )
    monkeypatch.setattr(
        module,
        "load_cardiac_geometry",
        lambda path: calls.update(geometry_path=path) or geometry,
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "validate_geometry_against_short_axis_nifti",
        lambda selected_geometry, path: calls.update(
            validated=(selected_geometry, path)
        ),
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "load_coronary_paths_from_nifti",
        lambda path, **kwargs: calls.update(coronary_load=(path, kwargs)) or [object()],
        raising=False,
    )

    def project(paths, selected_geometry, selected_study, selected_landmarks, **kwargs):
        calls["projection"] = (
            paths,
            selected_geometry,
            selected_study,
            selected_landmarks,
            kwargs,
        )
        return projected

    monkeypatch.setattr(module, "project_coronary_paths", project, raising=False)
    monkeypatch.setattr(
        module,
        "plot_lge_bullseye",
        lambda _result, _path, **kwargs: calls.update(plot=kwargs),
    )

    module.generate_intensity_bullseye(config, projection="0-4")

    assert calls["geometry_path"] == config.bullseye.geometry_json
    assert calls["validated"] == (geometry, config.bullseye.short_axis_mr)
    assert calls["coronary_load"][0] == config.bullseye.coronary_mask
    radial_frame = calls["projection"][4]["radial_frame"]
    assert radial_frame.minimum_radius == 0.0
    np.testing.assert_allclose(radial_frame.source_radii, [0.0, 4.0 / 3.0, 4.0])
    assert calls["plot"]["coronary_paths"] == projected
    assert calls["plot"]["coronary_label_colors"] == {2: "#123456"}
    assert calls["plot"]["coronary_label_names"] == {2: "Coronary LCX"}
    assert calls["plot"]["coronary_linewidth"] == pytest.approx(2.7)
