import numpy as np
import pytest
from dataclasses import replace
from types import SimpleNamespace

from cardiac_bullseye.territory.models import LandmarkPair, StudyData


def test_bullseye_workflow_requests_a_fresh_transposed_nifti_annotation_view(
    monkeypatch, app_config
):
    from cardiac_bullseye.workflows import bullseye as module

    study = StudyData(
        mr=np.ones((1, 4, 6), dtype=float),
        territory_mask=np.ones((1, 4, 6), dtype=np.uint8),
        slice_positions=np.asarray([0.0]),
    )

    geometry = SimpleNamespace(slice_positions_mm=np.asarray([0.0]))
    monkeypatch.setattr(module, "load_cardiac_geometry", lambda _path: geometry)
    monkeypatch.setattr(
        module, "validate_geometry_against_short_axis_nifti", lambda *_args: None
    )
    monkeypatch.setattr(module, "load_study_nifti", lambda *_args, **_kwargs: study)
    def select_transposed_view(
        selected_study,
        _output_path,
        *,
        resume_existing,
        transpose_for_display,
    ):
        assert selected_study is study
        assert resume_existing is False
        assert transpose_for_display is True
        return [None]

    monkeypatch.setattr(
        module, "select_landmarks_interactively", select_transposed_view
    )

    with pytest.raises(RuntimeError, match="No slice has a valid AIP/IIP pair"):
        module.generate_bullseye(app_config)


def test_bullseye_workflow_reuses_landmarks_without_opening_annotation_ui(
    monkeypatch, app_config
):
    from cardiac_bullseye.workflows import bullseye as module

    config = replace(
        app_config,
        bullseye=replace(app_config.bullseye, reuse_saved_points=True),
    )
    study = StudyData(
        mr=np.ones((1, 4, 6), dtype=float),
        territory_mask=np.ones((1, 4, 6), dtype=np.uint8),
        slice_positions=np.asarray([0.0]),
    )
    geometry = SimpleNamespace(slice_positions_mm=np.asarray([0.0]))
    monkeypatch.setattr(module, "load_cardiac_geometry", lambda _path: geometry)
    monkeypatch.setattr(
        module, "validate_geometry_against_short_axis_nifti", lambda *_args: None
    )
    monkeypatch.setattr(module, "load_study_nifti", lambda *_args, **_kwargs: study)
    monkeypatch.setattr(
        module,
        "select_landmarks_interactively",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("annotation UI must not open")
        ),
    )
    assert hasattr(module, "load_saved_landmarks")
    monkeypatch.setattr(module, "load_saved_landmarks", lambda *_args: [None])

    with pytest.raises(RuntimeError, match="No slice has a valid AIP/IIP pair"):
        module.generate_bullseye(config)


def test_bullseye_workflow_shares_mask_anchored_frame(monkeypatch, app_config):
    from cardiac_bullseye.workflows import bullseye as module

    mask = np.zeros((3, 4, 4), dtype=np.uint8)
    mask[1:, 1:3, 1:3] = 1
    study = StudyData(
        mr=np.zeros_like(mask, dtype=float),
        territory_mask=mask,
        slice_positions=np.asarray([0.0, 1.0, 2.0]),
    )
    geometry = SimpleNamespace(
        slice_positions_mm=np.asarray([0.0, 5.0, 15.0]),
        base_height_mm=15.0,
        short_axis_sampling_affine=np.eye(4),
    )
    landmarks = [
        None,
        LandmarkPair(aip=(1.0, 0.0), iip=(1.0, 3.0)),
        LandmarkPair(aip=(1.0, 0.0), iip=(1.0, 3.0)),
    ]
    manual_centers = [None, (1.25, 1.5), None]
    captured = {}

    monkeypatch.setattr(module, "validate_config", lambda *_args: None)
    monkeypatch.setattr(module, "load_cardiac_geometry", lambda _path: geometry)
    monkeypatch.setattr(
        module, "validate_geometry_against_short_axis_nifti", lambda *_args: None
    )
    monkeypatch.setattr(module, "load_study_nifti", lambda *_args, **_kwargs: study)
    monkeypatch.setattr(
        module,
        "select_landmarks_interactively",
        lambda *_args, **_kwargs: landmarks,
    )
    monkeypatch.setattr(
        module, "load_deleted_slice_indices", lambda *_args, **_kwargs: set()
    )
    monkeypatch.setattr(
        module,
        "load_center_overrides_json",
        lambda *_args, **_kwargs: manual_centers,
    )
    monkeypatch.setattr(
        module, "load_coronary_paths_from_nifti", lambda *_args, **_kwargs: [object()]
    )

    def capture_coronary(*_args, **kwargs):
        captured["coronary_frame"] = kwargs.get("radial_frame")
        return [object()]

    def capture_territory(*_args, **kwargs):
        captured["territory_radii"] = np.asarray(kwargs["source_radii"])
        captured["territory_rescale"] = kwargs.get("rescale_deleted_base")
        captured["territory_centers"] = kwargs.get("center_overrides")
        return object()

    def capture_distance(*args, **kwargs):
        captured["distance_radii"] = np.asarray(args[4])
        captured["distance_rescale"] = kwargs.get("rescale_deleted_base")
        captured["distance_centers"] = kwargs.get("center_overrides")
        return object()

    monkeypatch.setattr(module, "project_coronary_paths", capture_coronary)
    monkeypatch.setattr(module, "compute_bullseye", capture_territory)
    monkeypatch.setattr(
        module,
        "save_bullseye_figure",
        lambda _result, path, **_kwargs: path,
    )
    monkeypatch.setattr(
        module,
        "compute_distance_volume_nifti",
        lambda *_args, **_kwargs: SimpleNamespace(
            distance_mm=np.ones((2, 2, 2), dtype=float),
            affine=np.eye(4),
            header=object(),
        ),
    )
    monkeypatch.setattr(
        module, "save_distance_nifti", lambda path, *_args, **_kwargs: path
    )
    nifti_data = np.moveaxis(mask, 0, 2)
    fake_nifti = SimpleNamespace(
        shape=nifti_data.shape,
        affine=np.eye(4),
        header=object(),
        dataobj=nifti_data,
    )
    monkeypatch.setattr(module.nib, "load", lambda _path: fake_nifti)
    monkeypatch.setattr(
        module,
        "resample_distance_to_short_axis",
        lambda *_args, **_kwargs: np.ones(nifti_data.shape, dtype=float),
    )
    monkeypatch.setattr(module, "compute_distance_bullseye", capture_distance)
    monkeypatch.setattr(module, "plot_distance_bullseye", lambda *_args, **_kwargs: None)

    module.generate_bullseye(app_config)

    np.testing.assert_allclose(captured["territory_radii"], [-0.5, 1.0, 4.0])
    np.testing.assert_allclose(captured["distance_radii"], [-0.5, 1.0, 4.0])
    assert captured["coronary_frame"].apical_index == 1
    assert captured["coronary_frame"].basal_index == 2
    assert captured["territory_rescale"] is False
    assert captured["distance_rescale"] is False
    assert captured["territory_centers"] == manual_centers
    assert captured["distance_centers"] == manual_centers
