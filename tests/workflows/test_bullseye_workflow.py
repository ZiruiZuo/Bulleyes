import numpy as np
import pytest
from dataclasses import replace
from types import SimpleNamespace

from cardiac_bullseye.territory.models import LandmarkPair, StudyData
from cardiac_bullseye.workflows import bullseye_frame as frame_module


def test_bullseye_workflow_requests_a_fresh_transposed_nifti_annotation_view(
    monkeypatch, app_config
):
    from cardiac_bullseye.workflows import bullseye as module

    study = StudyData(
        mr=np.ones((2, 4, 6), dtype=float),
        territory_mask=np.ones((2, 4, 6), dtype=np.uint8),
        slice_positions=np.asarray([0.0, 6.0]),
    )

    geometry = SimpleNamespace(slice_positions_mm=np.asarray([0.0, 6.0]))
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
        cancel_is_error,
    ):
        assert selected_study is study
        assert resume_existing is False
        assert transpose_for_display is True
        assert cancel_is_error is False
        return [None, None]

    monkeypatch.setattr(
        frame_module, "select_landmarks_interactively", select_transposed_view
    )

    with pytest.raises(RuntimeError, match="No slice has a valid AIP/IIP pair"):
        module.generate_bullseye(app_config)


def test_bullseye_workflow_uses_single_reference_annotation_mode(
    monkeypatch, app_config
):
    from cardiac_bullseye.workflows import bullseye as module

    study = StudyData(
        mr=np.ones((2, 4, 6), dtype=float),
        territory_mask=np.ones((2, 4, 6), dtype=np.uint8),
        slice_positions=np.asarray([0.0, 6.0]),
    )
    geometry = SimpleNamespace(slice_positions_mm=np.asarray([0.0, 6.0]))
    calls = []
    monkeypatch.setattr(module, "load_cardiac_geometry", lambda _path: geometry)
    monkeypatch.setattr(
        module, "validate_geometry_against_short_axis_nifti", lambda *_args: None
    )
    monkeypatch.setattr(module, "load_study_nifti", lambda *_args, **_kwargs: study)
    monkeypatch.setattr(
        frame_module,
        "select_landmarks_interactively",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("per-slice annotation UI must not open")
        ),
    )

    def select_single_reference(
        selected_study,
        _output_path,
        *,
        resume_existing,
        transpose_for_display,
        cancel_is_error,
    ):
        assert cancel_is_error is False
        calls.append(
            (selected_study, resume_existing, transpose_for_display)
        )
        return [None, None]

    monkeypatch.setattr(
        frame_module,
        "select_single_reference_interactively",
        select_single_reference,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="No slice has a valid AIP/IIP pair"):
        module.generate_bullseye(app_config, landmark_mode="single-reference")

    assert calls == [(study, False, True)]


def test_bullseye_workflow_reuses_landmarks_without_opening_annotation_ui(
    monkeypatch, app_config
):
    from cardiac_bullseye.workflows import bullseye as module

    config = replace(
        app_config,
        bullseye=replace(app_config.bullseye, reuse_saved_points=True),
    )
    study = StudyData(
        mr=np.ones((2, 4, 6), dtype=float),
        territory_mask=np.ones((2, 4, 6), dtype=np.uint8),
        slice_positions=np.asarray([0.0, 6.0]),
    )
    geometry = SimpleNamespace(slice_positions_mm=np.asarray([0.0, 6.0]))
    monkeypatch.setattr(module, "load_cardiac_geometry", lambda _path: geometry)
    monkeypatch.setattr(
        module, "validate_geometry_against_short_axis_nifti", lambda *_args: None
    )
    monkeypatch.setattr(module, "load_study_nifti", lambda *_args, **_kwargs: study)
    monkeypatch.setattr(
        frame_module,
        "select_landmarks_interactively",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("annotation UI must not open")
        ),
    )
    config.case.output_path("landmarks.json").touch()
    monkeypatch.setattr(
        frame_module, "load_saved_landmarks", lambda *_args, **_kwargs: [None, None]
    )
    monkeypatch.setattr(
        frame_module, "load_deleted_slice_indices", lambda *_args: set()
    )
    monkeypatch.setattr(
        frame_module, "load_center_overrides_json", lambda *_args, **_kwargs: [None, None]
    )

    with pytest.raises(RuntimeError, match="No slice has a valid AIP/IIP pair"):
        module.generate_bullseye(config)


@pytest.mark.parametrize(
    (
        "projection",
        "expected_radii",
        "minimum_radius",
        "expected_apex",
        "expected_center_blend_radius",
    ),
    [
        ("1-4", [-0.5, 1.0, 4.0], 1.0, 3, None),
        ("0-4", [-2.0, 0.0, 4.0], 0.0, None, 4.0),
    ],
)
@pytest.mark.parametrize("explicit_path", [False, True])
def test_bullseye_workflow_shares_mask_anchored_frame(
    monkeypatch,
    app_config,
    projection,
    expected_radii,
    minimum_radius,
    expected_apex,
    expected_center_blend_radius,
    explicit_path,
):
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
    app_config = replace(
        app_config,
        bullseye=replace(app_config.bullseye, apex_territory=3),
    )
    expected_path = app_config.case.output_path("landmarks.json")
    if explicit_path:
        expected_path = app_config.case.output_directory / "shared.json"
        app_config = replace(
            app_config,
            bullseye=replace(app_config.bullseye, landmarks_path=expected_path),
        )

    monkeypatch.setattr(module, "validate_config", lambda *_args: None)
    monkeypatch.setattr(module, "load_cardiac_geometry", lambda _path: geometry)
    monkeypatch.setattr(
        module, "validate_geometry_against_short_axis_nifti", lambda *_args: None
    )
    monkeypatch.setattr(module, "load_study_nifti", lambda *_args, **_kwargs: study)

    def select_landmarks(_study, path, **_kwargs):
        assert path == expected_path
        return landmarks

    monkeypatch.setattr(frame_module, "select_landmarks_interactively", select_landmarks)
    monkeypatch.setattr(
        frame_module, "load_deleted_slice_indices", lambda *_args, **_kwargs: set()
    )
    monkeypatch.setattr(
        frame_module,
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
        captured["apex_territory"] = kwargs.get("apex_territory")
        captured["center_blend_radius"] = kwargs.get("center_blend_radius")
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

    module.generate_bullseye(app_config, projection=projection)

    np.testing.assert_allclose(captured["territory_radii"], expected_radii)
    np.testing.assert_allclose(captured["distance_radii"], expected_radii)
    assert captured["coronary_frame"].apical_index == 1
    assert captured["coronary_frame"].basal_index == 2
    assert captured["coronary_frame"].minimum_radius == minimum_radius
    assert captured["apex_territory"] == expected_apex
    assert captured["center_blend_radius"] == expected_center_blend_radius
    assert captured["territory_rescale"] is False
    assert captured["distance_rescale"] is False
    assert captured["territory_centers"] == manual_centers
    assert captured["distance_centers"] == manual_centers
