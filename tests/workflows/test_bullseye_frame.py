"""Shared frame preparation must validate and preserve annotation provenance."""

from dataclasses import FrozenInstanceError
from importlib import import_module
import json

import numpy as np
import pytest

from cardiac_bullseye.territory.io import save_landmarks_json
from cardiac_bullseye.territory.models import LandmarkPair, StudyData


def _study(count=4):
    return StudyData(
        np.ones((count, 4, 6)),
        np.ones((count, 4, 6), dtype=np.uint8),
        np.arange(count, dtype=float),
    )


def _prepare(study, path, **kwargs):
    module = import_module("cardiac_bullseye.workflows.bullseye_frame")
    options = dict(
        reuse_saved_points=True,
        landmark_mode="per-slice",
        projection="1-4",
        slice_positions=study.slice_positions,
        transpose_for_display=True,
    )
    options.update(kwargs)
    return module.prepare_bullseye_frame(study, path, **options)


PAIR = LandmarkPair(aip=(4.0, 1.0), iip=(4.0, 3.0))


@pytest.mark.parametrize(
    "projection, radii",
    [("1-4", [1., 2., 4., 7.]), ("0-4", [0., 4./3., 4., 8.])],
)
@pytest.mark.parametrize("mode", ["per-slice", "single-reference"])
def test_saved_frame_returns_deleted_centers_and_exact_physical_radii(
    tmp_path, monkeypatch, projection, radii, mode
):
    module = import_module("cardiac_bullseye.workflows.bullseye_frame")
    study = _study()
    path = tmp_path / "shared.json"
    save_landmarks_json(
        path,
        [PAIR] * 4,
        image_shape=study.mr.shape,
        deleted_slice_indices={3},
        center_overrides=[None, (4.5, 2.5), None, None],
    )
    saved_bytes = path.read_bytes()

    def unexpected_annotation(*_args, **_kwargs):
        pytest.fail("A saved landmark file must be reused without opening a UI")

    monkeypatch.setattr(module, "select_landmarks_interactively", unexpected_annotation)
    monkeypatch.setattr(
        module, "select_single_reference_interactively", unexpected_annotation
    )

    frame = _prepare(
        study, path, projection=projection, landmark_mode=mode,
        slice_positions=[0., 5., 15., 30.],
    )

    assert path.read_bytes() == saved_bytes
    assert frame.landmarks == [PAIR, PAIR, PAIR, None]
    assert frame.deleted_slice_indices == {3}
    assert frame.center_overrides == [None, (4.5, 2.5), None, None]
    np.testing.assert_allclose(frame.radial_frame.source_radii, radii)
    assert frame.radial_frame.basal_index == 2
    with pytest.raises(FrozenInstanceError):
        frame.landmarks = []


@pytest.mark.parametrize(
    "mode, selector",
    [("per-slice", "select_landmarks_interactively"),
     ("single-reference", "select_single_reference_interactively")],
)
@pytest.mark.parametrize("reuse, exists", [(True, False), (False, False), (False, True)])
def test_fresh_annotation_uses_requested_mode_and_overwrites_file(
    tmp_path, monkeypatch, mode, selector, reuse, exists
):
    module = import_module("cardiac_bullseye.workflows.bullseye_frame")
    study = _study()
    path = tmp_path / "shared.json"
    if exists:
        save_landmarks_json(path, [None] * 4, image_shape=study.mr.shape)

    def unexpected_annotation(*_args, **_kwargs):
        pytest.fail("Only the requested annotation mode must open")

    monkeypatch.setattr(module, "select_landmarks_interactively", unexpected_annotation)
    monkeypatch.setattr(
        module, "select_single_reference_interactively", unexpected_annotation
    )

    def select(
        selected_study, selected_path, *, resume_existing,
        transpose_for_display, cancel_is_error,
    ):
        assert selected_study is study
        assert selected_path == path
        assert resume_existing is False
        assert transpose_for_display is False
        assert cancel_is_error is True
        save_landmarks_json(selected_path, [PAIR] * 4, image_shape=study.mr.shape)
        return [PAIR] * 4

    monkeypatch.setattr(module, selector, select)
    frame = _prepare(
        study, str(path), reuse_saved_points=reuse, landmark_mode=mode,
        transpose_for_display=False, cancel_is_error=True,
    )

    assert frame.landmarks == [PAIR] * 4
    assert json.loads(path.read_text())["slices"][0]["aip"] == [4., 1.]


@pytest.mark.parametrize(
    "options, match",
    [({"landmark_mode": "invalid"}, "landmark_mode"),
     ({"projection": "invalid"}, "projection")],
)
def test_invalid_modes_are_rejected_before_annotation(tmp_path, options, match):
    with pytest.raises(ValueError, match=match):
        _prepare(_study(), tmp_path / "missing.json", **options)


@pytest.mark.parametrize(
    "payload_change, match",
    [({"image_shape": [4, 6, 4]}, "image shape"),
     ({"slices": []}, "contains 0 slices"),
     ({"coordinate_version": 99}, "Unsupported landmark coordinate version")],
)
def test_saved_shape_slice_and_version_errors_are_not_deferred(
    tmp_path, payload_change, match
):
    study = _study()
    path = tmp_path / "shared.json"
    save_landmarks_json(path, [PAIR] * 4, image_shape=study.mr.shape)
    payload = json.loads(path.read_text())
    payload.update(payload_change)
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=match):
        _prepare(study, path)


def test_no_valid_landmarks_are_rejected(tmp_path):
    path = tmp_path / "shared.json"
    save_landmarks_json(path, [None] * 4)
    with pytest.raises(RuntimeError, match="No slice has a valid AIP/IIP pair"):
        _prepare(_study(), path)


def test_one_retained_slice_is_rejected_during_preparation(tmp_path):
    study = _study(1)
    path = tmp_path / "shared.json"
    save_landmarks_json(path, [PAIR], image_shape=study.mr.shape)
    with pytest.raises(ValueError, match="at least two retained slices"):
        _prepare(study, path)


@pytest.mark.parametrize("legacy", [False, True])
def test_legacy_default_applies_to_both_landmarks_and_centers(tmp_path, legacy):
    study = _study()
    path = tmp_path / "case_lge_landmarks.json"
    path.write_text(json.dumps({"image_shape": [4, 6, 4], "slices": [
        {"aip": [1., 4.], "iip": [3., 4.], "center": [2.5, 4.5]},
        {"aip": [1., 4.], "iip": [3., 4.]},
        {"aip": [1., 4.], "iip": [3., 4.]}, {"status": "deleted"},
    ]}))
    if not legacy:
        with pytest.raises(ValueError, match="Ambiguous unversioned"):
            _prepare(study, path)
        return

    frame = _prepare(study, path, legacy_lge_default=True)

    assert frame.landmarks == [PAIR, PAIR, PAIR, None]
    assert frame.center_overrides == [(4.5, 2.5), None, None, None]
    assert frame.deleted_slice_indices == {3}
