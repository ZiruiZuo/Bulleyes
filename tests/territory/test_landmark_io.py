import json

import numpy as np
import pytest

import cardiac_bullseye.territory.landmarks as landmark_module

from cardiac_bullseye.territory.io import (
    load_center_overrides_json,
    load_deleted_slice_indices,
    load_landmarks_json,
    save_landmarks_json,
)
from cardiac_bullseye.territory.models import LandmarkPair, StudyData


def test_deleted_slice_is_persisted_separately_from_skipped_slice(tmp_path):
    path = tmp_path / "landmarks.json"
    pair = LandmarkPair(aip=(2.0, 3.0), iip=(4.0, 5.0))

    save_landmarks_json(path, [pair, None, None], deleted_slice_indices={2})

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["slices"][1] is None
    assert payload["slices"][2] == {"status": "deleted"}
    assert load_landmarks_json(path, 3) == [pair, None, None]
    assert load_deleted_slice_indices(path, 3) == {2}


def test_old_landmark_json_has_no_deleted_slices(tmp_path):
    path = tmp_path / "landmarks.json"
    path.write_text(
        json.dumps(
            {
                "slices": [
                    {"aip": [1.0, 2.0], "iip": [3.0, 4.0]},
                    None,
                ]
            }
        ),
        encoding="utf-8",
    )

    assert load_deleted_slice_indices(path, 2) == set()
    assert load_center_overrides_json(path, 2) == [None, None]


def test_manual_centers_are_saved_for_accepted_and_skipped_slices(tmp_path):
    path = tmp_path / "landmarks.json"
    pair = LandmarkPair(aip=(2.0, 3.0), iip=(4.0, 5.0))

    save_landmarks_json(
        path,
        [pair, None, None],
        deleted_slice_indices={2},
        image_shape=(3, 20, 30),
        center_overrides=[(10.0, 11.0), (12.0, 13.0), (14.0, 15.0)],
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["slices"][0] == {
        "aip": [2.0, 3.0],
        "iip": [4.0, 5.0],
        "center": [10.0, 11.0],
    }
    assert payload["slices"][1] == {
        "status": "skipped",
        "center": [12.0, 13.0],
    }
    assert payload["slices"][2] == {"status": "deleted"}
    assert load_landmarks_json(path, 3) == [pair, None, None]
    assert load_center_overrides_json(
        path,
        3,
        expected_image_shape=(3, 20, 30),
    ) == [(10.0, 11.0), (12.0, 13.0), None]


def test_saved_landmarks_load_without_opening_annotation_ui(tmp_path):
    path = tmp_path / "landmarks.json"
    pair = LandmarkPair(aip=(2.0, 3.0), iip=(4.0, 5.0))
    study = StudyData(
        mr=np.zeros((3, 8, 9), dtype=float),
        territory_mask=np.ones((3, 8, 9), dtype=np.uint8),
        slice_positions=np.arange(3.0),
    )
    assert hasattr(landmark_module, "load_saved_landmarks")
    save_landmarks_json(
        path,
        [pair, None, None],
        deleted_slice_indices={2},
        image_shape=study.mr.shape,
    )

    landmarks = landmark_module.load_saved_landmarks(study, path)

    assert landmarks == [pair, None, None]
    assert load_deleted_slice_indices(path, 3) == {2}


def test_saved_landmarks_reject_a_different_image_shape(tmp_path):
    path = tmp_path / "landmarks.json"
    pair = LandmarkPair(aip=(2.0, 3.0), iip=(4.0, 5.0))
    save_landmarks_json(path, [pair], image_shape=(1, 8, 9))
    changed_study = StudyData(
        mr=np.zeros((1, 9, 9), dtype=float),
        territory_mask=np.ones((1, 9, 9), dtype=np.uint8),
        slice_positions=np.asarray([0.0]),
    )

    with pytest.raises(ValueError, match="image shape"):
        landmark_module.load_saved_landmarks(changed_study, path)
