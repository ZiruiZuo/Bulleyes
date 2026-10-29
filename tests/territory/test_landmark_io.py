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
    path = tmp_path / "case_landmarks.json"
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


def test_single_reference_metadata_is_saved(tmp_path):
    path = tmp_path / "landmarks.json"
    pair = LandmarkPair(aip=(2.0, 3.0), iip=(4.0, 5.0))

    save_landmarks_json(
        path,
        [pair, pair, pair],
        annotation_mode="single-reference",
        reference_slice_index=1,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["annotation_mode"] == "single-reference"
    assert payload["reference_slice_index"] == 1
    assert payload["reference_slice_number"] == 2


def test_versioned_asymmetric_landmarks_and_centers_round_trip(tmp_path):
    from cardiac_bullseye.territory import io

    path = tmp_path / "explicit_shared.json"
    pair = LandmarkPair(aip=(2.5, 8.0), iip=(4.0, 10.0))
    save_landmarks_json(
        path, [pair], image_shape=(1, 12, 6), center_overrides=[(3.0, 9.0)]
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["coordinate_version"] == io.LANDMARK_COORDINATE_VERSION == 1
    assert load_landmarks_json(path, 1, expected_image_shape=(1, 12, 6)) == [pair]
    assert load_center_overrides_json(
        path, 1, expected_image_shape=(1, 12, 6)
    ) == [(3.0, 9.0)]


def test_recognized_unversioned_common_file_keeps_asymmetric_coordinates(tmp_path):
    path = tmp_path / "case_landmarks.json"
    path.write_text(json.dumps({
        "image_shape": [1, 12, 6],
        "slices": [{"aip": [2.5, 8.0], "iip": [4.0, 10.0], "center": [3.0, 9.0]}],
    }), encoding="utf-8")

    assert load_landmarks_json(path, 1, expected_image_shape=(1, 12, 6)) == [
        LandmarkPair(aip=(2.5, 8.0), iip=(4.0, 10.0))
    ]
    assert load_center_overrides_json(
        path, 1, expected_image_shape=(1, 12, 6)
    ) == [(3.0, 9.0)]


def test_declared_default_legacy_lge_swaps_points_centers_and_shape(tmp_path):
    path = tmp_path / "case_lge_landmarks.json"
    payload = {
        "image_shape": [3, 6, 12],
        "slices": [
            {"aip": [8.0, 2.5], "iip": [10.0, 4.0], "center": [9.0, 3.0]},
            {"status": "skipped", "center": [7.0, 2.0]},
            {"status": "deleted"},
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    study = StudyData(np.zeros((3, 12, 6)), np.ones((3, 12, 6), dtype=np.uint8), np.arange(3.0))

    assert landmark_module.load_saved_landmarks(study, path, legacy_default_lge=True) == [
        LandmarkPair(aip=(2.5, 8.0), iip=(4.0, 10.0)), None, None
    ]
    assert load_center_overrides_json(
        path, 3, expected_image_shape=study.mr.shape, legacy_default_lge=True
    ) == [(3.0, 9.0), (2.0, 7.0), None]
    assert load_deleted_slice_indices(path, 3) == {2}
    assert json.loads(path.read_text(encoding="utf-8")) == payload


@pytest.mark.parametrize("basename", ["shared.json", "case_lge_landmarks.json"])
@pytest.mark.parametrize("loader", [load_landmarks_json, load_center_overrides_json])
def test_explicit_unversioned_paths_are_ambiguous(tmp_path, basename, loader):
    path = tmp_path / basename
    path.write_text(json.dumps({"slices": [{"aip": [1, 2], "iip": [3, 4]}]}), encoding="utf-8")

    with pytest.raises(ValueError, match="[Aa]mbiguous.*unversioned"):
        loader(path, 1)


@pytest.mark.parametrize("legacy_default_lge", [False, True])
def test_legacy_default_declaration_cannot_migrate_another_basename(tmp_path, legacy_default_lge):
    path = tmp_path / "shared.json"
    path.write_text(json.dumps({"slices": [None]}), encoding="utf-8")

    with pytest.raises(ValueError, match="[Aa]mbiguous.*unversioned"):
        load_landmarks_json(path, 1, legacy_default_lge=legacy_default_lge)


def test_legacy_migration_still_checks_shape_and_slice_count(tmp_path):
    path = tmp_path / "case_lge_landmarks.json"
    path.write_text(json.dumps({"image_shape": [1, 6, 12], "slices": [None]}), encoding="utf-8")

    with pytest.raises(ValueError, match="image shape"):
        load_landmarks_json(path, 1, expected_image_shape=(1, 13, 6), legacy_default_lge=True)
    with pytest.raises(ValueError, match="contains 1 slices"):
        load_landmarks_json(path, 2, legacy_default_lge=True)


@pytest.mark.parametrize("loader", [load_landmarks_json, load_center_overrides_json])
def test_unknown_coordinate_version_is_rejected(tmp_path, loader):
    path = tmp_path / "case_landmarks.json"
    path.write_text(json.dumps({"coordinate_version": 99, "slices": [None]}), encoding="utf-8")

    with pytest.raises(ValueError, match="coordinate version"):
        loader(path, 1)
