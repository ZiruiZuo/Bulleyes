import numpy as np
import pytest

import cardiac_bullseye.territory.geometry as geometry
import cardiac_bullseye.territory.landmarks as landmark_ui
from cardiac_bullseye.lge.mapping import compute_slice_lge_profiles
from cardiac_bullseye.lge.scar import compute_aha16_segment_ids
from cardiac_bullseye.distance.mapping import compute_slice_distance_profiles
from cardiac_bullseye.territory.mapping import compute_slice_territory_probabilities
from cardiac_bullseye.territory.models import LandmarkPair, StudyData


def _ring(shape, center_xy, *, open_left=False):
    y, x = np.ogrid[: shape[0], : shape[1]]
    cx, cy = center_xy
    radius = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    mask = (radius >= 6.0) & (radius <= 10.0)
    if open_left:
        mask[:, :cx] = False
    return mask


def test_open_slices_copy_the_nearest_closed_cavity_center_with_basal_tie_break():
    assert hasattr(geometry, "estimate_lv_centers")
    masks = np.stack(
        [
            _ring((81, 81), (20, 40)),
            _ring((81, 81), (30, 40), open_left=True),
            _ring((81, 81), (40, 40), open_left=True),
            _ring((81, 81), (50, 40), open_left=True),
            _ring((81, 81), (60, 40)),
        ]
    )

    centers = geometry.estimate_lv_centers(masks)

    np.testing.assert_allclose(centers[0], [20.0, 40.0])
    np.testing.assert_allclose(centers[1], [20.0, 40.0])
    np.testing.assert_allclose(centers[2], [60.0, 40.0])
    np.testing.assert_allclose(centers[3], [60.0, 40.0])
    np.testing.assert_allclose(centers[4], [60.0, 40.0])


def test_center_sequence_rejects_a_stack_without_any_closed_cavity():
    assert hasattr(geometry, "estimate_lv_centers")
    masks = np.stack(
        [
            _ring((61, 61), (25, 30), open_left=True),
            _ring((61, 61), (35, 30), open_left=True),
        ]
    )

    with pytest.raises(ValueError, match="closed LV cavity"):
        geometry.estimate_lv_centers(masks)


def test_manual_center_override_replaces_only_the_selected_slice_center():
    masks = np.stack(
        [
            _ring((81, 81), (20, 40)),
            _ring((81, 81), (40, 40), open_left=True),
            _ring((81, 81), (60, 40)),
        ]
    )

    centers = geometry.estimate_lv_centers(
        masks,
        center_overrides=[None, (33.0, 44.0), None],
    )

    np.testing.assert_allclose(
        centers,
        [[20.0, 40.0], [33.0, 44.0], [60.0, 40.0]],
    )


def _all_open_study() -> tuple[StudyData, list[LandmarkPair]]:
    masks = np.stack(
        [
            _ring((61, 61), (25, 30), open_left=True),
            _ring((61, 61), (35, 30), open_left=True),
        ]
    )
    study = StudyData(
        mr=np.ones(masks.shape, dtype=float),
        territory_mask=masks.astype(np.uint8),
        slice_positions=np.asarray([0.0, 6.0]),
    )
    landmarks = [
        LandmarkPair(aip=(30.0, 20.0), iip=(30.0, 40.0)),
        LandmarkPair(aip=(40.0, 20.0), iip=(40.0, 40.0)),
    ]
    return study, landmarks


def _labelled_ring_study() -> tuple[StudyData, list[LandmarkPair], np.ndarray]:
    size = 41
    center = 20
    ring = _ring((size, size), (center, center))
    y, x = np.ogrid[:size, :size]
    labels = np.zeros((size, size), dtype=np.uint8)
    labels[ring & (x >= center)] = 1
    labels[ring & (x < center) & (y < center)] = 2
    labels[ring & (x < center) & (y >= center)] = 3
    mr = np.broadcast_to(np.arange(size, dtype=float), (size, size)).copy()
    study = StudyData(
        mr=mr[None, ...],
        territory_mask=labels[None, ...],
        slice_positions=np.asarray([0.0]),
    )
    landmarks = [LandmarkPair(aip=(20.0, 10.0), iip=(20.0, 30.0))]
    distance = (x + 2.0 * y).astype(float)
    return study, landmarks, distance[None, ...]


@pytest.mark.parametrize(
    "mapper",
    [
        lambda study, landmarks, distance, **kwargs: (
            compute_slice_territory_probabilities(study, landmarks, **kwargs)
        ),
        lambda study, landmarks, distance, **kwargs: (
            compute_slice_lge_profiles(study, landmarks, **kwargs)
        ),
        lambda study, landmarks, distance, **kwargs: (
            compute_aha16_segment_ids(study, landmarks, **kwargs)
        ),
        lambda study, landmarks, distance, **kwargs: compute_slice_distance_profiles(
            study, distance, landmarks, **kwargs
        ),
    ],
)
def test_bullseye_mappers_use_manual_center_overrides(mapper):
    study, landmarks, distance = _labelled_ring_study()
    options = {"six_segment_warp": False, "center_overrides": [(26.0, 20.0)]}

    automatic = mapper(
        study,
        landmarks,
        distance,
        six_segment_warp=False,
    )
    adjusted = mapper(study, landmarks, distance, **options)

    assert not np.allclose(automatic, adjusted, equal_nan=True)


def test_lge_scar_mapping_rejects_a_center_count_mismatch():
    study, landmarks, _distance = _labelled_ring_study()

    with pytest.raises(ValueError, match="one entry per slice"):
        compute_aha16_segment_ids(
            study,
            landmarks,
            center_overrides=[None, None],
        )


def test_skipped_lge_slice_uses_its_retained_manual_center():
    single_slice, single_landmark, _distance = _labelled_ring_study()
    study = StudyData(
        mr=np.repeat(single_slice.mr, 3, axis=0),
        territory_mask=np.repeat(single_slice.territory_mask, 3, axis=0),
        slice_positions=np.asarray([0.0, 6.0, 12.0]),
    )
    landmarks = [single_landmark[0], None, single_landmark[0]]

    automatic = compute_aha16_segment_ids(
        study,
        landmarks,
        six_segment_warp=False,
    )
    adjusted = compute_aha16_segment_ids(
        study,
        landmarks,
        six_segment_warp=False,
        center_overrides=[None, (26.0, 20.0), None],
    )

    np.testing.assert_array_equal(automatic[0], adjusted[0])
    assert not np.array_equal(automatic[1], adjusted[1])
    np.testing.assert_array_equal(automatic[2], adjusted[2])


@pytest.mark.parametrize(
    "mapper",
    [
        compute_slice_territory_probabilities,
        compute_slice_lge_profiles,
        compute_aha16_segment_ids,
    ],
)
def test_bullseye_mappers_reject_a_stack_without_any_closed_cavity(mapper):
    study, landmarks = _all_open_study()

    with pytest.raises(ValueError, match="closed LV cavity"):
        mapper(study, landmarks, six_segment_warp=False)


def test_landmark_ui_validates_closed_cavity_before_opening_a_window(
    tmp_path, monkeypatch
):
    study, _landmarks = _all_open_study()

    def fail_if_ui_opens():
        raise RuntimeError("UI opened")

    monkeypatch.setattr(landmark_ui, "_ensure_interactive_backend", fail_if_ui_opens)

    with pytest.raises(ValueError, match="closed LV cavity"):
        landmark_ui.select_landmarks_interactively(
            study,
            tmp_path / "landmarks.json",
            resume_existing=False,
        )
