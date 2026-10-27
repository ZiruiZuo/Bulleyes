import numpy as np

from cardiac_bullseye.territory.landmarks import propagate_reference_landmarks
from cardiac_bullseye.territory.models import LandmarkPair, StudyData


def _ring(shape, center_xy, inner_radius=3.0, outer_radius=6.0):
    y, x = np.indices(shape)
    radius = np.hypot(x - center_xy[0], y - center_xy[1])
    return (radius >= inner_radius) & (radius <= outer_radius)


def test_single_reference_directions_are_translated_to_each_slice_center():
    centers = [(12.0, 14.0), (16.0, 15.0), (19.0, 18.0)]
    masks = np.stack([_ring((40, 40), center) for center in centers])
    study = StudyData(
        mr=np.zeros_like(masks, dtype=float),
        territory_mask=masks.astype(np.uint8),
        slice_positions=np.asarray([0.0, 5.0, 10.0]),
    )
    reference = LandmarkPair(aip=(16.0, 14.0), iip=(12.0, 19.0))

    propagated = propagate_reference_landmarks(
        study,
        reference_slice_index=0,
        reference_pair=reference,
    )

    assert propagated[0] == reference
    for index, center in enumerate(centers):
        assert propagated[index] is not None
        np.testing.assert_allclose(
            np.subtract(propagated[index].aip, center),
            (4.0, 0.0),
        )
        np.testing.assert_allclose(
            np.subtract(propagated[index].iip, center),
            (0.0, 5.0),
        )


def test_single_reference_omits_deleted_and_empty_slices():
    masks = np.stack(
        [
            _ring((40, 40), (15.0, 15.0)),
            np.zeros((40, 40), dtype=bool),
            _ring((40, 40), (18.0, 17.0)),
        ]
    )
    study = StudyData(
        mr=np.zeros_like(masks, dtype=float),
        territory_mask=masks.astype(np.uint8),
        slice_positions=np.asarray([0.0, 5.0, 10.0]),
    )

    propagated = propagate_reference_landmarks(
        study,
        reference_slice_index=0,
        reference_pair=LandmarkPair(aip=(19.0, 15.0), iip=(15.0, 20.0)),
        deleted_slice_indices={2},
    )

    assert propagated[0] is not None
    assert propagated[1] is None
    assert propagated[2] is None
