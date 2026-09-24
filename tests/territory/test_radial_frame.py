import numpy as np
import pytest

from cardiac_bullseye.territory.radial import (
    build_mask_anchored_radial_frame,
)


def test_mask_anchors_map_first_occupied_and_last_retained_to_one_and_four():
    mask = np.zeros((6, 3, 3), dtype=np.uint8)
    mask[1:5, 1, 1] = 1
    positions = np.asarray([0.0, 2.0, 5.0, 9.0, 14.0, 20.0])

    frame = build_mask_anchored_radial_frame(
        mask,
        positions,
        deleted_slice_indices={5},
    )

    assert frame.apical_index == 1
    assert frame.basal_index == 4
    assert frame.apical_position == 2.0
    assert frame.basal_position == 14.0
    assert frame.source_radii[1] == pytest.approx(1.0)
    assert frame.source_radii[4] == pytest.approx(4.0)
    np.testing.assert_allclose(
        frame.map_positions([2.0, 8.0, 14.0]),
        [1.0, 2.5, 4.0],
    )


def test_mask_anchor_rejects_empty_mask():
    with pytest.raises(ValueError, match="mask is empty"):
        build_mask_anchored_radial_frame(
            np.zeros((3, 2, 2), dtype=np.uint8),
            [0.0, 1.0, 2.0],
        )


def test_mask_anchor_rejects_empty_final_retained_slice():
    mask = np.zeros((4, 2, 2), dtype=np.uint8)
    mask[0:2, 0, 0] = 1
    with pytest.raises(ValueError, match="final retained basal slice has no mask"):
        build_mask_anchored_radial_frame(mask, [0.0, 1.0, 2.0, 3.0])


def test_mask_anchor_requires_distinct_apical_and_basal_slices():
    mask = np.zeros((3, 2, 2), dtype=np.uint8)
    mask[1, 0, 0] = 1
    with pytest.raises(ValueError, match="at least two"):
        build_mask_anchored_radial_frame(
            mask,
            [0.0, 1.0, 2.0],
            deleted_slice_indices={2},
        )
