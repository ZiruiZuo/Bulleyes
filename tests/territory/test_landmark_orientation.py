import numpy as np

from cardiac_bullseye.territory.landmarks import (
    annotation_display_transform,
    display_xy_to_original_xy,
    original_xy_to_display_xy,
)


def test_annotation_uses_native_slice_orientation():
    """Reintroducing the legacy transpose must fail this regression test."""
    image = np.asarray([[1, 2, 3], [4, 5, 6]])

    displayed = annotation_display_transform(image)

    np.testing.assert_array_equal(displayed, [[1, 2, 3], [4, 5, 6]])


def test_annotation_points_stay_in_original_pixel_coordinates():
    """A display-only transpose must not silently swap saved AIP/IIP axes."""
    original = (17.25, 43.5)

    displayed = original_xy_to_display_xy(original)
    recovered = display_xy_to_original_xy(displayed)

    assert displayed == original
    assert recovered == original
