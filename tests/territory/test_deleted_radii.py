import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from cardiac_bullseye.territory.mapping import (
    build_bullseye,
    remove_deleted_slices_from_radial_axis,
)
from cardiac_bullseye.territory.plotting import plot_territory_bullseye
from cardiac_bullseye.coronary.projection import ProjectedCoronaryPath


def _slice_probabilities() -> np.ndarray:
    probabilities = np.zeros((5, 12, 3), dtype=float)
    probabilities[0:2, :, 0] = 1.0
    probabilities[2, :, 1] = 1.0
    probabilities[3:5, :, 2] = 1.0
    return probabilities


def test_deleted_basal_slices_are_removed_and_last_retained_slice_reaches_outer_radius(
    tmp_path,
):
    skipped = _slice_probabilities()
    skipped[2] = np.nan
    interpolated = build_bullseye(
        skipped,
        np.arange(5.0),
        source_radii=np.arange(5.0),
        radial_count=13,
        radial_smoothing_sigma=0.0,
    )
    assert np.isfinite(interpolated.probabilities).all()
    assert np.all(interpolated.territory_labels[-1] == 3)

    deleted = build_bullseye(
        _slice_probabilities(),
        np.arange(5.0),
        source_radii=np.arange(5.0),
        radial_count=13,
        radial_smoothing_sigma=0.0,
        deleted_slice_indices={3, 4},
    )
    assert np.isfinite(deleted.probabilities).all()
    radius_two = np.flatnonzero(np.isclose(deleted.radial_positions, 2.0))[0]
    assert np.all(deleted.territory_labels[radius_two] == 1)
    assert np.all(deleted.territory_labels[-1] == 2)

    promotion = tmp_path / "deleted.png"
    figure, axis = plot_territory_bullseye(
        deleted,
        promotion,
        coronary_paths=[
            ProjectedCoronaryPath(
                label=1,
                xy=np.asarray([[0.0, 2.0], [0.0, 4.0]]),
            )
        ],
        show=False,
    )
    assert promotion.is_file()
    assert not np.ma.getmaskarray(axis.collections[0].get_array()).any()
    coronary_lines = [line for line in axis.lines if line.get_zorder() == 20]
    assert len(coronary_lines) == 1
    np.testing.assert_allclose(coronary_lines[0].get_ydata(), [2.0, 4.0])


def test_delete_rejects_a_gap_inside_the_retained_slice_stack():
    with pytest.raises(ValueError, match="contiguous basal tail"):
        build_bullseye(
            _slice_probabilities(),
            np.arange(5.0),
            source_radii=np.arange(5.0),
            radial_smoothing_sigma=0.0,
            deleted_slice_indices={2},
        )


def test_segment_17_uses_aggregate_dense_aha_13_to_16_probabilities():
    probabilities = np.zeros((4, 12, 3), dtype=float)
    probabilities[0, :, 0] = 1.0
    probabilities[1:, :, 1] = 1.0

    result = build_bullseye(
        probabilities,
        np.arange(4.0),
        source_radii=[1.0, 1.1, 2.0, 4.0],
        radial_count=301,
        radial_smoothing_sigma=0.0,
        rescale_deleted_base=False,
    )

    assert result.apex_territory == 2


def test_preanchored_radii_are_not_rescaled_after_basal_deletion():
    values = np.arange(5.0)[:, None]
    retained, radii = remove_deleted_slices_from_radial_axis(
        values,
        [0.0, 1.0, 2.5, 4.0, 5.0],
        {4},
        rescale_retained_base=False,
    )

    np.testing.assert_array_equal(retained[:, 0], [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(radii, [0.0, 1.0, 2.5, 4.0])
