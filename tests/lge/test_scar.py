from __future__ import annotations

import errno
import csv
import matplotlib
import numpy as np
import os
import pytest
import SimpleITK as sitk
import time
from matplotlib.backend_bases import FigureCanvasBase, MouseButton, MouseEvent
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from pathlib import Path

matplotlib.use("Agg")

from cardiac_bullseye.lge import (
    apply_brush_stroke,
    compute_aha16_segment_ids,
    compute_aha16_scar_percentages,
    compute_lge_scar,
    load_reference_roi_polygons,
    load_reference_roi_masks,
    plot_lge_scar_bullseye,
    rasterize_reference_roi_polygons,
    save_reference_roi_polygons,
    save_reference_roi_masks,
    save_aha16_scar_csv,
    save_lge_scar_mask,
    select_reference_rois_interactively,
    zoom_limits_about_point,
)
from cardiac_bullseye.territory.models import LandmarkPair, StudyData
from cardiac_bullseye import lge as lge_api


def test_scar_threshold_is_computed_independently_from_each_manual_roi():
    myocardium = np.ones((2, 4, 4), dtype=np.uint8)
    lge = np.full(myocardium.shape, 10.0)
    roi_masks = np.zeros_like(myocardium, dtype=bool)
    roi_coordinates = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    first_values = [8.0, 9.0, 10.0, 10.0, 11.0, 12.0]
    second_values = [18.0, 19.0, 20.0, 20.0, 21.0, 22.0]
    for slice_index, values in enumerate((first_values, second_values)):
        for (row, column), value in zip(roi_coordinates, values, strict=True):
            roi_masks[slice_index, row, column] = True
            lge[slice_index, row, column] = value
    lge[0, 3, 3] = 18.0
    lge[1, 3, 3] = 28.0
    lge[:, 3, 2] = [17.0, 27.0]

    study = StudyData(lge, myocardium, np.asarray([0.0, 6.0]))
    landmark = LandmarkPair(aip=(1.5, 0.0), iip=(1.5, 3.0))

    result = compute_lge_scar(
        study,
        [landmark, landmark],
        roi_masks,
        sd_multiplier=5.0,
        min_roi_voxels=6,
    )

    expected_std = np.sqrt(2.0)
    np.testing.assert_allclose(result.reference_means, [10.0, 20.0])
    np.testing.assert_allclose(result.reference_stds, [expected_std, expected_std])
    np.testing.assert_allclose(
        result.thresholds,
        [10.0 + 5.0 * expected_std, 20.0 + 5.0 * expected_std],
    )
    assert result.scar_mask[0, 3, 3]
    assert result.scar_mask[1, 3, 3]
    assert not result.scar_mask[0, 3, 2]
    assert not result.scar_mask[1, 3, 2]
    assert not np.any(result.scar_mask[:, :2, :3] & roi_masks[:, :2, :3])


def test_native_voxels_are_assigned_to_aha16_without_segment_17():
    size = 25
    y, x = np.ogrid[:size, :size]
    ring = ((x - 12) ** 2 + (y - 12) ** 2 >= 25) & ((x - 12) ** 2 + (y - 12) ** 2 <= 81)
    myocardium = np.repeat(ring[None, :, :], 3, axis=0).astype(np.uint8)
    study = StudyData(
        np.zeros_like(myocardium, dtype=float),
        myocardium,
        np.asarray([0.0, 6.0, 12.0]),
    )
    landmark = LandmarkPair(aip=(12.0, 5.0), iip=(12.0, 19.0))

    segment_ids = compute_aha16_segment_ids(
        study, [landmark] * 3, six_segment_warp=False
    )

    assert set(np.unique(segment_ids[0][ring])) == {13, 14, 15, 16}
    assert set(np.unique(segment_ids[1][ring])) == {7, 8, 9, 10, 11, 12}
    assert set(np.unique(segment_ids[2][ring])) == {1, 2, 3, 4, 5, 6}
    assert 17 not in segment_ids
    assert np.all(segment_ids[~myocardium.astype(bool)] == 0)


def test_segment_scar_percentages_use_exact_native_voxel_counts():
    segment_ids = np.asarray([[[1, 1, 2, 2], [0, 0, 0, 0]]], dtype=np.uint8)
    scar_mask = np.asarray(
        [[[True, False, True, True], [False, False, False, False]]],
        dtype=bool,
    )

    summary = compute_aha16_scar_percentages(scar_mask, segment_ids)

    np.testing.assert_array_equal(summary.myocardial_voxel_counts[:3], [2, 2, 0])
    np.testing.assert_array_equal(summary.scar_voxel_counts[:3], [1, 2, 0])
    np.testing.assert_allclose(summary.scar_percentages[:2], [50.0, 100.0])
    assert np.isnan(summary.scar_percentages[2:]).all()


def test_reference_roi_polygons_round_trip_and_rasterize_in_xy_order(tmp_path):
    polygons = [
        [(0.5, 0.5), (3.5, 0.5), (3.5, 3.5), (0.5, 3.5)],
        None,
    ]
    output = tmp_path / "reference_rois.json"

    save_reference_roi_polygons(output, polygons)
    restored = load_reference_roi_polygons(output, expected_slice_count=2)
    masks = rasterize_reference_roi_polygons((2, 5, 5), restored)

    assert restored == polygons
    assert masks[0, 1:4, 1:4].all()
    assert np.count_nonzero(masks[0]) == 9
    assert not masks[1].any()


def test_brush_stroke_paints_a_continuous_path_and_erases_pixels():
    mask = np.zeros((15, 15), dtype=bool)

    apply_brush_stroke(mask, (2.0, 7.0), (12.0, 7.0), radius=1.0, value=True)

    assert mask[7, 2:13].all()
    assert not mask[0, 0]

    apply_brush_stroke(mask, (7.0, 7.0), (7.0, 7.0), radius=1.0, value=False)

    assert not mask[7, 7]
    assert mask[7, 2]
    assert mask[7, 12]


def test_brush_roi_masks_round_trip_with_disconnected_regions_and_a_hole(tmp_path):
    first = np.zeros((6, 7), dtype=bool)
    first[1:5, 1:6] = True
    first[2:4, 3] = False
    first[0, 6] = True
    output = tmp_path / "reference_rois.json"

    save_reference_roi_masks(output, [first, None])
    restored = load_reference_roi_masks(output, expected_shape=(2, 6, 7))

    np.testing.assert_array_equal(restored[0], first)
    assert restored[1] is None


def test_brush_roi_loader_converts_an_old_polygon_file(tmp_path):
    output = tmp_path / "reference_rois.json"
    save_reference_roi_polygons(
        output,
        [[(0.5, 0.5), (3.5, 0.5), (3.5, 3.5), (0.5, 3.5)]],
    )

    restored = load_reference_roi_masks(output, expected_shape=(1, 5, 5))

    assert restored[0] is not None
    assert restored[0][1:4, 1:4].all()
    assert np.count_nonzero(restored[0]) == 9


def test_zoom_keeps_the_cursor_on_the_same_image_coordinate():
    xlim = (-0.5, 99.5)
    ylim = (79.5, -0.5)
    cursor = (74.5, 19.5)

    zoomed_x, zoomed_y = zoom_limits_about_point(
        xlim, ylim, cursor, scale=0.5, image_shape=(80, 100)
    )

    assert zoomed_x == pytest.approx((37.0, 87.0))
    assert zoomed_y == pytest.approx((49.5, 9.5))


def test_roi_gui_left_drag_paints_a_brush_mask(monkeypatch, tmp_path):
    from cardiac_bullseye.lge import reference_roi as module

    callbacks = {}

    class FakeButton:
        def __init__(self, _axis, label):
            self.label = label

        def on_clicked(self, callback):
            callbacks[self.label] = callback

    def paint_and_accept(canvas, timeout=0):
        axis = canvas.figure.axes[0]

        def dispatch(name, point, button):
            display_x, display_y = axis.transData.transform(point)
            event = MouseEvent(name, canvas, display_x, display_y, button=button)
            canvas.callbacks.process(name, event)

        dispatch("button_press_event", (3.0, 7.0), MouseButton.LEFT)
        dispatch("motion_notify_event", (11.0, 7.0), MouseButton.LEFT)
        dispatch("button_release_event", (11.0, 7.0), MouseButton.LEFT)
        callbacks["Accept"](None)

    monkeypatch.setattr(module, "Button", FakeButton)
    monkeypatch.setattr(module, "_ensure_interactive_backend", lambda: None)
    monkeypatch.setattr(module.plt, "show", lambda **_kwargs: None)
    monkeypatch.setattr(FigureCanvasBase, "start_event_loop", paint_and_accept)
    study = StudyData(
        np.ones((1, 15, 15), dtype=float),
        np.ones((1, 15, 15), dtype=np.uint8),
        np.asarray([0.0]),
    )
    landmark = LandmarkPair(aip=(7.0, 2.0), iip=(7.0, 12.0))
    output = tmp_path / "rois.json"

    selected = select_reference_rois_interactively(
        study, [landmark], output, min_roi_voxels=2
    )

    assert selected[0] is not None
    assert selected[0][3:12, 7].all()
    np.testing.assert_array_equal(
        load_reference_roi_masks(output, study.mr.shape)[0], selected[0]
    )


def test_roi_gui_fresh_mode_ignores_a_saved_mask(monkeypatch, tmp_path):
    from cardiac_bullseye.lge import reference_roi as module

    callbacks = {}
    event_loop_called = False

    class FakeButton:
        def __init__(self, _axis, label):
            self.label = label

        def on_clicked(self, callback):
            callbacks[self.label] = callback

    def paint_fresh_mask_and_accept(canvas, timeout=0):
        nonlocal event_loop_called
        event_loop_called = True
        axis = canvas.figure.axes[0]

        def dispatch(name, point, button):
            display_x, display_y = axis.transData.transform(point)
            event = MouseEvent(name, canvas, display_x, display_y, button=button)
            canvas.callbacks.process(name, event)

        dispatch("button_press_event", (11.0, 11.0), MouseButton.LEFT)
        dispatch("button_release_event", (11.0, 11.0), MouseButton.LEFT)
        callbacks["Accept"](None)

    monkeypatch.setattr(module, "Button", FakeButton)
    monkeypatch.setattr(module, "_ensure_interactive_backend", lambda: None)
    monkeypatch.setattr(module.plt, "show", lambda **_kwargs: None)
    monkeypatch.setattr(
        FigureCanvasBase, "start_event_loop", paint_fresh_mask_and_accept
    )
    study = StudyData(
        np.ones((1, 15, 15), dtype=float),
        np.ones((1, 15, 15), dtype=np.uint8),
        np.asarray([0.0]),
    )
    landmark = LandmarkPair(aip=(7.0, 2.0), iip=(7.0, 12.0))
    output = tmp_path / "rois.json"
    saved_mask = np.zeros((15, 15), dtype=bool)
    saved_mask[0, 0] = True
    save_reference_roi_masks(output, [saved_mask])

    selected = select_reference_rois_interactively(
        study,
        [landmark],
        output,
        min_roi_voxels=2,
        resume_existing=False,
    )

    assert event_loop_called
    assert selected[0] is not None
    assert selected[0][11, 11]
    assert not selected[0][0, 0]
    np.testing.assert_array_equal(
        load_reference_roi_masks(output, study.mr.shape)[0], selected[0]
    )


def test_rectangular_roi_gui_displays_native_and_maps_paint_and_erase(monkeypatch, tmp_path):
    from cardiac_bullseye.lge import reference_roi as module

    callbacks = {}

    class FakeButton:
        def __init__(self, _axis, label):
            self.label = label

        def on_clicked(self, callback):
            callbacks[self.label] = callback

    internal_image = np.arange(40 * 24, dtype=float).reshape(40, 24)
    myocardium = np.zeros((40, 24), dtype=np.uint8)
    myocardium[15:31, 7:16] = 1

    def paint_erase_and_accept(canvas, timeout=0):
        axis = canvas.figure.axes[0]
        np.testing.assert_array_equal(axis.images[0].get_array(), internal_image.T)
        assert axis.get_xlim() == pytest.approx((-0.5, 39.5))
        assert axis.get_ylim() == pytest.approx((23.5, -0.5))
        assert axis.lines[0].get_xdata()[0] == 19.0
        assert axis.lines[0].get_ydata()[0] == 8.0

        def dispatch(name, point, button):
            display_x, display_y = axis.transData.transform(point)
            event = MouseEvent(name, canvas, display_x, display_y, button=button)
            canvas.callbacks.process(name, event)

        dispatch("button_press_event", (17.0, 11.0), MouseButton.LEFT)
        dispatch("motion_notify_event", (28.0, 11.0), MouseButton.LEFT)
        dispatch("button_release_event", (28.0, 11.0), MouseButton.LEFT)
        dispatch("button_press_event", (17.0, 11.0), MouseButton.RIGHT)
        dispatch("button_release_event", (17.0, 11.0), MouseButton.RIGHT)
        callbacks["Accept"](None)

    monkeypatch.setattr(module, "Button", FakeButton)
    monkeypatch.setattr(module, "_ensure_interactive_backend", lambda: None)
    monkeypatch.setattr(module.plt, "show", lambda **_kwargs: None)
    monkeypatch.setattr(FigureCanvasBase, "start_event_loop", paint_erase_and_accept)
    study = StudyData(internal_image[None], myocardium[None], np.asarray([0.0]))
    output = tmp_path / "rois.json"

    selected = select_reference_rois_interactively(
        study, [LandmarkPair(aip=(8.0, 19.0), iip=(14.0, 27.0))], output, min_roi_voxels=2
    )

    assert selected[0].shape == (40, 24)
    assert selected[0][25:29, 11].all()
    assert not selected[0][17, 11]
    assert not np.any(selected[0] & ~myocardium.astype(bool))
    np.testing.assert_array_equal(load_reference_roi_masks(output, study.mr.shape)[0], selected[0])


def test_scar_bullseye_plots_segments_1_to_16_and_marks_center_na(tmp_path):
    segment_ids = np.arange(1, 17, dtype=np.uint8).reshape(1, 4, 4)
    scar_mask = segment_ids % 2 == 0
    summary = compute_aha16_scar_percentages(scar_mask, segment_ids)
    output = tmp_path / "scar_bullseye.png"

    figure, axis = plot_lge_scar_bullseye(summary, output, show=False)

    assert output.is_file() and output.stat().st_size > 1000
    assert "N/A" in {text.get_text() for text in axis.texts}
    assert "17" not in {text.get_text() for text in axis.texts}
    np.testing.assert_allclose(axis.patches[0].get_facecolor(), to_rgba("#ffffff"))
    np.testing.assert_allclose(axis.patches[1].get_facecolor(), to_rgba("#d7301f"))
    colorbar_cmap = figure.axes[1].collections[-1].cmap
    np.testing.assert_allclose(colorbar_cmap(0.0), to_rgba("#ffffff"))
    np.testing.assert_allclose(colorbar_cmap(1.0), to_rgba("#d7301f"))
    labels = {text.get_text(): text for text in axis.texts}
    assert labels["1\n0.0%"].get_color() == "black"
    assert labels["2\n100.0%"].get_color() == "white"
    assert len(figure.axes) == 2


def test_scar_plot_retries_atomic_replace_and_overwrites_existing_png(
    monkeypatch, tmp_path
):
    segment_ids = np.arange(1, 17, dtype=np.uint8).reshape(1, 4, 4)
    summary = compute_aha16_scar_percentages(segment_ids % 2 == 0, segment_ids)
    output = tmp_path / "scar_bullseye.png"
    output.write_bytes(b"old png")
    real_savefig = Figure.savefig
    real_replace = os.replace
    replace_attempts = 0

    def reject_direct_save(figure, filename, *args, **kwargs):
        if Path(filename) == output:
            raise OSError(errno.EINVAL, "direct overwrite rejected", str(output))
        return real_savefig(figure, filename, *args, **kwargs)

    def fail_first_replace(source, destination):
        nonlocal replace_attempts
        replace_attempts += 1
        if replace_attempts == 1:
            raise PermissionError("temporarily busy")
        return real_replace(source, destination)

    monkeypatch.setattr(Figure, "savefig", reject_direct_save)
    monkeypatch.setattr(os, "replace", fail_first_replace)
    monkeypatch.setattr(time, "sleep", lambda _: None)

    plot_lge_scar_bullseye(summary, output, show=False)

    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert replace_attempts == 2
    assert not list(tmp_path.glob(".scar_bullseye.*.tmp.png"))


def test_scar_plot_keeps_old_png_if_atomic_replace_remains_locked(
    monkeypatch, tmp_path
):
    segment_ids = np.arange(1, 17, dtype=np.uint8).reshape(1, 4, 4)
    summary = compute_aha16_scar_percentages(segment_ids % 2 == 0, segment_ids)
    output = tmp_path / "scar_bullseye.png"
    output.write_bytes(b"old png")

    monkeypatch.setattr(
        os, "replace", lambda *_: (_ for _ in ()).throw(PermissionError("busy"))
    )
    monkeypatch.setattr(time, "sleep", lambda _: None)

    with pytest.raises(OSError, match="Close any application"):
        plot_lge_scar_bullseye(summary, output, show=False)

    assert output.read_bytes() == b"old png"
    assert not list(tmp_path.glob(".scar_bullseye.*.tmp.png"))


def test_scar_mask_is_restored_to_the_complete_original_lge_grid(tmp_path):
    lge = np.zeros((4, 5, 7), dtype=np.float32)
    myocardium = np.zeros_like(lge, dtype=np.uint8)
    myocardium[1:3, 1:4, 1:6] = 1
    lge_image = sitk.GetImageFromArray(lge)
    lge_image.SetSpacing((0.8, 0.9, 6.0))
    lge_image.SetOrigin((1.0, 2.0, 3.0))
    lge_image.SetDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
    mask_image = sitk.GetImageFromArray(myocardium)
    mask_image.CopyInformation(lge_image)
    lge_path = tmp_path / "lge.nii.gz"
    mask_path = tmp_path / "myocardium.nrrd"
    output = tmp_path / "scar.nii.gz"
    sitk.WriteImage(lge_image, str(lge_path))
    sitk.WriteImage(mask_image, str(mask_path))
    cropped_scar = np.zeros((2, 7, 5), dtype=bool)
    cropped_scar[0, 5, 2] = True
    cropped_scar[1, 4, 3] = True
    cropped_scar[0, 0, 0] = True  # Saved scar remains clipped to myocardium.

    save_lge_scar_mask(output, lge_path, mask_path, cropped_scar)

    saved_image = sitk.ReadImage(str(output))
    saved = sitk.GetArrayFromImage(saved_image)
    assert saved.shape == lge.shape
    assert saved[1, 2, 5] == 1
    assert saved[2, 3, 4] == 1
    assert np.count_nonzero(saved) == 2
    np.testing.assert_allclose(saved_image.GetSpacing(), lge_image.GetSpacing())
    np.testing.assert_allclose(saved_image.GetOrigin(), lge_image.GetOrigin())
    np.testing.assert_allclose(saved_image.GetDirection(), lge_image.GetDirection())


def test_segment_csv_records_counts_and_blank_missing_percentages(tmp_path):
    segment_ids = np.asarray([[[1, 1, 2, 2]]], dtype=np.uint8)
    scar_mask = np.asarray([[[True, False, True, True]]], dtype=bool)
    summary = compute_aha16_scar_percentages(scar_mask, segment_ids)
    output = tmp_path / "segments.csv"

    save_aha16_scar_csv(output, summary)

    rows = output.read_text(encoding="utf-8").splitlines()
    assert rows[0] == (
        "segment_id,segment_name,myocardium_voxels,scar_voxels,scar_percent"
    )
    assert rows[1] == "1,basal anterior,2,1,50.000000"
    assert rows[2] == "2,basal anteroseptal,2,2,100.000000"
    assert rows[3] == "3,basal inferoseptal,0,0,"
    assert len(rows) == 17


def test_deleted_slice_clears_a_stale_reference_roi_without_opening_gui(
    tmp_path,
):
    study = StudyData(
        np.zeros((1, 5, 5), dtype=float),
        np.ones((1, 5, 5), dtype=np.uint8),
        np.asarray([0.0]),
    )
    output = tmp_path / "reference_rois.json"
    stale_polygon = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0)]
    save_reference_roi_polygons(output, [stale_polygon])

    polygons = select_reference_rois_interactively(
        study,
        [None],
        output,
        min_roi_voxels=2,
        deleted_slice_indices={0},
    )

    assert polygons == [None]
    assert load_reference_roi_polygons(output, 1) == [None]


def test_skipped_landmark_slice_still_gets_scar_threshold_and_aha_segments():
    size = 25
    y, x = np.ogrid[:size, :size]
    ring = ((x - 12) ** 2 + (y - 12) ** 2 >= 25) & ((x - 12) ** 2 + (y - 12) ** 2 <= 81)
    myocardium = np.repeat(ring[None, :, :], 3, axis=0).astype(np.uint8)
    lge = np.zeros_like(myocardium, dtype=float)
    roi_masks = np.zeros_like(myocardium, dtype=bool)
    roi_masks[:, 4:12, 4:12] = myocardium[:, 4:12, 4:12] > 0
    lge[1, 12, 20] = 100.0
    study = StudyData(lge, myocardium, np.asarray([0.0, 6.0, 12.0]))
    apical = LandmarkPair(aip=(12.0, 5.0), iip=(12.0, 19.0))
    basal = LandmarkPair(aip=(11.0, 5.0), iip=(13.0, 19.0))
    landmarks = [apical, None, basal]

    scar_result = compute_lge_scar(study, landmarks, roi_masks, min_roi_voxels=6)
    segment_ids = compute_aha16_segment_ids(study, landmarks, six_segment_warp=False)

    assert scar_result.thresholds[1] == 0.0
    assert scar_result.scar_mask[1, 12, 20]
    assert np.all(segment_ids[1][ring] >= 7)
    assert np.all(segment_ids[1][ring] <= 12)


def test_empty_myocardium_slice_does_not_require_a_reference_roi():
    myocardium = np.ones((3, 4, 4), dtype=np.uint8)
    myocardium[1] = 0
    lge = np.zeros_like(myocardium, dtype=float)
    roi_masks = myocardium.astype(bool)
    study = StudyData(lge, myocardium, np.asarray([0.0, 6.0, 12.0]))
    landmark = LandmarkPair(aip=(1.5, 0.0), iip=(1.5, 3.0))

    result = compute_lge_scar(
        study, [landmark, None, landmark], roi_masks, min_roi_voxels=6
    )

    assert np.isnan(result.thresholds[1])
    assert result.reference_voxel_counts[1] == 0
    assert not result.scar_mask[1].any()


def test_empty_myocardium_slice_does_not_open_the_roi_gui(monkeypatch, tmp_path):
    study = StudyData(
        np.zeros((1, 5, 5), dtype=float),
        np.zeros((1, 5, 5), dtype=np.uint8),
        np.asarray([0.0]),
    )
    monkeypatch.setattr(
        "cardiac_bullseye.lge.reference_roi._ensure_interactive_backend",
        lambda: pytest.fail("ROI GUI must not open for an empty myocardium slice"),
    )

    polygons = select_reference_rois_interactively(
        study, [None], tmp_path / "rois.json", min_roi_voxels=2
    )

    assert polygons == [None]


def _projection_study(slice_count):
    y, x = np.ogrid[:25, :25]
    ring = ((x - 12) ** 2 + (y - 12) ** 2 >= 25) & (
        (x - 12) ** 2 + (y - 12) ** 2 <= 81
    )
    mask = np.repeat(ring[None], slice_count, axis=0).astype(np.uint8)
    study = StudyData(np.zeros_like(mask, dtype=float), mask, np.arange(slice_count))
    pair = LandmarkPair(aip=(17.0, 7.0), iip=(19.0, 12.0))
    return study, pair


def test_projected_ids_use_exact_radial_boundaries_and_asymmetric_angles():
    radii = [0.0, np.nextafter(1.0, 0.0), 1.0, np.nextafter(2.0, 0.0),
             2.0, np.nextafter(3.0, 0.0), 3.0, np.nextafter(4.0, 0.0), 4.0]
    study, pair = _projection_study(len(radii))

    ids = lge_api.compute_aha_segment_ids(
        study, [pair] * len(radii), radii, include_segment_17=True
    )

    # This oblique voxel has canonical angle 56.2 degrees before the warp,
    # about 75 degrees afterward: basal segment 3 becomes segment 4.
    np.testing.assert_array_equal(ids[:, 16, 18], [17, 17, 15, 15, 10, 10, 4, 4, 4])
    np.testing.assert_array_equal(ids[:, 8, 18], [17, 17, 14, 14, 8, 8, 2, 2, 2])
    assert ids.dtype == np.uint8
    assert not ids[study.territory_mask == 0].any()


def test_projected_ids_keep_optional_warp_and_manual_centers():
    study, pair = _projection_study(1)
    default = lge_api.compute_aha_segment_ids(
        study, [pair], [3.0], include_segment_17=False, six_segment_warp=False
    )
    manual = lge_api.compute_aha_segment_ids(
        study, [pair], [3.0], include_segment_17=False, six_segment_warp=False,
        center_overrides=[(15.0, 12.0)],
    )
    assert default[0, 16, 18] == 3
    assert manual[0, 16, 18] == 4


def test_projected_ids_interpolate_missing_landmarks_and_keep_deleted_tail_zero():
    study, pair = _projection_study(4)
    basal = LandmarkPair(aip=(15.0, 7.0), iip=(19.0, 14.0))
    middle = LandmarkPair(aip=(16.0, 7.0), iip=(19.0, 13.0))
    options = dict(include_segment_17=True, deleted_slice_indices={3})
    actual = lge_api.compute_aha_segment_ids(
        study, [pair, None, basal, None], [0.0, 2.0, 4.0, 6.0], **options
    )
    expected = lge_api.compute_aha_segment_ids(
        study, [pair, middle, basal, None], [0.0, 2.0, 4.0, 6.0], **options
    )
    np.testing.assert_array_equal(actual, expected)
    assert np.all(actual[0][study.territory_mask[0] > 0] == 17)
    assert set(np.unique(actual[1])) == {0, 7, 8, 9, 10, 11, 12}
    assert not actual[3].any()


@pytest.mark.parametrize("radii", [
    [0.0], [[0.0, 2.0, 4.0]], [0.0, np.nan, 4.0], [0.0, 2.0, np.inf],
    [0.0, 2.0, 2.0], [0.0, 4.0, 2.0], [-0.1, 2.0, 4.0], [0.0, 2.0, 4.1],
])
def test_projected_ids_reject_invalid_radii(radii):
    study, pair = _projection_study(3)
    with pytest.raises(ValueError, match="source_radii"):
        lge_api.compute_aha_segment_ids(
            study, [pair] * 3, radii, include_segment_17=True
        )


def test_projected_ids_validate_retained_applicability_and_deletion():
    study, pair = _projection_study(3)
    study.territory_mask[0] = 0
    ids = lge_api.compute_aha_segment_ids(
        study, [None, pair, pair], [-2.0, 1.0, 4.0], include_segment_17=False
    )
    assert not ids[0].any()
    assert 17 not in ids
    for deleted in ({1}, {0, 1, 2}, {3}):
        with pytest.raises(ValueError):
            lge_api.compute_aha_segment_ids(
                study, [pair] * 3, [0.0, 2.0, 4.0],
                include_segment_17=True, deleted_slice_indices=deleted,
            )
    with pytest.raises(ValueError, match="17"):
        lge_api.compute_aha_segment_ids(
            study, [pair] * 3, [-2.0, 0.5, 4.0], include_segment_17=False
        )


def test_projected_ids_validate_landmarks_and_centers():
    study, pair = _projection_study(3)
    for landmarks, centers, message in (
        ([pair], None, "landmark"),
        ([None] * 3, None, "AIP/IIP"),
        ([pair] * 3, [None], "center_overrides"),
        ([pair] * 3, [(np.nan, 12.0), None, None], "finite"),
    ):
        with pytest.raises(ValueError, match=message):
            lge_api.compute_aha_segment_ids(
                study, landmarks, [0.0, 2.0, 4.0], include_segment_17=True,
                center_overrides=centers,
            )


def test_segment_17_burden_pools_unequal_native_slice_counts():
    segments = np.array([[[17, 0, 0]], [[17, 17, 17]]], dtype=np.uint8)
    scar = np.array([[[True, False, False]], [[False, False, False]]])
    summary = lge_api.compute_aha_scar_percentages(scar, segments, segment_count=17)
    assert isinstance(summary, lge_api.AhaScarSummary)
    assert summary.myocardial_voxel_counts[16] == 4
    assert summary.scar_voxel_counts[16] == 1
    assert summary.scar_percentages[16] == 25.0  # Not the 50% slice mean.
    assert np.isnan(summary.scar_percentages[:16]).all()


@pytest.mark.parametrize("count", [0, 15, 18, 16.0, True])
def test_general_scar_summary_requires_exactly_16_or_17_segments(count):
    with pytest.raises(ValueError, match="16 or 17"):
        lge_api.compute_aha_scar_percentages([False], [0], segment_count=count)


def test_general_scar_summary_rejects_unrepresentable_ids_and_shape_mismatch():
    for scar, segments, count in (([True], [17], 16), ([True], [18], 17),
                                   ([True], [1.5], 17), ([True], [-1], 17),
                                   ([True], [[1]], 17)):
        with pytest.raises(ValueError):
            lge_api.compute_aha_scar_percentages(scar, segments, segment_count=count)


@pytest.mark.parametrize("count", [16, 17])
@pytest.mark.parametrize("compatibility_name", [False, True])
def test_general_scar_csv_preserves_columns_and_exact_segment_row_count(
    tmp_path, count, compatibility_name
):
    ids = np.arange(1, count + 1, dtype=np.uint8)
    summary = lge_api.compute_aha_scar_percentages(ids == count, ids, segment_count=count)
    save_csv = save_aha16_scar_csv if compatibility_name else lge_api.save_aha_scar_csv
    output = save_csv(tmp_path / "aha16_scar.csv", summary)
    with output.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        rows = list(reader)
    assert reader.fieldnames == ["segment_id", "segment_name", "myocardium_voxels",
                                 "scar_voxels", "scar_percent"]
    assert len(rows) == count
    assert rows[-1]["segment_id"] == str(count)
    assert rows[-1]["scar_percent"] == "100.000000"
    if count == 17:
        assert rows[-1]["segment_name"] == "apex"


def test_scar_plot_16_segments_without_apex_leaves_center_blank():
    summary = compute_aha16_scar_percentages([False], [0])
    _, axis = plot_lge_scar_bullseye(summary, include_apex=False, show=False)
    assert not any(text.get_position() == (0.0, 0.0) for text in axis.texts)
    assert len(axis.patches) == 16


def test_scar_plot_measures_segment_17_and_rejects_hiding_it():
    summary = lge_api.compute_aha_scar_percentages(
        [True, False, False, False], [17, 17, 17, 17], segment_count=17
    )
    figure, axis = plot_lge_scar_bullseye(summary, include_apex=True, show=False)
    assert axis.get_title() == "AHA-17 infarct scar burden"
    assert "17\n25.0%" in {text.get_text() for text in axis.texts}
    color_map = figure.axes[1].collections[-1].cmap
    np.testing.assert_allclose(axis.patches[-1].get_facecolor(), color_map(0.25))
    assert len(axis.patches) == 17
    with pytest.raises(ValueError, match="include_apex"):
        plot_lge_scar_bullseye(summary, include_apex=False, show=False)
