from __future__ import annotations

import csv
import json
from dataclasses import fields, replace

import numpy as np
import pytest
import SimpleITK as sitk

from cardiac_bullseye.config import load_config
from cardiac_bullseye.territory.io import save_landmarks_json
from cardiac_bullseye.territory.models import LandmarkPair
from cardiac_bullseye.workflows import generate_lge_bullseye


def test_lge_workflow_uses_toml_grid_and_case_prefixed_outputs(monkeypatch, tmp_path):
    """Changing from TOML to hard-coded defaults must break this workflow test."""
    size = 25
    y, x = np.ogrid[:size, :size]
    ring = ((x - 12) ** 2 + (y - 12) ** 2 >= 25) & ((x - 12) ** 2 + (y - 12) ** 2 <= 81)
    mask = np.repeat(ring[None, :, :], 3, axis=0).astype(np.uint8)
    lge = np.zeros_like(mask, dtype=np.float32)
    for index, value in enumerate((10.0, 20.0, 30.0)):
        lge[index, ring] = value
        lge[index, 12, 20] = value + 100.0

    lge_image = sitk.GetImageFromArray(lge)
    mask_image = sitk.GetImageFromArray(mask)
    lge_image.SetSpacing((1.0, 1.0, 6.0))
    mask_image.CopyInformation(lge_image)
    lge_path = tmp_path / "lge.nii.gz"
    mask_path = tmp_path / "mask.seg.nrrd"
    sitk.WriteImage(lge_image, str(lge_path))
    sitk.WriteImage(mask_image, str(mask_path))

    config_path = tmp_path / "case.toml"
    config_path.write_text(
        f"""
[case]
id = "020501"
output_directory = "results"
[lge_bullseye]
lge = '{lge_path.as_posix()}'
mask = '{mask_path.as_posix()}'
angle_count = 24
radial_count = 9
angular_smoothing_sigma_deg = 0.0
radial_smoothing_sigma = 0.0
six_segment_warp = false
scar_sd_multiplier = 5.0
normal_roi_min_voxels = 20
""",
        encoding="utf-8",
    )
    config = load_config(config_path, "lge-bullseye")
    landmark = LandmarkPair(aip=(12.0, 5.0), iip=(12.0, 19.0))

    def select_fresh_landmarks(study, output_path, *, resume_existing,
                               transpose_for_display, cancel_is_error):
        assert resume_existing is False
        assert transpose_for_display is True
        landmarks = [landmark] * study.slice_count
        save_landmarks_json(output_path, landmarks, image_shape=study.mr.shape)
        return landmarks

    monkeypatch.setattr(
        "cardiac_bullseye.workflows.bullseye_frame.select_landmarks_interactively",
        select_fresh_landmarks,
    )
    reference_mask = np.zeros((size, size), dtype=bool)
    reference_mask[:, :12] = ring[:, :12]

    def select_fresh_reference_rois(
        study,
        landmarks,
        output_path,
        min_roi_voxels,
        *,
        sd_multiplier,
        deleted_slice_indices,
        resume_existing,
    ):
        assert resume_existing is False
        return [reference_mask] * study.slice_count

    monkeypatch.setattr(
        "cardiac_bullseye.workflows.lge.select_reference_rois_interactively",
        select_fresh_reference_rois,
    )

    outputs = generate_lge_bullseye(config)

    assert config.lge_bullseye.lge_scar_mask is None
    assert [field.name for field in fields(outputs)] == [
        "landmarks", "numeric_data", "figure", "reference_rois",
        "scar_mask", "scar_segments", "scar_figure",
    ]
    assert all(getattr(outputs, field.name).is_file() for field in fields(outputs))
    assert outputs.landmarks.name == "020501_lge_landmarks.json"
    assert outputs.numeric_data.name == "020501_lge_intensity_bullseye.npz"
    assert outputs.figure.name == "020501_lge_intensity_bullseye.png"
    assert outputs.reference_rois.name == "020501_lge_normal_rois.json"
    assert outputs.scar_mask.name == "020501_lge_scar_mask.nii.gz"
    assert outputs.scar_segments.name == "020501_lge_scar_segments.csv"
    assert outputs.scar_figure.name == "020501_lge_scar_bullseye.png"
    assert outputs.numeric_data.is_file()
    assert outputs.figure.is_file()
    assert outputs.reference_rois.is_file()
    assert outputs.scar_mask.is_file()
    assert outputs.scar_segments.is_file()
    assert outputs.scar_figure.is_file()
    with np.load(outputs.numeric_data) as payload:
        assert payload["intensity"].shape == (9, 24)
        assert float(payload["apex_intensity"]) > 10.0
    saved_scar = sitk.GetArrayFromImage(sitk.ReadImage(str(outputs.scar_mask)))
    assert np.count_nonzero(saved_scar) == 3
    with outputs.reference_rois.open(encoding="utf-8") as reference_file:
        reference_payload = json.load(reference_file)
    assert reference_payload["scar_sd_multiplier"] == 5.0
    assert reference_payload["slices"][0]["mean"] == 10.0
    assert reference_payload["slices"][0]["std"] == 0.0
    assert reference_payload["slices"][0]["threshold"] == 10.0
    with outputs.scar_segments.open(encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert len(rows) == 16
    assert sum(int(row["scar_voxels"]) for row in rows) == 3

    reused_config = replace(
        config,
        lge_bullseye=replace(config.lge_bullseye, reuse_saved_points=True),
    )
    reference_roi_selections = []

    def redraw_reference_rois(*args, **kwargs):
        reference_roi_selections.append(True)
        return select_fresh_reference_rois(*args, **kwargs)

    monkeypatch.setattr(
        "cardiac_bullseye.workflows.lge.select_reference_rois_interactively",
        redraw_reference_rois,
    )
    monkeypatch.setattr(
        "cardiac_bullseye.workflows.bullseye_frame.select_landmarks_interactively",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("annotation UI must not open")
        ),
    )
    reused_outputs = generate_lge_bullseye(reused_config)
    assert reused_outputs.figure.is_file()
    assert reference_roi_selections == [True]


PAIR = LandmarkPair(aip=(12.0, 7.0), iip=(12.0, 21.0))


def _write_shared_case(tmp_path, *, slice_count=5, lesion=True):
    """Use a non-square grid and empty end padding to expose coordinate errors."""
    y, x = np.ogrid[:25, :29]
    ring = ((x - 14) ** 2 + (y - 12) ** 2 >= 25) & (
        (x - 14) ** 2 + (y - 12) ** 2 <= 81
    )
    mask = np.zeros((slice_count + 2, 25, 29), dtype=np.uint8)
    mask[1:-1] = ring
    lge = np.zeros(mask.shape, dtype=np.float32)
    for index in range(slice_count):
        lge[index + 1, ring] = (index + 1) * 10.
        if lesion:
            lge[index + 1, 12, 22] += 100.
    for name, array in (("lge.nii.gz", lge), ("mask.nii.gz", mask)):
        image = sitk.GetImageFromArray(array)
        image.SetSpacing((1., 1., 6.))
        sitk.WriteImage(image, str(tmp_path / name))
    config_path = tmp_path / "case.toml"
    config_path.write_text('''
[case]
id = "shared"
output_directory = "results"
[lge_bullseye]
lge = "lge.nii.gz"
mask = "mask.nii.gz"
landmarks_path = "shared.json"
reuse_saved_points = true
angle_count = 24
radial_count = 13
angular_smoothing_sigma_deg = 0.0
radial_smoothing_sigma = 0.0
six_segment_warp = false
normal_roi_min_voxels = 20
''', encoding="utf-8")
    return load_config(config_path, "lge-bullseye"), ring.T


def _no_annotation(*_args, **_kwargs):
    pytest.fail("saved landmarks must not open annotation UI")


def _stub_rois(monkeypatch, ring):
    roi = ring.copy()
    roi[14:] = False
    calls = []

    def select(study, landmarks, path, minimum, *, sd_multiplier,
               deleted_slice_indices, resume_existing):
        assert resume_existing is False
        assert minimum == 20
        assert sd_multiplier == 5.
        calls.append((landmarks, set(deleted_slice_indices)))
        return [None if index in deleted_slice_indices else roi
                for index in range(study.slice_count)]

    monkeypatch.setattr(
        "cardiac_bullseye.workflows.lge.select_reference_rois_interactively", select
    )
    return calls


@pytest.mark.parametrize("mode", ["per-slice", "single-reference"])
@pytest.mark.parametrize("projection, radii, segment_count", [
    ("1-4", [1., 1.75, 2.5, 3.25, 4.], 16),
    ("0-4", [0., 1., 2., 3., 4.], 17),
])
def test_shared_reuse_maps_intensity_and_native_scar_on_same_frame(
    monkeypatch, tmp_path, mode, projection, radii, segment_count
):
    from cardiac_bullseye.workflows import lge as module, lge_common
    from matplotlib.patches import Circle

    config, ring = _write_shared_case(tmp_path)
    path = config.lge_bullseye.landmarks_path
    centers = [(12., 14.)] * 5
    save_landmarks_json(path, [PAIR] * 5, image_shape=(5, 29, 25),
                        center_overrides=centers)
    saved = path.read_bytes()
    for selector in ("select_landmarks_interactively", "select_single_reference_interactively"):
        monkeypatch.setattr("cardiac_bullseye.workflows.bullseye_frame." + selector,
                            _no_annotation)
    roi_calls = _stub_rois(monkeypatch, ring)
    captured = {}
    real_compute = module.compute_lge_bullseye_products

    def compute(*args, **kwargs):
        np.testing.assert_allclose(kwargs["source_radii"], radii)
        assert kwargs["minimum_radius"] == radii[0]
        assert kwargs["center_overrides"] == centers
        result = real_compute(*args, **kwargs)
        captured["products"] = result
        return result

    real_plot = lge_common.plot_lge_scar_bullseye

    def plot(*args, **kwargs):
        figure, axis = real_plot(*args, **kwargs)
        captured["axis"] = axis
        return figure, axis

    monkeypatch.setattr(module, "compute_lge_bullseye_products", compute)
    monkeypatch.setattr(lge_common, "plot_lge_scar_bullseye", plot)

    outputs = generate_lge_bullseye(config, landmark_mode=mode, projection=projection)

    assert outputs.landmarks == path
    assert path.read_bytes() == saved
    assert roi_calls == [([PAIR] * 5, set())]
    products = captured["products"]
    assert products.scar.scar_voxel_counts.sum() == 5
    assert products.scar.myocardial_voxel_counts.sum() == ring.sum() * 5
    with outputs.scar_segments.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == segment_count
    axis = captured["axis"]
    center_labels = [text.get_text() for text in axis.texts
                     if text.get_position() == (0., 0.)]
    cap = next(patch for patch in axis.patches if isinstance(patch, Circle))
    if projection == "0-4":
        assert products.scar.scar_voxel_counts[16] == 1
        assert products.scar.myocardial_voxel_counts[16] == ring.sum()
        expected = 100. / ring.sum()
        assert products.scar.scar_percentages[16] == pytest.approx(expected)
        assert float(rows[16]["scar_percent"]) == pytest.approx(expected, abs=1e-6)
        assert center_labels == [f"17\n{expected:.1f}%"]
        assert cap.get_facecolor() != (1., 1., 1., 1.)
        # Angular variation fades to zero at the origin but exists at r=1.
        np.testing.assert_allclose(products.intensity.intensity[0],
                                   products.intensity.intensity[0, 0])
        assert np.ptp(products.intensity.intensity[3]) > 0.
        full_contrast = np.ptp(products.intensity.intensity[3])
        assert np.ptp(products.intensity.intensity[1]) == pytest.approx(
            full_contrast * 7. / 27.
        )
        assert np.ptp(products.intensity.intensity[2]) == pytest.approx(
            full_contrast * 20. / 27.
        )
    else:
        # Original thirds: two basal, one middle, two apical slices.
        np.testing.assert_array_equal(products.scar.scar_voxel_counts,
                                      [0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0])
        assert center_labels == ["N/A"]
        assert cap.get_facecolor() == (1., 1., 1., 1.)
        assert products.intensity.apex_intensity == pytest.approx(10. + 100. / ring.sum())
    with outputs.reference_rois.open(encoding="utf-8") as stream:
        reference = json.load(stream)
    assert [entry["threshold"] for entry in reference["slices"]] == [10., 20., 30., 40., 50.]
    saved_scar = sitk.GetArrayFromImage(sitk.ReadImage(str(outputs.scar_mask)))
    assert saved_scar.shape == (7, 25, 29)
    assert not saved_scar[[0, -1]].any()
    assert np.count_nonzero(saved_scar) == 5


@pytest.mark.parametrize("projection, expected", [
    ("1-4", [10., 40./3., 50./3., 20., 70./3., 80./3., 30.,
             100./3., 110./3., 40., 130./3., 140./3., 50.]),
    ("0-4", [10., 40./3., 50./3., 20., 70./3., 80./3., 30.,
             100./3., 110./3., 40., 130./3., 140./3., 50.]),
])
def test_shared_projection_preserves_mask_anchored_continuous_values(
    monkeypatch, tmp_path, projection, expected
):
    config, ring = _write_shared_case(tmp_path, lesion=False)
    save_landmarks_json(config.lge_bullseye.landmarks_path, [PAIR] * 5,
                        image_shape=(5, 29, 25))
    _stub_rois(monkeypatch, ring)
    outputs = generate_lge_bullseye(config, projection=projection)
    with np.load(outputs.numeric_data) as payload:
        np.testing.assert_allclose(payload["intensity"],
                                   np.repeat(np.asarray(expected)[:, None], 24, axis=1))
        np.testing.assert_allclose(payload["radial_positions"],
                                   np.linspace(1. if projection == "1-4" else 0., 4., 13))


@pytest.mark.parametrize("reuse, mode", [(True, "single-reference"), (False, "per-slice")])
def test_missing_shared_file_falls_back_and_fresh_annotation_overwrites(
    monkeypatch, tmp_path, reuse, mode
):
    config, ring = _write_shared_case(tmp_path)
    config = replace(config, lge_bullseye=replace(config.lge_bullseye,
                                               reuse_saved_points=reuse))
    path = config.lge_bullseye.landmarks_path
    if not reuse:
        path.write_text("old data must be overwritten", encoding="utf-8")

    def select(study, output_path, *, resume_existing, transpose_for_display,
               cancel_is_error):
        assert resume_existing is False
        assert transpose_for_display is True
        assert output_path == path
        points = [PAIR, None, PAIR, PAIR, None]
        save_landmarks_json(path, points, image_shape=study.mr.shape,
                            deleted_slice_indices={4},
                            center_overrides=[None, None, (13., 14.), None, None])
        return points

    for selector in ("select_landmarks_interactively", "select_single_reference_interactively"):
        monkeypatch.setattr("cardiac_bullseye.workflows.bullseye_frame." + selector,
                            _no_annotation)
    selector = ("select_single_reference_interactively" if reuse
                else "select_landmarks_interactively")
    monkeypatch.setattr("cardiac_bullseye.workflows.bullseye_frame." + selector, select)
    calls = _stub_rois(monkeypatch, ring)
    outputs = generate_lge_bullseye(config, landmark_mode=mode)
    assert outputs.landmarks == path
    assert calls == [([PAIR, None, PAIR, PAIR, None], {4})]
    assert json.loads(path.read_text())["slices"][0]["aip"] == [12., 7.]
    with outputs.scar_segments.open(encoding="utf-8", newline="") as stream:
        assert sum(int(row["scar_voxels"]) for row in csv.DictReader(stream)) == 4


@pytest.mark.parametrize("explicit", [False, True])
def test_legacy_default_coordinates_migrate_only_at_default_path(
    monkeypatch, tmp_path, explicit
):
    from cardiac_bullseye.workflows import lge as module

    config, ring = _write_shared_case(tmp_path)
    path = config.case.output_path("lge_landmarks.json")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"image_shape": [5, 25, 29], "slices": [
        {"aip": [7., 12.], "iip": [21., 12.], "center": [14., 12.]}
    ] * 5}))
    config = replace(config, lge_bullseye=replace(config.lge_bullseye,
                                               landmarks_path=path if explicit else None))
    _stub_rois(monkeypatch, ring)
    real_compute = module.compute_lge_bullseye_products

    def compute(study, landmarks, *args, **kwargs):
        assert landmarks == [PAIR] * 5
        assert kwargs["center_overrides"] == [(12., 14.)] * 5
        return real_compute(study, landmarks, *args, **kwargs)

    monkeypatch.setattr(module, "compute_lge_bullseye_products", compute)
    if explicit:
        with pytest.raises(ValueError, match="Ambiguous unversioned"):
            generate_lge_bullseye(config, projection="1-4")
    else:
        assert generate_lge_bullseye(config, projection="1-4").landmarks == path


@pytest.mark.parametrize("basename, accepted", [
    ("other_patient_landmarks.json", False),
    ("patient_lge_landmarks.json", True),
])
def test_lge_workflow_uses_case_specific_common_legacy_provenance(
    monkeypatch, tmp_path, basename, accepted
):
    config, ring = _write_shared_case(tmp_path)
    path = tmp_path / basename
    path.write_text(json.dumps({"image_shape": [5, 29, 25], "slices": [
        {"aip": [12., 7.], "iip": [12., 21.], "center": [12., 14.]}
    ] * 5}))
    config = replace(
        config, case=replace(config.case, id="patient_lge"),
        lge_bullseye=replace(config.lge_bullseye, landmarks_path=path),
    )
    monkeypatch.setattr(
        "cardiac_bullseye.workflows.bullseye_frame.select_landmarks_interactively",
        _no_annotation,
    )
    _stub_rois(monkeypatch, ring)
    if accepted:
        assert generate_lge_bullseye(config).landmarks == path
    else:
        with pytest.raises(ValueError, match="Ambiguous unversioned"):
            generate_lge_bullseye(config)


@pytest.mark.parametrize("problem, match", [
    ("shape", "image shape"), ("version", "Unsupported landmark coordinate version"),
    ("one-slice", "at least two retained slices"), ("deletion", "contiguous basal")
])
def test_invalid_shared_frame_fails_before_roi_drawing(monkeypatch, tmp_path, problem, match):
    config, _ring = _write_shared_case(tmp_path, slice_count=1 if problem == "one-slice" else 5)
    path = config.lge_bullseye.landmarks_path
    count = 1 if problem == "one-slice" else 5
    save_landmarks_json(path, [PAIR] * count, image_shape=(count, 29, 25))
    payload = json.loads(path.read_text())
    if problem == "shape":
        payload["image_shape"] = [5, 25, 29]
    elif problem == "version":
        payload["coordinate_version"] = 99
    elif problem == "deletion":
        payload["slices"][2] = {"status": "deleted"}
    path.write_text(json.dumps(payload))
    monkeypatch.setattr("cardiac_bullseye.workflows.lge.select_reference_rois_interactively",
                        lambda *_args, **_kwargs: pytest.fail("ROI UI must not open"))
    with pytest.raises(ValueError, match=match):
        generate_lge_bullseye(config, projection="0-4")
