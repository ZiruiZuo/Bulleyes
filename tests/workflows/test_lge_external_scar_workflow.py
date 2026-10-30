from __future__ import annotations

from dataclasses import fields, replace
import json

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import SimpleITK as sitk

from cardiac_bullseye.config import load_config
from cardiac_bullseye.coronary import CardiacGeometry, ProjectedCoronaryPath
from cardiac_bullseye.territory.io import save_landmarks_json
from cardiac_bullseye.territory.models import LandmarkPair
from cardiac_bullseye.workflows import generate_lge_bullseye


def _write_external_scar_case(tmp_path, *, scar_foreground: bool):
    size = 25
    y, x = np.ogrid[:size, :size]
    ring = (
        ((x - 12) ** 2 + (y - 12) ** 2 >= 25)
        & ((x - 12) ** 2 + (y - 12) ** 2 <= 81)
    )
    myocardium = np.repeat(ring[None, :, :], 3, axis=0).astype(np.uint8)
    lge = np.zeros_like(myocardium, dtype=np.float32)
    for index, value in enumerate((10.0, 20.0, 30.0)):
        lge[index, ring] = value
    scar = np.zeros_like(myocardium, dtype=np.uint8)
    if scar_foreground:
        scar[:, 12, 20] = 1

    lge_image = sitk.GetImageFromArray(lge)
    myocardium_image = sitk.GetImageFromArray(myocardium)
    scar_image = sitk.GetImageFromArray(scar)
    lge_image.SetSpacing((1.0, 1.0, 6.0))
    myocardium_image.CopyInformation(lge_image)
    scar_image.CopyInformation(lge_image)
    lge_path = tmp_path / "lge.nii.gz"
    myocardium_path = tmp_path / "myocardium.seg.nrrd"
    scar_path = tmp_path / "scar.nii.gz"
    sitk.WriteImage(lge_image, str(lge_path))
    sitk.WriteImage(myocardium_image, str(myocardium_path))
    sitk.WriteImage(scar_image, str(scar_path))

    config_path = tmp_path / "case.toml"
    config_path.write_text(
        f"""
[case]
id = "032602"
output_directory = "results"
[lge_bullseye]
lge = '{lge_path.as_posix()}'
mask = '{myocardium_path.as_posix()}'
lge_scar_mask = '{scar_path.as_posix()}'
angle_count = 24
radial_count = 9
angular_smoothing_sigma_deg = 0.0
radial_smoothing_sigma = 0.0
six_segment_warp = false
""",
        encoding="utf-8",
    )
    return load_config(config_path, "lge-bullseye")


def _fake_fresh_landmarks(study, output_path, *, resume_existing, cancel_is_error,
                          transpose_for_display):
    assert resume_existing is False
    assert cancel_is_error is True
    assert transpose_for_display is True
    landmarks = [LandmarkPair(aip=(12.0, 5.0), iip=(12.0, 19.0))] * study.slice_count
    save_landmarks_json(output_path, landmarks, image_shape=study.mr.shape)
    return landmarks


def _forbidden(*_args, **_kwargs):
    raise AssertionError("external-scar workflow must not select ROIs or compute scar")


@pytest.fixture(autouse=True)
def forbid_interactive_scar_calculation(monkeypatch):
    """External runs must never open the normal-ROI UI or estimate scar."""
    monkeypatch.setattr(
        "cardiac_bullseye.workflows.lge.select_reference_rois_interactively", _forbidden,
    )
    monkeypatch.setattr("cardiac_bullseye.workflows.lge.compute_lge_scar", _forbidden)


def test_external_scar_branch_writes_landmarks_and_scar_png(monkeypatch, tmp_path):
    """Replacing this with the ROI/scar workflow must fail this test."""
    config = _write_external_scar_case(tmp_path, scar_foreground=True)
    monkeypatch.setattr(
        "cardiac_bullseye.workflows.bullseye_frame.select_landmarks_interactively",
        _fake_fresh_landmarks,
    )
    outputs = generate_lge_bullseye(config)

    assert outputs.reference_rois is None
    assert outputs.scar_mask is None
    assert outputs.scar_segments is None
    assert outputs.landmarks.name == "032602_lge_landmarks.json"
    assert outputs.scar_figure.name == "032602_lge_scar_bullseye.png"
    assert outputs.landmarks.is_file()
    assert outputs.scar_figure.stat().st_size > 0
    assert {path.name for path in config.case.output_directory.iterdir()} == {
        "032602_lge_landmarks.json",
        "032602_lge_scar_bullseye.png",
    }


def test_external_scar_branch_leaves_preexisting_interactive_outputs_untouched(
    monkeypatch, tmp_path
):
    """Cleaning or regenerating interactive artifacts would corrupt these sentinels."""
    config = _write_external_scar_case(tmp_path, scar_foreground=True)
    config.case.output_directory.mkdir()
    sentinels = {}
    for basename in (
        "lge_intensity_bullseye.npz", "lge_normal_rois.json",
        "lge_scar_mask.nii.gz", "lge_scar_segments.csv", "unrelated.txt",
    ):
        path = config.case.output_path(basename)
        path.write_bytes(f"previous {basename}".encode())
        sentinels[path] = (path.read_bytes(), path.stat().st_mtime_ns)
    monkeypatch.setattr(
        "cardiac_bullseye.workflows.bullseye_frame.select_landmarks_interactively",
        _fake_fresh_landmarks,
    )
    generate_lge_bullseye(config)

    for path, (content, modified) in sentinels.items():
        assert path.read_bytes() == content
        assert path.stat().st_mtime_ns == modified


@pytest.mark.parametrize("problem, match", [
    ("grid", "scar mask spacing differ"), ("outside", "outside the myocardium"),
])
def test_external_scar_is_validated_before_landmark_annotation(
    monkeypatch, tmp_path, problem, match
):
    """Opening annotation before validating the external mask would hit the guard."""
    config = _write_external_scar_case(tmp_path, scar_foreground=True)
    path = config.lge_bullseye.lge_scar_mask
    image = sitk.ReadImage(str(path))
    if problem == "grid":
        image.SetSpacing((2., 1., 6.))
    else:
        array = sitk.GetArrayFromImage(image)
        array[0, 0, 0] = 1
        invalid = sitk.GetImageFromArray(array)
        invalid.CopyInformation(image)
        image = invalid
    sitk.WriteImage(image, str(path))
    original = path.read_bytes()
    for selector in ("select_landmarks_interactively", "select_single_reference_interactively"):
        monkeypatch.setattr(
            "cardiac_bullseye.workflows.bullseye_frame." + selector,
            lambda *_args, **_kwargs: pytest.fail("annotation must not open"),
        )

    with pytest.raises(ValueError, match=match):
        generate_lge_bullseye(config)

    assert path.read_bytes() == original
    assert not list(config.case.output_directory.glob("*"))


def test_external_scar_workflow_passes_saved_manual_centers(monkeypatch, tmp_path):
    from cardiac_bullseye.workflows import lge as module

    config = _write_external_scar_case(tmp_path, scar_foreground=True)
    manual_centers = [(14.0, 12.0), None, (10.0, 12.0)]

    def select_with_centers(
        study, output_path, *, resume_existing, cancel_is_error, transpose_for_display
    ):
        landmarks = [
            LandmarkPair(aip=(12.0, 5.0), iip=(12.0, 19.0))
        ] * study.slice_count
        save_landmarks_json(
            output_path,
            landmarks,
            image_shape=study.mr.shape,
            center_overrides=manual_centers,
        )
        return landmarks

    captured = {}
    real_compute = module.compute_lge_scar_summary

    def capture_centers(*args, **kwargs):
        captured["centers"] = kwargs.get("center_overrides")
        return real_compute(*args, **kwargs)

    monkeypatch.setattr(
        "cardiac_bullseye.workflows.bullseye_frame.select_landmarks_interactively",
        select_with_centers,
    )
    monkeypatch.setattr(module, "compute_lge_scar_summary", capture_centers)

    generate_lge_bullseye(config)

    assert captured["centers"] == manual_centers


def test_external_scar_workflow_reuses_landmarks_without_ui(monkeypatch, tmp_path):
    config = _write_external_scar_case(tmp_path, scar_foreground=True)
    config = replace(
        config,
        lge_bullseye=replace(
            config.lge_bullseye,
            reuse_saved_points=True,
        ),
    )
    config.case.output_directory.mkdir(parents=True, exist_ok=True)
    landmark = LandmarkPair(aip=(12.0, 5.0), iip=(12.0, 19.0))
    save_landmarks_json(
        config.case.output_path("lge_landmarks.json"),
        [landmark] * 3,
        image_shape=(3, 25, 25),
    )
    monkeypatch.setattr(
        "cardiac_bullseye.workflows.bullseye_frame.select_landmarks_interactively",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("annotation UI must not open")
        ),
    )

    outputs = generate_lge_bullseye(config)

    assert outputs.scar_figure.is_file()


def test_external_scar_workflow_allows_an_empty_scar_mask(monkeypatch, tmp_path):
    """An external mask with no scar still produces the AHA-16 scar figure."""
    config = _write_external_scar_case(tmp_path, scar_foreground=False)
    monkeypatch.setattr(
        "cardiac_bullseye.workflows.bullseye_frame.select_landmarks_interactively",
        _fake_fresh_landmarks,
    )

    outputs = generate_lge_bullseye(config)

    assert {path.name for path in config.case.output_directory.iterdir()} == {
        outputs.landmarks.name,
        outputs.scar_figure.name,
    }
    assert outputs.scar_figure.stat().st_size > 0


def test_external_scar_figure_contains_only_aha16_data_and_rendered_labels(
    monkeypatch, tmp_path
):
    """A center cap or its N/A label would reintroduce segment 17 externally."""
    from cardiac_bullseye.workflows import lge_common

    config = _write_external_scar_case(tmp_path, scar_foreground=True)
    monkeypatch.setattr(
        "cardiac_bullseye.workflows.bullseye_frame.select_landmarks_interactively",
        _fake_fresh_landmarks,
    )
    plot = lge_common.plot_lge_scar_bullseye
    rendered = []

    def capture_real_plot(summary, *args, **kwargs):
        figure, axis = plot(summary, *args, **kwargs)
        figure.canvas.draw()
        rendered.append(axis)
        return figure, axis

    monkeypatch.setattr(lge_common, "plot_lge_scar_bullseye", capture_real_plot)

    outputs = generate_lge_bullseye(config)

    axis = rendered[0]
    assert {text.get_text().splitlines()[0] for text in axis.texts} == {
        str(segment) for segment in range(1, 17)
    }
    assert all("N/A" not in text.get_text() for text in axis.texts)
    assert len(axis.patches) == 16  # No colored or white segment-17 cap.
    assert outputs.scar_figure.stat().st_size > 0


@pytest.mark.parametrize("mode", ["per-slice", "single-reference"])
def test_external_scar_landmark_cancel_is_an_error(
    monkeypatch, tmp_path, mode
):
    """Rendering before cancellation would overwrite or create a figure."""
    config = _write_external_scar_case(tmp_path, scar_foreground=True)
    config.case.output_directory.mkdir()
    intensity_path = config.case.output_path("lge_intensity_bullseye.png")
    scar_path = config.case.output_path("lge_scar_bullseye.png")
    intensity_path.write_bytes(b"previous intensity figure")

    def cancel_annotation(*_args, **kwargs):
        assert kwargs["cancel_is_error"] is True
        raise RuntimeError("AIP/IIP annotation was cancelled")

    selector = ("select_landmarks_interactively" if mode == "per-slice"
                else "select_single_reference_interactively")
    monkeypatch.setattr(
        "cardiac_bullseye.workflows.bullseye_frame." + selector,
        cancel_annotation,
    )

    with pytest.raises(RuntimeError, match=r"^AIP/IIP annotation was cancelled$"):
        generate_lge_bullseye(config, landmark_mode=mode)

    assert intensity_path.read_bytes() == b"previous intensity figure"
    assert not scar_path.exists()


SHARED_PAIR = LandmarkPair(aip=(12., 7.), iip=(12., 21.))


def _write_shared_external_case(tmp_path, *, slice_count=5):
    """Padding and unequal in-plane dimensions expose frame/cropping errors."""
    y, x = np.ogrid[:25, :29]
    ring = ((x - 14) ** 2 + (y - 12) ** 2 >= 25) & (
        (x - 14) ** 2 + (y - 12) ** 2 <= 81
    )
    mask = np.zeros((slice_count + 2, 25, 29), dtype=np.uint8)
    mask[1:-1] = ring
    lge = np.zeros(mask.shape, dtype=np.float32)
    scar = np.zeros_like(mask)
    for index in range(slice_count):
        lge[index + 1, ring] = (index + 1) * 10.
        lge[index + 1, 12, 22] += 100.
        scar[index + 1, 12, 22] = 1
    for name, array in (("lge.nii.gz", lge), ("mask.nii.gz", mask),
                        ("scar.nii.gz", scar)):
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
lge_scar_mask = "scar.nii.gz"
landmarks_path = "shared.json"
reuse_saved_points = true
angle_count = 24
radial_count = 13
angular_smoothing_sigma_deg = 0.0
radial_smoothing_sigma = 0.0
six_segment_warp = false
''', encoding="utf-8")
    return load_config(config_path, "lge-bullseye"), ring.T


def _no_annotation(*_args, **_kwargs):
    pytest.fail("shared reuse must not open annotation")


def test_external_lge_scar_overlay_projects_coronaries_on_cropped_geometry(
    monkeypatch, tmp_path
):
    from cardiac_bullseye.workflows import lge as module

    config, _ring = _write_shared_external_case(tmp_path)
    save_landmarks_json(
        config.lge_bullseye.landmarks_path,
        [SHARED_PAIR] * 5,
        image_shape=(5, 29, 25),
    )
    geometry_path = tmp_path / "geometry.json"
    coronary_path = tmp_path / "coronary.nii.gz"
    geometry_path.touch()
    coronary_path.touch()
    config = replace(
        config,
        bullseye=replace(
            config.bullseye,
            intensity_coronary_overlay=True,
            geometry_json=geometry_path,
            coronary_mask=coronary_path,
            coronary_label_colors={2: "#123456"},
            coronary_label_names={2: "Coronary LCX"},
            coronary_linewidth=2.7,
            show_final_figure=False,
        ),
    )
    sampling_affine = np.diag([1.0, 1.0, 6.0, 1.0])
    geometry = CardiacGeometry(
        true_apex_world=np.asarray([0.0, 0.0, 0.0]),
        base_center_world=np.asarray([0.0, 0.0, 36.0]),
        long_axis=np.asarray([0.0, 0.0, 1.0]),
        short_axis_x=np.asarray([1.0, 0.0, 0.0]),
        short_axis_y=np.asarray([0.0, 1.0, 0.0]),
        short_axis_affine=sampling_affine,
        short_axis_shape=(29, 25, 7),
        slice_positions_mm=np.arange(7, dtype=float) * 6.0,
        base_height_mm=36.0,
        short_axis_sampling_affine=sampling_affine,
    )
    projected = [
        ProjectedCoronaryPath(
            label=2,
            xy=np.asarray([[1.0, 0.0], [2.0, 0.5]], dtype=float),
        )
    ]
    calls = {}
    monkeypatch.setattr(module, "load_cardiac_geometry", lambda path: geometry)
    monkeypatch.setattr(
        module,
        "validate_geometry_against_short_axis_nifti",
        lambda selected, path: calls.update(validated=(selected, path)),
    )
    monkeypatch.setattr(
        module,
        "load_coronary_paths_from_nifti",
        lambda path, **kwargs: calls.update(loaded=(path, kwargs)) or [object()],
    )

    def project(paths, cropped_geometry, study, landmarks, **kwargs):
        calls["projection"] = (paths, cropped_geometry, study, landmarks, kwargs)
        return projected

    monkeypatch.setattr(module, "project_coronary_paths", project)
    monkeypatch.setattr(
        module,
        "save_lge_scar_figure",
        lambda summary, path, **kwargs: calls.update(plot=(summary, path, kwargs)) or path,
    )

    generate_lge_bullseye(config, projection="0-4")

    assert calls["validated"] == (geometry, config.lge_bullseye.lge)
    cropped = calls["projection"][1]
    assert cropped.short_axis_shape == (29, 25, 5)
    np.testing.assert_allclose(cropped.slice_positions_mm, [6, 12, 18, 24, 30])
    np.testing.assert_allclose(cropped.short_axis_sampling_affine[:3, 3], [0, 0, 6])
    np.testing.assert_allclose(
        calls["projection"][4]["radial_frame"].source_radii,
        [0, 1, 2, 3, 4],
    )
    assert calls["plot"][2]["coronary_paths"] == projected
    assert calls["plot"][2]["coronary_label_colors"] == {2: "#123456"}
    assert calls["plot"][2]["coronary_label_names"] == {2: "Coronary LCX"}
    assert calls["plot"][2]["coronary_linewidth"] == pytest.approx(2.7)


@pytest.mark.parametrize("mode", ["per-slice", "single-reference"])
@pytest.mark.parametrize("projection, radii, segment_count", [
    ("1-4", [1., 1.75, 2.5, 3.25, 4.], 16),
    ("0-4", [0., 1., 2., 3., 4.], 17),
])
def test_external_scar_branch_forwards_shared_frame(
    monkeypatch, tmp_path, mode, projection, radii, segment_count
):
    """A private landmark path or separate scar radius must break this test."""
    from cardiac_bullseye.workflows import lge as module, lge_common

    config, ring = _write_shared_external_case(tmp_path)
    path = config.lge_bullseye.landmarks_path
    centers = [(12., 14.)] * 5
    save_landmarks_json(path, [SHARED_PAIR] * 5, image_shape=(5, 29, 25),
                        center_overrides=centers)
    saved = path.read_bytes()
    for selector in ("select_landmarks_interactively", "select_single_reference_interactively"):
        monkeypatch.setattr("cardiac_bullseye.workflows.bullseye_frame." + selector,
                            _no_annotation)
    captured = {}
    real_compute = module.compute_lge_scar_summary

    def compute(study, landmarks, *args, **kwargs):
        np.testing.assert_allclose(study.slice_positions, [0., 6., 12., 18., 24.])
        np.testing.assert_allclose(kwargs["source_radii"], radii)
        assert kwargs["minimum_radius"] == radii[0]
        assert kwargs["center_overrides"] == centers
        assert kwargs["deleted_slice_indices"] == set()
        summary = real_compute(study, landmarks, *args, **kwargs)
        captured["summary"] = summary
        return summary

    def capture_plot(name, real_plot):
        def plot(*args, **kwargs):
            figure, axis = real_plot(*args, **kwargs)
            captured[name] = axis
            return figure, axis
        return plot

    monkeypatch.setattr(module, "compute_lge_scar_summary", compute)
    monkeypatch.setattr(lge_common, "plot_lge_scar_bullseye",
                        capture_plot("scar_axis", lge_common.plot_lge_scar_bullseye))

    outputs = generate_lge_bullseye(config, landmark_mode=mode, projection=projection)

    assert outputs.landmarks == path
    assert path.read_bytes() == saved
    assert outputs.reference_rois is None
    assert outputs.scar_mask is None
    assert outputs.scar_segments is None
    assert [field.name for field in fields(outputs)] == [
        "landmarks", "reference_rois", "scar_mask", "scar_segments", "scar_figure",
    ]
    assert {file.name for file in config.case.output_directory.iterdir()} == {
        "shared_lge_scar_bullseye.png"
    }
    summary = captured["summary"]
    assert len(summary.scar_percentages) == segment_count
    assert summary.scar_voxel_counts.sum() == 5
    assert summary.myocardial_voxel_counts.sum() == 5 * ring.sum()
    scar_axis = captured["scar_axis"]
    assert {text.get_text().splitlines()[0] for text in scar_axis.texts} == {
        str(segment) for segment in range(1, segment_count + 1)
    }
    if projection == "0-4":
        expected = 100. / ring.sum()
        assert summary.scar_voxel_counts[16] == 1
        assert summary.myocardial_voxel_counts[16] == ring.sum()
        assert summary.scar_percentages[16] == pytest.approx(expected)
        center = [text.get_text() for text in scar_axis.texts
                  if text.get_position() == (0., 0.)]
        assert center == [f"17\n{expected:.1f}%"]
        assert scar_axis.patches[-1].get_facecolor() != (1., 1., 1., 1.)
    else:
        assert len(scar_axis.patches) == 16


@pytest.mark.parametrize("reuse, mode", [(True, "single-reference"), (False, "per-slice")])
def test_external_missing_shared_fallback_and_fresh_overwrite(
    monkeypatch, tmp_path, reuse, mode
):
    """Ignoring missing reuse or resuming old points would retain stale content."""
    from cardiac_bullseye.workflows import lge as module

    config, ring = _write_shared_external_case(tmp_path)
    config = replace(config, lge_bullseye=replace(
        config.lge_bullseye, reuse_saved_points=reuse))
    path = config.lge_bullseye.landmarks_path
    if not reuse:
        path.write_text("stale points", encoding="utf-8")
    centers = [None, (13., 14.), None, None, None]

    def select(study, output_path, *, resume_existing, transpose_for_display,
               cancel_is_error):
        assert output_path == path
        assert resume_existing is False
        assert transpose_for_display is True
        assert cancel_is_error is True
        points = [SHARED_PAIR, None, SHARED_PAIR, SHARED_PAIR, None]
        save_landmarks_json(path, points, image_shape=study.mr.shape,
                            deleted_slice_indices={4}, center_overrides=centers)
        return points

    for selector in ("select_landmarks_interactively", "select_single_reference_interactively"):
        monkeypatch.setattr("cardiac_bullseye.workflows.bullseye_frame." + selector,
                            _no_annotation)
    selector = ("select_single_reference_interactively" if reuse
                else "select_landmarks_interactively")
    monkeypatch.setattr("cardiac_bullseye.workflows.bullseye_frame." + selector, select)
    real_compute = module.compute_lge_scar_summary
    summaries = []

    def compute(*args, **kwargs):
        assert kwargs["center_overrides"] == centers
        assert kwargs["deleted_slice_indices"] == {4}
        # Deleted basal positions still have extrapolated radii, but are excluded.
        np.testing.assert_allclose(kwargs["source_radii"], [1., 2., 3., 4., 5.])
        summary = real_compute(*args, **kwargs)
        summaries.append(summary)
        return summary

    monkeypatch.setattr(module, "compute_lge_scar_summary", compute)
    assert generate_lge_bullseye(config, landmark_mode=mode).landmarks == path
    assert json.loads(path.read_text())["slices"][0]["aip"] == [12., 7.]
    assert summaries[0].scar_voxel_counts.sum() == 4
    assert summaries[0].myocardial_voxel_counts.sum() == 4 * ring.sum()


@pytest.mark.parametrize("explicit", [False, True])
def test_external_legacy_lge_coordinates_migrate_only_at_default_path(
    monkeypatch, tmp_path, explicit
):
    from cardiac_bullseye.workflows import lge as module

    config, _ring = _write_shared_external_case(tmp_path)
    path = config.case.output_path("lge_landmarks.json")
    path.parent.mkdir()
    path.write_text(json.dumps({"image_shape": [5, 25, 29], "slices": [
        {"aip": [7., 12.], "iip": [21., 12.], "center": [14., 12.]}
    ] * 5}))
    config = replace(config, lge_bullseye=replace(
        config.lge_bullseye, landmarks_path=path if explicit else None))
    real_compute = module.compute_lge_scar_summary

    def compute(study, landmarks, *args, **kwargs):
        assert landmarks == [SHARED_PAIR] * 5
        assert kwargs["center_overrides"] == [(12., 14.)] * 5
        return real_compute(study, landmarks, *args, **kwargs)

    monkeypatch.setattr(module, "compute_lge_scar_summary", compute)
    if explicit:
        with pytest.raises(ValueError, match="Ambiguous unversioned"):
            generate_lge_bullseye(config)
    else:
        assert generate_lge_bullseye(config).landmarks == path


@pytest.mark.parametrize("problem, match", [
    ("shape", "image shape"), ("version", "Unsupported landmark coordinate version"),
    ("one-slice", "at least two retained slices"), ("deletion", "contiguous basal")
])
def test_external_invalid_shared_frame_fails_before_mapping(
    monkeypatch, tmp_path, problem, match
):
    config, _ring = _write_shared_external_case(
        tmp_path, slice_count=1 if problem == "one-slice" else 5)
    count = 1 if problem == "one-slice" else 5
    path = config.lge_bullseye.landmarks_path
    save_landmarks_json(path, [SHARED_PAIR] * count, image_shape=(count, 29, 25))
    payload = json.loads(path.read_text())
    if problem == "shape":
        payload["image_shape"] = [5, 25, 29]
    elif problem == "version":
        payload["coordinate_version"] = 99
    elif problem == "deletion":
        payload["slices"][2] = {"status": "deleted"}
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(
        "cardiac_bullseye.workflows.lge.compute_lge_scar_summary",
        lambda *_args, **_kwargs: pytest.fail("mapping must not start"))
    with pytest.raises(ValueError, match=match):
        generate_lge_bullseye(config, projection="0-4")
