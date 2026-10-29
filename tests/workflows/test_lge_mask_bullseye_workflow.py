from __future__ import annotations

import matplotlib
from dataclasses import fields, replace
import json

matplotlib.use("Agg")

import numpy as np
import pytest
import SimpleITK as sitk

from cardiac_bullseye.config import load_config
from cardiac_bullseye.territory.io import save_landmarks_json
from cardiac_bullseye.territory.models import LandmarkPair
from cardiac_bullseye.workflows import generate_lge_mask_bullseye


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
[lge_mask_bullseye]
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
    return load_config(config_path, "lge-mask-bullseye")


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


def test_external_scar_workflow_writes_only_three_outputs(monkeypatch, tmp_path):
    """Replacing this with the ROI/scar workflow must fail this test."""
    config = _write_external_scar_case(tmp_path, scar_foreground=True)
    monkeypatch.setattr(
        "cardiac_bullseye.workflows.bullseye_frame.select_landmarks_interactively",
        _fake_fresh_landmarks,
    )
    monkeypatch.setattr(
        "cardiac_bullseye.workflows.lge.select_reference_rois_interactively",
        _forbidden,
    )
    monkeypatch.setattr(
        "cardiac_bullseye.workflows.lge.compute_lge_scar",
        _forbidden,
    )

    outputs = generate_lge_mask_bullseye(config)

    assert outputs.landmarks.name == "032602_lge_landmarks.json"
    assert outputs.intensity_figure.name == "032602_lge_intensity_bullseye.png"
    assert outputs.scar_figure.name == "032602_lge_scar_bullseye.png"
    assert outputs.landmarks.is_file()
    assert outputs.intensity_figure.stat().st_size > 0
    assert outputs.scar_figure.stat().st_size > 0
    assert {path.name for path in config.case.output_directory.iterdir()} == {
        "032602_lge_landmarks.json",
        "032602_lge_intensity_bullseye.png",
        "032602_lge_scar_bullseye.png",
    }


def test_external_scar_workflow_passes_saved_manual_centers(monkeypatch, tmp_path):
    from cardiac_bullseye.workflows import lge_mask_bullseye as module

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
    real_compute = module.compute_lge_bullseye_products

    def capture_centers(*args, **kwargs):
        captured["centers"] = kwargs.get("center_overrides")
        return real_compute(*args, **kwargs)

    monkeypatch.setattr(
        "cardiac_bullseye.workflows.bullseye_frame.select_landmarks_interactively",
        select_with_centers,
    )
    monkeypatch.setattr(module, "compute_lge_bullseye_products", capture_centers)

    generate_lge_mask_bullseye(config)

    assert captured["centers"] == manual_centers


def test_external_scar_workflow_reuses_landmarks_without_ui(monkeypatch, tmp_path):
    config = _write_external_scar_case(tmp_path, scar_foreground=True)
    config = replace(
        config,
        lge_mask_bullseye=replace(
            config.lge_mask_bullseye,
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

    outputs = generate_lge_mask_bullseye(config)

    assert outputs.intensity_figure.is_file()
    assert outputs.scar_figure.is_file()


def test_external_scar_workflow_allows_an_empty_scar_mask(monkeypatch, tmp_path):
    """An external mask with no scar still produces the two AHA-16 figures."""
    config = _write_external_scar_case(tmp_path, scar_foreground=False)
    monkeypatch.setattr(
        "cardiac_bullseye.workflows.bullseye_frame.select_landmarks_interactively",
        _fake_fresh_landmarks,
    )

    outputs = generate_lge_mask_bullseye(config)

    assert {path.name for path in config.case.output_directory.iterdir()} == {
        outputs.landmarks.name,
        outputs.intensity_figure.name,
        outputs.scar_figure.name,
    }
    assert outputs.intensity_figure.stat().st_size > 0
    assert outputs.scar_figure.stat().st_size > 0


def test_external_intensity_contains_only_aha16_data_and_rendered_labels(
    monkeypatch, tmp_path
):
    """Computing or drawing the apex cap would introduce segment 17 again."""
    from cardiac_bullseye.workflows import lge_common

    config = _write_external_scar_case(tmp_path, scar_foreground=True)
    monkeypatch.setattr(
        "cardiac_bullseye.workflows.bullseye_frame.select_landmarks_interactively",
        _fake_fresh_landmarks,
    )
    plot = lge_common.plot_lge_bullseye
    rendered = []

    def capture_real_plot(result, *args, **kwargs):
        figure, axis = plot(result, *args, **kwargs)
        figure.canvas.draw()
        rendered.append((result, axis))
        return figure, axis

    monkeypatch.setattr(lge_common, "plot_lge_bullseye", capture_real_plot)

    outputs = generate_lge_mask_bullseye(config)

    result, axis = rendered[0]
    assert {text.get_text() for text in axis.texts} == {
        str(segment) for segment in range(1, 17)
    }
    assert np.isnan(result.apex_intensity)
    assert len(axis.patches) == 0  # No colored segment-17 cap.
    np.testing.assert_allclose(
        axis.collections[0].get_array().reshape(9, 24), result.intensity
    )
    assert outputs.intensity_figure.stat().st_size > 0


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

    outputs = generate_lge_mask_bullseye(config)

    axis = rendered[0]
    assert {text.get_text().splitlines()[0] for text in axis.texts} == {
        str(segment) for segment in range(1, 17)
    }
    assert all("N/A" not in text.get_text() for text in axis.texts)
    assert len(axis.patches) == 16  # No colored or white segment-17 cap.
    assert outputs.scar_figure.stat().st_size > 0


@pytest.mark.parametrize("mode", ["per-slice", "single-reference"])
def test_external_workflow_cancellation_preserves_existing_and_missing_pngs(
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
        generate_lge_mask_bullseye(config, landmark_mode=mode)

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
[lge_mask_bullseye]
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
    return load_config(config_path, "lge-mask-bullseye"), ring.T


def _no_annotation(*_args, **_kwargs):
    pytest.fail("shared reuse must not open annotation")


@pytest.mark.parametrize("mode", ["per-slice", "single-reference"])
@pytest.mark.parametrize("projection, radii, segment_count", [
    ("1-4", [1., 1.75, 2.5, 3.25, 4.], 16),
    ("0-4", [0., 1., 2., 3., 4.], 17),
])
def test_external_shared_frame_and_segment17_presentation(
    monkeypatch, tmp_path, mode, projection, radii, segment_count
):
    """A private landmark path or separate scar radius must break this test."""
    from cardiac_bullseye.workflows import lge_mask_bullseye as module, lge_common

    config, ring = _write_shared_external_case(tmp_path)
    path = config.lge_mask_bullseye.landmarks_path
    centers = [(12., 14.)] * 5
    save_landmarks_json(path, [SHARED_PAIR] * 5, image_shape=(5, 29, 25),
                        center_overrides=centers)
    saved = path.read_bytes()
    for selector in ("select_landmarks_interactively", "select_single_reference_interactively"):
        monkeypatch.setattr("cardiac_bullseye.workflows.bullseye_frame." + selector,
                            _no_annotation)
    captured = {}
    real_compute = module.compute_lge_bullseye_products

    def compute(study, landmarks, *args, **kwargs):
        np.testing.assert_allclose(study.slice_positions, [0., 6., 12., 18., 24.])
        np.testing.assert_allclose(kwargs["source_radii"], radii)
        assert kwargs["minimum_radius"] == radii[0]
        assert kwargs["center_overrides"] == centers
        products = real_compute(study, landmarks, *args, **kwargs)
        captured["products"] = products
        return products

    def capture_plot(name, real_plot):
        def plot(*args, **kwargs):
            figure, axis = real_plot(*args, **kwargs)
            captured[name] = axis
            return figure, axis
        return plot

    monkeypatch.setattr(module, "compute_lge_bullseye_products", compute)
    monkeypatch.setattr(lge_common, "plot_lge_bullseye",
                        capture_plot("intensity_axis", lge_common.plot_lge_bullseye))
    monkeypatch.setattr(lge_common, "plot_lge_scar_bullseye",
                        capture_plot("scar_axis", lge_common.plot_lge_scar_bullseye))

    outputs = generate_lge_mask_bullseye(config, landmark_mode=mode, projection=projection)

    assert outputs.landmarks == path
    assert path.read_bytes() == saved
    assert {field.name for field in fields(outputs)} == {
        "landmarks", "intensity_figure", "scar_figure"
    }
    assert {file.name for file in config.case.output_directory.iterdir()} == {
        "shared_lge_intensity_bullseye.png", "shared_lge_scar_bullseye.png"
    }
    products = captured["products"]
    assert len(products.scar.scar_percentages) == segment_count
    assert products.scar.scar_voxel_counts.sum() == 5
    assert products.scar.myocardial_voxel_counts.sum() == 5 * ring.sum()
    scar_axis = captured["scar_axis"]
    intensity_axis = captured["intensity_axis"]
    assert {text.get_text().splitlines()[0] for text in scar_axis.texts} == {
        str(segment) for segment in range(1, segment_count + 1)
    }
    if projection == "0-4":
        expected = 100. / ring.sum()
        assert products.scar.scar_voxel_counts[16] == 1
        assert products.scar.myocardial_voxel_counts[16] == ring.sum()
        assert products.scar.scar_percentages[16] == pytest.approx(expected)
        center = [text.get_text() for text in scar_axis.texts
                  if text.get_position() == (0., 0.)]
        assert center == [f"17\n{expected:.1f}%"]
        assert scar_axis.patches[-1].get_facecolor() != (1., 1., 1., 1.)
        values = products.intensity.intensity
        np.testing.assert_allclose(values[0], values[0, 0])
        assert np.ptp(values[3]) > 0.
        assert np.ptp(values[1]) == pytest.approx(np.ptp(values[3]) * 7. / 27.)
        assert np.ptp(values[2]) == pytest.approx(np.ptp(values[3]) * 20. / 27.)
        assert not intensity_axis.patches  # Continuous intensity reaches the origin.
    else:
        assert np.isnan(products.intensity.apex_intensity)
        assert len(scar_axis.patches) == 16
        assert not intensity_axis.patches
        assert {text.get_text() for text in intensity_axis.texts} == {
            str(segment) for segment in range(1, 17)
        }
        assert products.intensity.radial_positions[0] == 1.


@pytest.mark.parametrize("reuse, mode", [(True, "single-reference"), (False, "per-slice")])
def test_external_missing_shared_fallback_and_fresh_overwrite(
    monkeypatch, tmp_path, reuse, mode
):
    """Ignoring missing reuse or resuming old points would retain stale content."""
    from cardiac_bullseye.workflows import lge_mask_bullseye as module

    config, ring = _write_shared_external_case(tmp_path)
    config = replace(config, lge_mask_bullseye=replace(
        config.lge_mask_bullseye, reuse_saved_points=reuse))
    path = config.lge_mask_bullseye.landmarks_path
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
    real_compute = module.compute_lge_bullseye_products
    products = []

    def compute(*args, **kwargs):
        assert kwargs["center_overrides"] == centers
        assert kwargs["deleted_slice_indices"] == {4}
        # Deleted basal positions still have extrapolated radii, but are excluded.
        np.testing.assert_allclose(kwargs["source_radii"], [1., 2., 3., 4., 5.])
        product = real_compute(*args, **kwargs)
        products.append(product)
        return product

    monkeypatch.setattr(module, "compute_lge_bullseye_products", compute)
    assert generate_lge_mask_bullseye(config, landmark_mode=mode).landmarks == path
    assert json.loads(path.read_text())["slices"][0]["aip"] == [12., 7.]
    assert products[0].scar.scar_voxel_counts.sum() == 4
    assert products[0].scar.myocardial_voxel_counts.sum() == 4 * ring.sum()


@pytest.mark.parametrize("explicit", [False, True])
def test_external_legacy_lge_coordinates_migrate_only_at_default_path(
    monkeypatch, tmp_path, explicit
):
    from cardiac_bullseye.workflows import lge_mask_bullseye as module

    config, _ring = _write_shared_external_case(tmp_path)
    path = config.case.output_path("lge_landmarks.json")
    path.parent.mkdir()
    path.write_text(json.dumps({"image_shape": [5, 25, 29], "slices": [
        {"aip": [7., 12.], "iip": [21., 12.], "center": [14., 12.]}
    ] * 5}))
    config = replace(config, lge_mask_bullseye=replace(
        config.lge_mask_bullseye, landmarks_path=path if explicit else None))
    real_compute = module.compute_lge_bullseye_products

    def compute(study, landmarks, *args, **kwargs):
        assert landmarks == [SHARED_PAIR] * 5
        assert kwargs["center_overrides"] == [(12., 14.)] * 5
        return real_compute(study, landmarks, *args, **kwargs)

    monkeypatch.setattr(module, "compute_lge_bullseye_products", compute)
    if explicit:
        with pytest.raises(ValueError, match="Ambiguous unversioned"):
            generate_lge_mask_bullseye(config)
    else:
        assert generate_lge_mask_bullseye(config).landmarks == path


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
    path = config.lge_mask_bullseye.landmarks_path
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
        "cardiac_bullseye.workflows.lge_mask_bullseye.compute_lge_bullseye_products",
        lambda *_args, **_kwargs: pytest.fail("mapping must not start"))
    with pytest.raises(ValueError, match=match):
        generate_lge_mask_bullseye(config, projection="0-4")
