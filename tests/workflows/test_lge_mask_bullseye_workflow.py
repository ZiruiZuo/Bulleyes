from __future__ import annotations

import matplotlib
from dataclasses import replace

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


def _fake_fresh_landmarks(study, output_path, *, resume_existing, cancel_is_error):
    assert resume_existing is False
    assert cancel_is_error is True
    landmarks = [LandmarkPair(aip=(12.0, 5.0), iip=(12.0, 19.0))] * study.slice_count
    save_landmarks_json(output_path, landmarks)
    return landmarks


def _forbidden(*_args, **_kwargs):
    raise AssertionError("external-scar workflow must not select ROIs or compute scar")


def test_external_scar_workflow_writes_only_three_outputs(monkeypatch, tmp_path):
    """Replacing this with the ROI/scar workflow must fail this test."""
    config = _write_external_scar_case(tmp_path, scar_foreground=True)
    monkeypatch.setattr(
        "cardiac_bullseye.workflows.lge_mask_bullseye.select_landmarks_interactively",
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
        study, output_path, *, resume_existing, cancel_is_error
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

    monkeypatch.setattr(module, "select_landmarks_interactively", select_with_centers)
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
        "cardiac_bullseye.workflows.lge_mask_bullseye.select_landmarks_interactively",
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
        "cardiac_bullseye.workflows.lge_mask_bullseye.select_landmarks_interactively",
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
        "cardiac_bullseye.workflows.lge_mask_bullseye.select_landmarks_interactively",
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
        "cardiac_bullseye.workflows.lge_mask_bullseye.select_landmarks_interactively",
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


def test_external_workflow_cancellation_preserves_existing_and_missing_pngs(
    monkeypatch, tmp_path
):
    """Rendering before cancellation would overwrite or create a figure."""
    config = _write_external_scar_case(tmp_path, scar_foreground=True)
    config.case.output_directory.mkdir()
    intensity_path = config.case.output_path("lge_intensity_bullseye.png")
    scar_path = config.case.output_path("lge_scar_bullseye.png")
    intensity_path.write_bytes(b"previous intensity figure")

    def cancel_annotation(*_args, **_kwargs):
        raise RuntimeError("AIP/IIP annotation was cancelled")

    monkeypatch.setattr(
        "cardiac_bullseye.workflows.lge_mask_bullseye.select_landmarks_interactively",
        cancel_annotation,
    )

    with pytest.raises(RuntimeError, match=r"^AIP/IIP annotation was cancelled$"):
        generate_lge_mask_bullseye(config)

    assert intensity_path.read_bytes() == b"previous intensity figure"
    assert not scar_path.exists()
