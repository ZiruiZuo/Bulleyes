from __future__ import annotations

import csv
import json
from dataclasses import replace

import numpy as np
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

    def select_fresh_landmarks(study, output_path, *, resume_existing):
        assert resume_existing is False
        landmarks = [landmark] * study.slice_count
        save_landmarks_json(output_path, landmarks, image_shape=study.mr.shape)
        return landmarks

    monkeypatch.setattr(
        "cardiac_bullseye.workflows.lge.select_landmarks_interactively",
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
    monkeypatch.setattr(
        "cardiac_bullseye.workflows.lge.select_landmarks_interactively",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("annotation UI must not open")
        ),
    )
    reused_outputs = generate_lge_bullseye(reused_config)
    assert reused_outputs.figure.is_file()
