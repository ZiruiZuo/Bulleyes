from pathlib import Path


def test_generate_short_axis_returns_saved_paths(monkeypatch, app_config, tmp_path):
    from cardiac_bullseye.workflows import short_axis as module

    def fake_interactive_convert_to_short_axis(**kwargs):
        for key in ("output_path", "mask_output_path", "geometry_output_path"):
            Path(kwargs[key]).touch()
        return object()

    monkeypatch.setattr(
        module,
        "interactive_convert_to_short_axis",
        fake_interactive_convert_to_short_axis,
    )
    outputs = module.generate_short_axis(app_config)
    assert outputs.mr.name == "patient_043_short_axis.nii.gz"
    assert outputs.mask.name == "patient_043_short_axis_mask.nii.gz"
    assert outputs.geometry.name == "patient_043_geometry.json"
