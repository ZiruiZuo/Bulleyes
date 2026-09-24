from pathlib import Path
from dataclasses import replace


def test_generate_short_axis_returns_saved_paths(monkeypatch, app_config, tmp_path):
    from cardiac_bullseye.workflows import short_axis as module

    def fake_interactive_convert_to_short_axis(**kwargs):
        for key in (
            "output_path",
            "mask_output_path",
            "geometry_output_path",
            "points_output_path",
        ):
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
    assert outputs.points.name == "patient_043_short_axis_points.json"


def test_generate_short_axis_passes_orientation_reference(monkeypatch, app_config, tmp_path):
    from dataclasses import replace
    from cardiac_bullseye.workflows import short_axis as module

    reference = tmp_path / "lge_sax.nii.gz"
    reference.touch()
    config = replace(
        app_config,
        short_axis=replace(app_config.short_axis, orientation_reference=reference),
    )
    received = {}

    def fake_interactive_convert_to_short_axis(**kwargs):
        received.update(kwargs)
        for key in (
            "output_path",
            "mask_output_path",
            "geometry_output_path",
            "points_output_path",
        ):
            Path(kwargs[key]).touch()
        return object()

    monkeypatch.setattr(module, "interactive_convert_to_short_axis", fake_interactive_convert_to_short_axis)

    module.generate_short_axis(config)

    assert received["orientation_reference_path"] == reference


def test_generate_short_axis_reuses_points_without_opening_planner(
    monkeypatch, app_config
):
    from cardiac_bullseye.workflows import short_axis as module

    config = replace(
        app_config,
        short_axis=replace(app_config.short_axis, reuse_saved_points=True),
    )
    points = config.case.output_path("short_axis_points.json")
    points.touch()
    received = {}

    monkeypatch.setattr(
        module,
        "interactive_convert_to_short_axis",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("short-axis planner must not open")
        ),
    )

    def fake_reuse(**kwargs):
        received.update(kwargs)
        for key in ("output_path", "mask_output_path", "geometry_output_path"):
            Path(kwargs[key]).touch()

    monkeypatch.setattr(module, "convert_to_short_axis_from_saved_points", fake_reuse)

    outputs = module.generate_short_axis(config)

    assert received["points_path"] == points
    assert outputs.points == points
