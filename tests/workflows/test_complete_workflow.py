def test_complete_passes_short_axis_outputs_to_bullseye(
    monkeypatch, app_config, tmp_path
):
    from cardiac_bullseye.workflows import complete as module
    from cardiac_bullseye.workflows.bullseye import BullseyeOutputs
    from cardiac_bullseye.workflows.short_axis import ShortAxisOutputs

    short = ShortAxisOutputs(
        tmp_path / "sax.nii.gz",
        tmp_path / "mask.nii.gz",
        tmp_path / "geometry.json",
    )
    final = BullseyeOutputs(
        tmp_path / "b.png",
        tmp_path / "d.png",
        tmp_path / "w.nii.gz",
        tmp_path / "s.nii.gz",
    )
    received = []
    monkeypatch.setattr(module, "generate_short_axis", lambda config: short)
    monkeypatch.setattr(
        module,
        "generate_bullseye",
        lambda config, outputs: received.append(outputs) or final,
    )
    assert module.run_complete_pipeline(app_config) == final
    assert received == [short]
