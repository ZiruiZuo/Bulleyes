from pathlib import Path

import pytest

from cardiac_bullseye.config import ConfigError, load_config


def write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "case.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_pipeline_config_parses_windows_paths_and_defaults(tmp_path):
    mr = tmp_path / "mr.nii.gz"
    mask = tmp_path / "mask.nii.gz"
    coronary = tmp_path / "coronary.nii.gz"
    myocardium = tmp_path / "myocardium.nii.gz"
    for path in (mr, mask, coronary, myocardium):
        path.touch()
    config_path = write_config(
        tmp_path,
        f"""
[case]
id = "patient_043"
output_directory = '{tmp_path.as_posix()}/results'
[short_axis]
source_mr = '{mr.as_posix()}'
source_mask = '{mask.as_posix()}'
[bullseye]
coronary_mask = '{coronary.as_posix()}'
world_myocardium_mask = '{myocardium.as_posix()}'
""",
    )
    config = load_config(config_path, "pipeline")
    assert config.case.id == "patient_043"
    assert config.short_axis.slice_spacing_mm == 8.0
    assert config.bullseye.angle_count == 360
    assert config.bullseye.coronary_label_names == {
        1: "Coronary LAD",
        2: "Coronary LCX",
        3: "Coronary RCA",
    }
    assert config.bullseye.distance_vessel_labels == (1, 2, 3)
    assert config.short_axis.reuse_saved_points is False
    assert config.bullseye.reuse_saved_points is False
    assert config.lge_bullseye.reuse_saved_points is False
    assert config.lge_mask_bullseye.reuse_saved_points is False
    assert config.case.output_path("bullseye.png").name == "patient_043_bullseye.png"


def test_reuse_saved_points_must_be_enabled_explicitly(tmp_path):
    source = tmp_path / "source.nii.gz"
    source.touch()
    config_path = write_config(
        tmp_path,
        f"""
[case]
id = "case"
[short_axis]
source_mr = '{source.as_posix()}'
reuse_saved_points = true
[bullseye]
reuse_saved_points = true
[lge_bullseye]
reuse_saved_points = true
[lge_mask_bullseye]
reuse_saved_points = true
""",
    )

    config = load_config(config_path, "short-axis")

    assert config.short_axis.reuse_saved_points is True
    assert config.bullseye.reuse_saved_points is True
    assert config.lge_bullseye.reuse_saved_points is True
    assert config.lge_mask_bullseye.reuse_saved_points is True


def test_short_axis_parses_optional_orientation_reference(tmp_path):
    source = tmp_path / "source.nii.gz"
    reference = tmp_path / "lge_sax.nii.gz"
    source.touch()
    reference.touch()
    config_path = write_config(
        tmp_path,
        f"""
[case]
id = "case"
[short_axis]
source_mr = '{source.as_posix()}'
orientation_reference = '{reference.as_posix()}'
""",
    )

    config = load_config(config_path, "short-axis")

    assert config.short_axis.orientation_reference == reference


def test_lge_bullseye_config_parses_inputs_and_mapping_parameters(tmp_path):
    """The LGE command must receive its complete input contract from TOML."""
    lge = tmp_path / "lge_sax.nii.gz"
    mask = tmp_path / "lge_mask.seg.nrrd"
    lge.touch()
    mask.touch()
    config_path = write_config(
        tmp_path,
        f"""
[case]
id = "020501"
output_directory = "results"
[lge_bullseye]
lge = '{lge.as_posix()}'
mask = '{mask.as_posix()}'
angle_count = 180
radial_count = 120
angular_smoothing_sigma_deg = 2.0
radial_smoothing_sigma = 0.5
six_segment_warp = false
scar_sd_multiplier = 5.5
normal_roi_min_voxels = 24
""",
    )

    config = load_config(config_path, "lge-bullseye")

    assert config.lge_bullseye.lge == lge
    assert config.lge_bullseye.mask == mask
    assert config.lge_bullseye.angle_count == 180
    assert config.lge_bullseye.radial_count == 120
    assert config.lge_bullseye.angular_smoothing_sigma_deg == 2.0
    assert config.lge_bullseye.radial_smoothing_sigma == 0.5
    assert config.lge_bullseye.six_segment_warp is False
    assert config.lge_bullseye.scar_sd_multiplier == 5.5
    assert config.lge_bullseye.normal_roi_min_voxels == 24
    assert config.case.output_directory == tmp_path / "results"


def test_lge_mask_bullseye_config_parses_three_inputs_and_defaults(tmp_path):
    lge = tmp_path / "lge_sax.nii.gz"
    mask = tmp_path / "myocardium_mask.nii.gz"
    scar = tmp_path / "scar_mask.nii.gz"
    for path in (lge, mask, scar):
        path.touch()
    config_path = write_config(
        tmp_path,
        f"""
[case]
id = "032602"
output_directory = "results"
[lge_mask_bullseye]
lge = '{lge.as_posix()}'
mask = '{mask.as_posix()}'
lge_scar_mask = '{scar.as_posix()}'
""",
    )

    config = load_config(config_path, "lge-mask-bullseye")

    assert config.lge_mask_bullseye.lge == lge
    assert config.lge_mask_bullseye.mask == mask
    assert config.lge_mask_bullseye.lge_scar_mask == scar
    assert config.lge_mask_bullseye.angle_count == 360
    assert config.lge_mask_bullseye.radial_count == 300
    assert config.lge_mask_bullseye.six_segment_warp is True
    assert config.case.output_directory == tmp_path / "results"


def test_lge_mask_bullseye_reports_all_missing_inputs(tmp_path):
    config_path = write_config(
        tmp_path,
        """
[case]
id = "032602"
[lge_mask_bullseye]
lge = "missing_lge.nii.gz"
mask = "missing_myocardium.nii.gz"
lge_scar_mask = "missing_scar.nii.gz"
""",
    )

    with pytest.raises(ConfigError) as caught:
        load_config(config_path, "lge-mask-bullseye")

    message = str(caught.value)
    assert "lge_mask_bullseye.lge" in message
    assert "lge_mask_bullseye.mask" in message
    assert "lge_mask_bullseye.lge_scar_mask" in message


@pytest.mark.parametrize(
    ("section", "workflow", "requires_scar"),
    [
        ("lge_bullseye", "lge-bullseye", False),
        ("lge_mask_bullseye", "lge-mask-bullseye", True),
    ],
)
def test_lge_sections_reject_mapping_grids_too_small_for_annotation(
    tmp_path, section, workflow, requires_scar
):
    """Allowing either value would defer a known invalid grid until after UI."""
    lge = tmp_path / "lge.nii.gz"
    mask = tmp_path / "mask.nrrd"
    scar = tmp_path / "scar.nii.gz"
    for path in (lge, mask, scar):
        path.touch()
    scar_input = f"lge_scar_mask = '{scar.as_posix()}'" if requires_scar else ""
    config_path = write_config(
        tmp_path,
        f"""
[case]
id = "case"
[{section}]
lge = '{lge.as_posix()}'
mask = '{mask.as_posix()}'
    {scar_input}
angle_count = 3
radial_count = 2
""",
    )

    with pytest.raises(ConfigError) as caught:
        load_config(config_path, workflow)

    message = str(caught.value)
    assert f"{section}.angle_count" in message
    assert f"{section}.radial_count" in message


@pytest.mark.parametrize(
    ("field", "value"),
    [("scar_sd_multiplier", "0.0"), ("normal_roi_min_voxels", "1")],
)
def test_lge_bullseye_rejects_invalid_scar_parameters(tmp_path, field, value):
    lge = tmp_path / "lge.nii.gz"
    mask = tmp_path / "mask.nrrd"
    lge.touch()
    mask.touch()
    config_path = write_config(
        tmp_path,
        f"""
[case]
id = "case"
[lge_bullseye]
lge = '{lge.as_posix()}'
mask = '{mask.as_posix()}'
{field} = {value}
""",
    )

    with pytest.raises(ConfigError, match=field):
        load_config(config_path, "lge-bullseye")


def test_bullseye_validation_aggregates_all_missing_inputs(tmp_path):
    config_path = write_config(
        tmp_path,
        f"""
[case]
id = "case"
output_directory = '{tmp_path.as_posix()}/results'
[short_axis]
source_mr = '{tmp_path.as_posix()}/unused.nii.gz'
[bullseye]
short_axis_mr = '{tmp_path.as_posix()}/missing_mr.nii.gz'
short_axis_mask = '{tmp_path.as_posix()}/missing_mask.nii.gz'
geometry_json = '{tmp_path.as_posix()}/missing_geometry.json'
coronary_mask = '{tmp_path.as_posix()}/missing_coronary.nii.gz'
world_myocardium_mask = '{tmp_path.as_posix()}/missing_myocardium.nii.gz'
""",
    )
    with pytest.raises(ConfigError) as caught:
        load_config(config_path, "bullseye")
    message = str(caught.value)
    assert "bullseye.short_axis_mr" in message
    assert "bullseye.short_axis_mask" in message
    assert "bullseye.geometry_json" in message
    assert "bullseye.coronary_mask" in message
    assert "bullseye.world_myocardium_mask" in message


def test_explicit_empty_label_mappings_override_defaults(tmp_path):
    source = tmp_path / "source.nii.gz"
    source.touch()
    config_path = write_config(
        tmp_path,
        f"""
[case]
id = "case"
output_directory = '{tmp_path.as_posix()}/results'
[short_axis]
source_mr = '{source.as_posix()}'
[bullseye]
coronary_mask = '{source.as_posix()}'
world_myocardium_mask = '{source.as_posix()}'
territory_label_map = {{}}
coronary_label_colors = {{}}
coronary_label_names = {{}}
""",
    )
    config = load_config(config_path, "short-axis")
    assert config.bullseye.territory_label_map == {}
    assert config.bullseye.coronary_label_colors == {}
    assert config.bullseye.coronary_label_names == {}


@pytest.mark.parametrize(
    ("field", "value"),
    [("slice_spacing_mm", 0), ("pixel_spacing_mm", -1)],
)
def test_short_axis_rejects_nonpositive_spacing(tmp_path, field, value):
    source = tmp_path / "source.nii.gz"
    source.touch()
    config_path = write_config(
        tmp_path,
        f"""
[case]
id = "case"
output_directory = '{tmp_path.as_posix()}/results'
[short_axis]
source_mr = '{source.as_posix()}'
{field} = {value}
[bullseye]
coronary_mask = '{source.as_posix()}'
world_myocardium_mask = '{source.as_posix()}'
""",
    )
    with pytest.raises(ConfigError, match=field):
        load_config(config_path, "short-axis")


@pytest.mark.parametrize("fov_mm", ([192.0, 192.0], [256.0, 192.0]))
def test_short_axis_accepts_independent_positive_fov_dimensions(tmp_path, fov_mm):
    source = tmp_path / "source.nii.gz"
    source.touch()
    config_path = write_config(
        tmp_path,
        f"""
[case]
id = "case"
output_directory = '{tmp_path.as_posix()}/results'
[short_axis]
source_mr = '{source.as_posix()}'
fov_mm = {fov_mm}
[bullseye]
coronary_mask = '{source.as_posix()}'
world_myocardium_mask = '{source.as_posix()}'
""",
    )

    config = load_config(config_path, "short-axis")

    assert config.short_axis.fov_mm == tuple(fov_mm)


@pytest.mark.parametrize("fov_mm", ("[0.0, 192.0]", "[192.0, -1.0]"))
def test_short_axis_rejects_nonpositive_fov_dimensions(tmp_path, fov_mm):
    source = tmp_path / "source.nii.gz"
    source.touch()
    config_path = write_config(
        tmp_path,
        f"""
[case]
id = "case"
output_directory = '{tmp_path.as_posix()}/results'
[short_axis]
source_mr = '{source.as_posix()}'
fov_mm = {fov_mm}
[bullseye]
coronary_mask = '{source.as_posix()}'
world_myocardium_mask = '{source.as_posix()}'
""",
    )

    with pytest.raises(ConfigError, match="short_axis.fov_mm"):
        load_config(config_path, "short-axis")


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("short_axis", "slice_spacing_mm", "inf"),
        ("bullseye", "coronary_smoothing_sigma_voxels", "nan"),
    ],
)
def test_rejects_nonfinite_positive_and_nonnegative_values(
    tmp_path, section, field, value
):
    source = tmp_path / "source.nii.gz"
    source.touch()
    config_path = write_config(
        tmp_path,
        f"""
[case]
id = "case"
output_directory = '{tmp_path.as_posix()}/results'
[short_axis]
source_mr = '{source.as_posix()}'
{"" if section == "bullseye" else f"{field} = {value}"}
[bullseye]
coronary_mask = '{source.as_posix()}'
world_myocardium_mask = '{source.as_posix()}'
{"" if section == "short_axis" else f"{field} = {value}"}
""",
    )
    with pytest.raises(ConfigError, match=field):
        load_config(config_path, "short-axis")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fov_mm", "[nan, 10.0]"),
        ("distance_value_range_mm", "[0.0, inf]"),
    ],
)
def test_rejects_nonfinite_two_value_ranges(tmp_path, field, value):
    source = tmp_path / "source.nii.gz"
    source.touch()
    section = "short_axis" if field == "fov_mm" else "bullseye"
    config_path = write_config(
        tmp_path,
        f"""
[case]
id = "case"
output_directory = '{tmp_path.as_posix()}/results'
[short_axis]
source_mr = '{source.as_posix()}'
{field} = {value}
[bullseye]
coronary_mask = '{source.as_posix()}'
world_myocardium_mask = '{source.as_posix()}'
"""
        if section == "short_axis"
        else f"""
[case]
id = "case"
output_directory = '{tmp_path.as_posix()}/results'
[short_axis]
source_mr = '{source.as_posix()}'
[bullseye]
coronary_mask = '{source.as_posix()}'
world_myocardium_mask = '{source.as_posix()}'
{field} = {value}
""",
    )
    with pytest.raises(ConfigError, match=field):
        load_config(config_path, "short-axis")
