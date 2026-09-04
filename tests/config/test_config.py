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
    assert config.bullseye.distance_vessel_labels == (2, 3, 4)
    assert config.case.output_path("bullseye.png").name == "patient_043_bullseye.png"


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
