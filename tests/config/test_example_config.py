import tomllib
from pathlib import Path

from cardiac_bullseye.config import load_config


def test_example_config_exercises_all_config_sections(monkeypatch):
    """Keep the documented example parseable by the real pipeline loader."""
    example_path = Path(__file__).parents[2] / "config.example.toml"

    with example_path.open("rb") as config_file:
        raw_config = tomllib.load(config_file)
    assert raw_config.keys() == {"case", "short_axis", "bullseye", "lge_bullseye"}

    monkeypatch.setattr(Path, "is_file", lambda _path: True)
    config = load_config(example_path, "pipeline")

    assert config.case.id == "patient_043"
    assert config.short_axis.slice_spacing_mm == 8.0
    assert config.short_axis.fov_mm == (192.0, 192.0)
    assert config.bullseye.angle_count == 360
    assert config.bullseye.territory_label_map == {2: 1, 3: 2, 4: 3}
    assert config.bullseye.distance_vessel_labels == (1, 2, 3)
    assert config.lge_bullseye.lge is not None
    assert config.lge_bullseye.mask is not None
    assert config.lge_bullseye.lge_scar_mask is None


def test_focused_lge_bullseye_example_loads_external_scar(monkeypatch):
    example_path = Path(__file__).parents[2] / "toml" / "lge_bullseye.example.toml"

    monkeypatch.setattr(Path, "is_file", lambda _path: True)
    config = load_config(example_path, "lge-bullseye")

    assert config.case.id == "patient_043"
    assert config.lge_bullseye.lge is not None
    assert config.lge_bullseye.mask is not None
    assert config.lge_bullseye.lge_scar_mask is not None


def test_only_the_unified_focused_lge_example_is_available():
    example_directory = Path(__file__).parents[2] / "toml"

    assert {path.name for path in example_directory.glob("lge*.example.toml")} == {
        "lge_bullseye.example.toml"
    }
