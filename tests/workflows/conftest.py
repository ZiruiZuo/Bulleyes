from pathlib import Path

import pytest

from cardiac_bullseye.config import (
    AppConfig,
    BullseyeConfig,
    CaseConfig,
    ShortAxisConfig,
)


@pytest.fixture
def app_config(tmp_path: Path) -> AppConfig:
    paths = {
        name: tmp_path / name
        for name in (
            "source_mr.nii.gz",
            "source_mask.nii.gz",
            "short_axis.nii.gz",
            "short_axis_mask.nii.gz",
            "geometry.json",
            "coronary.nii.gz",
            "myocardium.nii.gz",
        )
    }
    for path in paths.values():
        path.touch()

    return AppConfig(
        case=CaseConfig(id="patient_043", output_directory=tmp_path),
        short_axis=ShortAxisConfig(
            source_mr=paths["source_mr.nii.gz"],
            source_mask=paths["source_mask.nii.gz"],
        ),
        bullseye=BullseyeConfig(
            short_axis_mr=paths["short_axis.nii.gz"],
            short_axis_mask=paths["short_axis_mask.nii.gz"],
            geometry_json=paths["geometry.json"],
            coronary_mask=paths["coronary.nii.gz"],
            world_myocardium_mask=paths["myocardium.nii.gz"],
        ),
    )
