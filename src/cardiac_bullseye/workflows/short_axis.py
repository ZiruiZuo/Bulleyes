"""Interactive short-axis planning workflow."""

from dataclasses import dataclass
from pathlib import Path

from ..config import AppConfig, validate_config
from ..short_axis import interactive_convert_to_short_axis


@dataclass(frozen=True)
class ShortAxisOutputs:
    mr: Path
    mask: Path | None
    geometry: Path


def generate_short_axis(config: AppConfig) -> ShortAxisOutputs:
    """Plan and save a case's short-axis MR, mask, and geometry."""
    validate_config(config, "short-axis")

    mr = config.case.output_path("short_axis.nii.gz")
    mask = (
        config.case.output_path("short_axis_mask.nii.gz")
        if config.short_axis.source_mask
        else None
    )
    geometry = config.case.output_path("geometry.json")
    interactive_convert_to_short_axis(
        image_path=config.short_axis.source_mr,
        output_path=mr,
        mask_path=config.short_axis.source_mask,
        mask_output_path=mask,
        geometry_output_path=geometry,
        slice_spacing_mm=config.short_axis.slice_spacing_mm,
        pixel_spacing_mm=config.short_axis.pixel_spacing_mm,
        fov_mm=config.short_axis.fov_mm,
    )

    missing = [
        path
        for path in (mr, mask, geometry)
        if path is not None and not path.is_file()
    ]
    if missing:
        raise RuntimeError(
            "Short-axis planning closed before these outputs were saved: "
            + ", ".join(map(str, missing))
        )
    return ShortAxisOutputs(mr, mask, geometry)
