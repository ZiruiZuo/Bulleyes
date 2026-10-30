"""TOML configuration for cardiac short-axis and bullseye workflows."""

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Iterable, Literal

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - used only by Python 3.10
    import tomli as tomllib


WorkflowName = Literal[
    "short-axis",
    "bullseye",
    "pipeline",
    "lge-bullseye",
    "intensity-bullseye",
]


class ConfigError(ValueError):
    """Raised when configuration contains one or more validation errors."""

    def __init__(self, errors: Iterable[str]) -> None:
        self.errors = tuple(errors)
        super().__init__("Configuration validation failed:\n" + "\n".join(
            f"- {error}" for error in self.errors
        ))


@dataclass(frozen=True)
class CaseConfig:
    id: str
    output_directory: Path

    def output_path(self, suffix: str) -> Path:
        return self.output_directory / f"{self.id}_{suffix}"


@dataclass(frozen=True)
class ShortAxisConfig:
    source_mr: Path | None
    source_mask: Path | None = None
    orientation_reference: Path | None = None
    slice_spacing_mm: float = 8.0
    pixel_spacing_mm: float | None = None
    fov_mm: tuple[float, float] | None = None
    reuse_saved_points: bool = False


@dataclass(frozen=True)
class BullseyeConfig:
    short_axis_mr: Path | None
    short_axis_mask: Path | None
    geometry_json: Path | None
    coronary_mask: Path | None
    world_myocardium_mask: Path | None
    territory_label_map: dict[int, int] = field(
        default_factory=lambda: {2: 1, 3: 2, 4: 3}
    )
    angle_count: int = 360
    radial_count: int = 300
    angular_smoothing_sigma_deg: float = 1.5
    radial_smoothing_sigma: float = 1.5
    six_segment_warp: bool = True
    intensity_coronary_overlay: bool = False
    coronary_mask_is_centerline: bool = False
    coronary_min_path_length_mm: float = 2.0
    coronary_resample_step_mm: float = 0.75
    coronary_smoothing_sigma_voxels: float = 0.0
    coronary_default_color: str | None = None
    coronary_label_colors: dict[int, str] = field(default_factory=dict)
    coronary_label_names: dict[int, str] = field(default_factory=lambda: {
        1: "Coronary LAD", 2: "Coronary LCX", 3: "Coronary RCA"
    })
    coronary_linewidth: float = 1.8
    distance_vessel_labels: tuple[int, ...] = (1, 2, 3)
    distance_colormap: str = "turbo"
    distance_value_range_mm: tuple[float, float] | None = None
    apex_territory: int | None = None
    show_final_figure: bool = True
    reuse_saved_points: bool = False
    landmarks_path: Path | None = None


@dataclass(frozen=True)
class LgeBullseyeConfig:
    lge: Path | None
    mask: Path | None
    lge_scar_mask: Path | None = None
    angle_count: int = 360
    radial_count: int = 300
    angular_smoothing_sigma_deg: float = 1.5
    radial_smoothing_sigma: float = 1.5
    six_segment_warp: bool = True
    scar_sd_multiplier: float = 5.0
    normal_roi_min_voxels: int = 20
    reuse_saved_points: bool = False
    landmarks_path: Path | None = None


@dataclass(frozen=True)
class AppConfig:
    case: CaseConfig
    short_axis: ShortAxisConfig
    bullseye: BullseyeConfig
    lge_bullseye: LgeBullseyeConfig = field(
        default_factory=lambda: LgeBullseyeConfig(lge=None, mask=None)
    )


def _path(value: Any, base_directory: Path) -> Path | None:
    if value is None or value == "":
        return None
    path = Path(value)
    return path if path.is_absolute() else base_directory / path


def _integer_keys(values: dict[str, Any] | None) -> dict[int, Any]:
    return {int(key): value for key, value in (values or {}).items()}


def _pair(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    return tuple(value)  # type: ignore[return-value]


def _input_error(errors: list[str], name: str, path: Path | None) -> None:
    if path is None or not path.is_file():
        errors.append(f"{name} must be an existing file")


def _positive_integer(errors: list[str], name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        errors.append(f"{name} must be a positive integer")


def _minimum_integer(
    errors: list[str], name: str, value: Any, minimum: int
) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        errors.append(f"{name} must be an integer of at least {minimum}")


def _positive_number(errors: list[str], name: str, value: Any) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        errors.append(f"{name} must be positive")


def _nonnegative_number(errors: list[str], name: str, value: Any) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        errors.append(f"{name} must be nonnegative")


def _validate_lge_mapping_options(
    errors: list[str], prefix: str, options: Any
) -> None:
    _minimum_integer(errors, f"{prefix}.angle_count", options.angle_count, 4)
    _minimum_integer(errors, f"{prefix}.radial_count", options.radial_count, 3)
    _nonnegative_number(
        errors,
        f"{prefix}.angular_smoothing_sigma_deg",
        options.angular_smoothing_sigma_deg,
    )
    _nonnegative_number(
        errors,
        f"{prefix}.radial_smoothing_sigma",
        options.radial_smoothing_sigma,
    )


def _increasing_pair(errors: list[str], name: str, value: Any) -> None:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(item)
            for item in value
        )
        or value[0] >= value[1]
    ):
        errors.append(f"{name} must contain two strictly increasing values")


def _positive_pair(errors: list[str], name: str, value: Any) -> None:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(item)
            or item <= 0
            for item in value
        )
    ):
        errors.append(f"{name} must contain two positive values")


def validate_config(config: AppConfig, workflow: WorkflowName) -> None:
    """Raise ``ConfigError`` if a workflow configuration is invalid."""
    errors: list[str] = []

    if workflow == "short-axis":
        _input_error(errors, "short_axis.source_mr", config.short_axis.source_mr)
    elif workflow == "bullseye":
        for name, path in (
            ("bullseye.short_axis_mr", config.bullseye.short_axis_mr),
            ("bullseye.short_axis_mask", config.bullseye.short_axis_mask),
            ("bullseye.geometry_json", config.bullseye.geometry_json),
            ("bullseye.coronary_mask", config.bullseye.coronary_mask),
            ("bullseye.world_myocardium_mask", config.bullseye.world_myocardium_mask),
        ):
            _input_error(errors, name, path)
    elif workflow == "intensity-bullseye":
        for name, path in (
            ("bullseye.short_axis_mr", config.bullseye.short_axis_mr),
            ("bullseye.short_axis_mask", config.bullseye.short_axis_mask),
        ):
            _input_error(errors, name, path)
        if config.bullseye.intensity_coronary_overlay:
            for name, path in (
                ("bullseye.geometry_json", config.bullseye.geometry_json),
                ("bullseye.coronary_mask", config.bullseye.coronary_mask),
            ):
                _input_error(errors, name, path)
    elif workflow == "pipeline":
        for name, path in (
            ("short_axis.source_mr", config.short_axis.source_mr),
            ("short_axis.source_mask", config.short_axis.source_mask),
            ("bullseye.coronary_mask", config.bullseye.coronary_mask),
            ("bullseye.world_myocardium_mask", config.bullseye.world_myocardium_mask),
        ):
            _input_error(errors, name, path)
    elif workflow == "lge-bullseye":
        for name, path in (
            ("lge_bullseye.lge", config.lge_bullseye.lge),
            ("lge_bullseye.mask", config.lge_bullseye.mask),
        ):
            _input_error(errors, name, path)
        if config.lge_bullseye.lge_scar_mask is not None:
            _input_error(
                errors, "lge_bullseye.lge_scar_mask", config.lge_bullseye.lge_scar_mask
            )
    else:
        errors.append(
            "workflow must be one of short-axis, bullseye, pipeline, "
            "lge-bullseye, intensity-bullseye; "
            f"got {workflow!r}"
        )

    _positive_number(errors, "short_axis.slice_spacing_mm", config.short_axis.slice_spacing_mm)
    if config.short_axis.pixel_spacing_mm is not None:
        _positive_number(errors, "short_axis.pixel_spacing_mm", config.short_axis.pixel_spacing_mm)
    if config.short_axis.fov_mm is not None:
        _positive_pair(errors, "short_axis.fov_mm", config.short_axis.fov_mm)
    if (
        workflow in ("short-axis", "pipeline")
        and config.short_axis.orientation_reference is not None
    ):
        _input_error(
            errors,
            "short_axis.orientation_reference",
            config.short_axis.orientation_reference,
        )
    _positive_integer(errors, "bullseye.angle_count", config.bullseye.angle_count)
    _positive_integer(errors, "bullseye.radial_count", config.bullseye.radial_count)
    _nonnegative_number(
        errors,
        "bullseye.angular_smoothing_sigma_deg",
        config.bullseye.angular_smoothing_sigma_deg,
    )
    _nonnegative_number(
        errors,
        "bullseye.radial_smoothing_sigma",
        config.bullseye.radial_smoothing_sigma,
    )
    _positive_number(
        errors,
        "bullseye.coronary_min_path_length_mm",
        config.bullseye.coronary_min_path_length_mm,
    )
    _positive_number(
        errors,
        "bullseye.coronary_resample_step_mm",
        config.bullseye.coronary_resample_step_mm,
    )
    _nonnegative_number(
        errors,
        "bullseye.coronary_smoothing_sigma_voxels",
        config.bullseye.coronary_smoothing_sigma_voxels,
    )
    _positive_number(errors, "bullseye.coronary_linewidth", config.bullseye.coronary_linewidth)
    if config.bullseye.distance_value_range_mm is not None:
        _increasing_pair(
            errors,
            "bullseye.distance_value_range_mm",
            config.bullseye.distance_value_range_mm,
        )
    if config.bullseye.apex_territory not in (None, 1, 2, 3):
        errors.append("bullseye.apex_territory must be 1, 2, or 3")

    _validate_lge_mapping_options(errors, "lge_bullseye", config.lge_bullseye)
    _positive_number(
        errors,
        "lge_bullseye.scar_sd_multiplier",
        config.lge_bullseye.scar_sd_multiplier,
    )
    if (
        isinstance(config.lge_bullseye.normal_roi_min_voxels, bool)
        or not isinstance(config.lge_bullseye.normal_roi_min_voxels, int)
        or config.lge_bullseye.normal_roi_min_voxels < 2
    ):
        errors.append("lge_bullseye.normal_roi_min_voxels must be an integer of at least 2")

    if errors:
        raise ConfigError(errors)


def load_config(path: str | Path, workflow: WorkflowName) -> AppConfig:
    """Read a TOML config, resolve relative paths, and validate it for a workflow."""
    config_path = Path(path)
    with config_path.open("rb") as config_file:
        data = tomllib.load(config_file)
    base_directory = config_path.parent

    case_data = data.get("case", {})
    short_axis_data = data.get("short_axis", {})
    bullseye_data = data.get("bullseye", {})
    lge_bullseye_data = data.get("lge_bullseye", {})

    config = AppConfig(
        case=CaseConfig(
            id=case_data.get("id", ""),
            output_directory=_path(case_data.get("output_directory", ""), base_directory)
            or base_directory,
        ),
        short_axis=ShortAxisConfig(
            source_mr=_path(short_axis_data.get("source_mr"), base_directory),
            source_mask=_path(short_axis_data.get("source_mask"), base_directory),
            orientation_reference=_path(
                short_axis_data.get("orientation_reference"), base_directory
            ),
            reuse_saved_points=short_axis_data.get("reuse_saved_points", False),
            slice_spacing_mm=short_axis_data.get("slice_spacing_mm", 8.0),
            pixel_spacing_mm=short_axis_data.get("pixel_spacing_mm"),
            fov_mm=_pair(short_axis_data.get("fov_mm")),
        ),
        bullseye=BullseyeConfig(
            short_axis_mr=_path(bullseye_data.get("short_axis_mr"), base_directory),
            short_axis_mask=_path(bullseye_data.get("short_axis_mask"), base_directory),
            geometry_json=_path(bullseye_data.get("geometry_json"), base_directory),
            coronary_mask=_path(bullseye_data.get("coronary_mask"), base_directory),
            world_myocardium_mask=_path(
                bullseye_data.get("world_myocardium_mask"), base_directory
            ),
            territory_label_map=(
                _integer_keys(bullseye_data["territory_label_map"])
                if "territory_label_map" in bullseye_data
                else {2: 1, 3: 2, 4: 3}
            ),
            angle_count=bullseye_data.get("angle_count", 360),
            radial_count=bullseye_data.get("radial_count", 300),
            angular_smoothing_sigma_deg=bullseye_data.get(
                "angular_smoothing_sigma_deg", 1.5
            ),
            radial_smoothing_sigma=bullseye_data.get("radial_smoothing_sigma", 1.5),
            six_segment_warp=bullseye_data.get("six_segment_warp", True),
            intensity_coronary_overlay=bullseye_data.get(
                "intensity_coronary_overlay", False
            ),
            coronary_mask_is_centerline=bullseye_data.get(
                "coronary_mask_is_centerline", False
            ),
            coronary_min_path_length_mm=bullseye_data.get(
                "coronary_min_path_length_mm", 2.0
            ),
            coronary_resample_step_mm=bullseye_data.get(
                "coronary_resample_step_mm", 0.75
            ),
            coronary_smoothing_sigma_voxels=bullseye_data.get(
                "coronary_smoothing_sigma_voxels", 0.0
            ),
            coronary_default_color=bullseye_data.get("coronary_default_color"),
            coronary_label_colors=_integer_keys(
                bullseye_data.get("coronary_label_colors")
            ),
            coronary_label_names=(
                _integer_keys(bullseye_data["coronary_label_names"])
                if "coronary_label_names" in bullseye_data
                else {1: "Coronary LAD", 2: "Coronary LCX", 3: "Coronary RCA"}
            ),
            coronary_linewidth=bullseye_data.get("coronary_linewidth", 1.8),
            distance_vessel_labels=tuple(
                bullseye_data.get("distance_vessel_labels", (1, 2, 3))
            ),
            distance_colormap=bullseye_data.get("distance_colormap", "turbo"),
            distance_value_range_mm=_pair(
                bullseye_data.get("distance_value_range_mm")
            ),
            apex_territory=bullseye_data.get("apex_territory"),
            show_final_figure=bullseye_data.get("show_final_figure", True),
            reuse_saved_points=bullseye_data.get("reuse_saved_points", False),
            landmarks_path=_path(bullseye_data.get("landmarks_path"), base_directory),
        ),
        lge_bullseye=LgeBullseyeConfig(
            lge=_path(lge_bullseye_data.get("lge"), base_directory),
            mask=_path(lge_bullseye_data.get("mask"), base_directory),
            lge_scar_mask=_path(
                lge_bullseye_data.get("lge_scar_mask"), base_directory
            ),
            angle_count=lge_bullseye_data.get("angle_count", 360),
            radial_count=lge_bullseye_data.get("radial_count", 300),
            angular_smoothing_sigma_deg=lge_bullseye_data.get(
                "angular_smoothing_sigma_deg", 1.5
            ),
            radial_smoothing_sigma=lge_bullseye_data.get(
                "radial_smoothing_sigma", 1.5
            ),
            six_segment_warp=lge_bullseye_data.get("six_segment_warp", True),
            scar_sd_multiplier=lge_bullseye_data.get(
                "scar_sd_multiplier", 5.0
            ),
            normal_roi_min_voxels=lge_bullseye_data.get(
                "normal_roi_min_voxels", 20
            ),
            reuse_saved_points=lge_bullseye_data.get(
                "reuse_saved_points", False
            ),
            landmarks_path=_path(
                lge_bullseye_data.get("landmarks_path"), base_directory
            ),
        ),
    )
    validate_config(config, workflow)
    return config
