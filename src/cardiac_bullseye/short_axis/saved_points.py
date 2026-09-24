"""Persist and validate the geometry selected in the short-axis planner."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .geometry import LongAxisPlan


SCHEMA_NAME = "cardiac_bullseye.short_axis_points"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SavedShortAxisPoints:
    plan: LongAxisPlan
    apex_t_mm: float
    base_t_mm: float
    slice_spacing_mm: float
    pixel_spacing_mm: float
    fov_mm: tuple[float, float] | None
    reference_x_axis: np.ndarray | None
    display_affine: np.ndarray
    orientation_reference_path: Path | None = None
    reference_long_axis_difference_deg: float | None = None


_PLAN_FIELDS = (
    "vla_x_axis",
    "vla_y_axis",
    "vla_normal",
    "hla_x_axis",
    "hla_y_axis",
    "hla_normal",
    "apex",
    "valve",
    "long_axis",
)


def save_short_axis_points(
    path: str | Path,
    *,
    plan: LongAxisPlan,
    apex_t_mm: float,
    base_t_mm: float,
    slice_spacing_mm: float,
    pixel_spacing_mm: float,
    fov_mm: tuple[float, float] | None,
    source_shape: tuple[int, ...],
    source_affine: np.ndarray,
    reference_x_axis: np.ndarray | None,
    display_affine: np.ndarray,
    orientation_reference_path: str | Path | None = None,
    reference_long_axis_difference_deg: float | None = None,
) -> Path:
    """Save the complete interactive plan needed for a headless repeat run."""

    path = Path(path).expanduser().resolve()
    payload = {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "coordinate_system": "NIfTI world coordinates in millimetres",
        "source_shape": [int(value) for value in source_shape],
        "source_affine": np.asarray(source_affine, dtype=float).tolist(),
        "plan": {
            name: np.asarray(getattr(plan, name), dtype=float).tolist()
            for name in _PLAN_FIELDS
        },
        "apex_plane_offset_mm": float(apex_t_mm),
        "base_plane_offset_mm": float(base_t_mm),
        "slice_spacing_mm": float(slice_spacing_mm),
        "pixel_spacing_mm": float(pixel_spacing_mm),
        "fov_mm": None if fov_mm is None else [float(value) for value in fov_mm],
        "reference_x_axis_world": (
            None
            if reference_x_axis is None
            else np.asarray(reference_x_axis, dtype=float).tolist()
        ),
        "display_affine": np.asarray(display_affine, dtype=float).tolist(),
        "orientation_reference_path": (
            None
            if orientation_reference_path is None
            else str(Path(orientation_reference_path).expanduser().resolve())
        ),
        "reference_long_axis_difference_deg": reference_long_axis_difference_deg,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def load_short_axis_points(
    path: str | Path,
    *,
    source_shape: tuple[int, ...],
    source_affine: np.ndarray,
) -> SavedShortAxisPoints:
    """Load a saved plan and reject reuse on a different source image grid."""

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            "reuse_saved_points=true requires an existing short-axis point file: "
            f"{path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA_NAME:
        raise ValueError(f"Unsupported short-axis point file schema: {path}")
    stored_shape = tuple(int(value) for value in payload["source_shape"])
    stored_affine = np.asarray(payload["source_affine"], dtype=float)
    if stored_shape != tuple(source_shape) or not np.allclose(
        stored_affine,
        np.asarray(source_affine, dtype=float),
        atol=1.0e-4,
        rtol=1.0e-5,
    ):
        raise ValueError(
            "Saved short-axis points do not match the current source image grid "
            "(shape and affine must be unchanged)"
        )
    plan_payload = payload["plan"]
    plan = LongAxisPlan(
        **{
            name: np.asarray(plan_payload[name], dtype=float)
            for name in _PLAN_FIELDS
        }
    )
    reference_x = payload.get("reference_x_axis_world")
    fov = payload.get("fov_mm")
    orientation_path = payload.get("orientation_reference_path")
    return SavedShortAxisPoints(
        plan=plan,
        apex_t_mm=float(payload["apex_plane_offset_mm"]),
        base_t_mm=float(payload["base_plane_offset_mm"]),
        slice_spacing_mm=float(payload["slice_spacing_mm"]),
        pixel_spacing_mm=float(payload["pixel_spacing_mm"]),
        fov_mm=None if fov is None else (float(fov[0]), float(fov[1])),
        reference_x_axis=(
            None if reference_x is None else np.asarray(reference_x, dtype=float)
        ),
        display_affine=np.asarray(payload["display_affine"], dtype=float),
        orientation_reference_path=(
            None if orientation_path is None else Path(orientation_path)
        ),
        reference_long_axis_difference_deg=payload.get(
            "reference_long_axis_difference_deg"
        ),
    )
