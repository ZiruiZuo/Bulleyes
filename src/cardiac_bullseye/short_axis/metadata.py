"""Export short-axis planning geometry in NIfTI world coordinates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .geometry import LongAxisPlan, StackResult, unit


SCHEMA_NAME = "convert2shortaxis.geometry"
SCHEMA_VERSION = 1


def default_geometry_path(short_axis_path: str | Path) -> Path:
    path = Path(short_axis_path).expanduser().resolve()
    if path.name.lower().endswith(".nii.gz"):
        stem = path.name[:-7]
    else:
        stem = path.stem
    return path.with_name(f"{stem}_geometry.json")


def save_planning_geometry(
    path: str | Path,
    *,
    plan: LongAxisPlan,
    stack: StackResult,
    apex_t_mm: float,
    base_t_mm: float,
    source_image_path: str | Path,
    short_axis_path: str | Path,
) -> Path:
    """Save the complete spatial contract needed by downstream bullseye tools."""

    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    long_axis = unit(plan.long_axis, "LV long axis")
    sa_x = unit(stack.affine[:3, 0], "short-axis x axis")
    sa_y = unit(stack.affine[:3, 1], "short-axis y axis")
    sa_normal = unit(stack.affine[:3, 2], "short-axis normal")
    if np.dot(sa_normal, long_axis) < 0:
        sa_normal = -sa_normal

    true_apex = np.asarray(plan.apex, dtype=float)
    base_center = true_apex + float(base_t_mm) * long_axis
    first_center = true_apex + float(stack.positions_mm[0]) * long_axis
    last_center = true_apex + float(stack.positions_mm[-1]) * long_axis
    payload = {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "coordinate_system": "NIfTI world coordinates in millimetres",
        "source_image_path": str(Path(source_image_path).expanduser().resolve()),
        "short_axis_image_path": str(Path(short_axis_path).expanduser().resolve()),
        "true_apex_world_mm": true_apex.tolist(),
        "mitral_center_world_mm": np.asarray(plan.valve, dtype=float).tolist(),
        "selected_apex_plane_center_world_mm": (
            true_apex + float(apex_t_mm) * long_axis
        ).tolist(),
        "selected_base_center_world_mm": base_center.tolist(),
        "first_slice_center_world_mm": first_center.tolist(),
        "last_slice_center_world_mm": last_center.tolist(),
        "long_axis_apex_to_base": long_axis.tolist(),
        "short_axis_x_world": sa_x.tolist(),
        "short_axis_y_world": sa_y.tolist(),
        "short_axis_normal_world": sa_normal.tolist(),
        "apex_plane_offset_mm": float(apex_t_mm),
        "base_plane_offset_mm": float(base_t_mm),
        "slice_positions_from_true_apex_mm": np.asarray(
            stack.positions_mm, dtype=float
        ).tolist(),
        "short_axis_affine": np.asarray(stack.affine, dtype=float).tolist(),
        "short_axis_shape": [int(value) for value in stack.data.shape],
    }

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)
    return path
