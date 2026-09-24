"""Geometry and interpolation primitives for cardiac short-axis planning.

All public coordinates in this module are NIfTI world coordinates (millimetres).
The module deliberately does not import nibabel, which keeps the numerical core
easy to test and reuse.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import numpy as np
from scipy.ndimage import map_coordinates


EPS = 1.0e-8

def unit(vector: Iterable[float], name: str = "vector") -> np.ndarray:
    """Return a normalised 3-vector or raise a useful error."""
    value = np.asarray(vector, dtype=float)
    if value.shape != (3,):
        raise ValueError(f"{name} must contain three values, got {value.shape}")
    length = float(np.linalg.norm(value))
    if length < EPS:
        raise ValueError(f"{name} is too short to define a direction")
    return value / length


def undirected_axis_angle_deg(
    first: Iterable[float], second: Iterable[float]
) -> float:
    """Return the acute angle between two axes, ignoring normal polarity."""

    first_axis = unit(first, "first axis")
    second_axis = unit(second, "second axis")
    cosine = float(np.clip(abs(np.dot(first_axis, second_axis)), 0.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


@dataclass(frozen=True)
class PlaneGrid:
    """A regularly sampled 2-D plane embedded in 3-D world space."""

    center: np.ndarray
    x_axis: np.ndarray
    y_axis: np.ndarray
    width: int
    height: int
    pixel_spacing: float

    def __post_init__(self) -> None:
        if self.width < 2 or self.height < 2:
            raise ValueError("plane width and height must both be at least 2")
        if self.pixel_spacing <= 0:
            raise ValueError("pixel spacing must be positive")
        x_axis = unit(self.x_axis, "plane x axis")
        y_raw = np.asarray(self.y_axis, dtype=float)
        y_axis = unit(y_raw - np.dot(y_raw, x_axis) * x_axis, "plane y axis")
        object.__setattr__(self, "center", np.asarray(self.center, dtype=float))
        object.__setattr__(self, "x_axis", x_axis)
        object.__setattr__(self, "y_axis", y_axis)

    @property
    def normal(self) -> np.ndarray:
        return unit(np.cross(self.x_axis, self.y_axis), "plane normal")

    def pixel_to_world(self, x: float, y: float) -> np.ndarray:
        dx = (float(x) - (self.width - 1) / 2.0) * self.pixel_spacing
        dy = (float(y) - (self.height - 1) / 2.0) * self.pixel_spacing
        return self.center + dx * self.x_axis + dy * self.y_axis

    def world_grid(self) -> np.ndarray:
        xs = (np.arange(self.width) - (self.width - 1) / 2.0) * self.pixel_spacing
        ys = (np.arange(self.height) - (self.height - 1) / 2.0) * self.pixel_spacing
        return (
            self.center[None, None, :]
            + xs[None, :, None] * self.x_axis[None, None, :]
            + ys[:, None, None] * self.y_axis[None, None, :]
        )


@dataclass(frozen=True)
class LongAxisPlan:
    """The two orthogonal reference planes and the final LV long axis."""

    vla_x_axis: np.ndarray
    vla_y_axis: np.ndarray
    vla_normal: np.ndarray
    hla_x_axis: np.ndarray
    hla_y_axis: np.ndarray
    hla_normal: np.ndarray
    apex: np.ndarray
    valve: np.ndarray
    long_axis: np.ndarray


@dataclass(frozen=True)
class StackResult:
    data: np.ndarray
    affine: np.ndarray
    positions_mm: np.ndarray


def voxel_axis_world_direction(affine: np.ndarray, axis: int) -> np.ndarray:
    return unit(np.asarray(affine, dtype=float)[:3, axis], f"voxel axis {axis}")


def world_corners(shape: tuple[int, int, int], affine: np.ndarray) -> np.ndarray:
    """Return the world coordinates of the eight voxel-centre corners."""
    corners = []
    for ijk in product(*[(0.0, float(size - 1)) for size in shape]):
        corners.append((np.asarray(affine, float) @ [*ijk, 1.0])[:3])
    return np.asarray(corners)


def fit_plane_grid(
    shape: tuple[int, int, int],
    affine: np.ndarray,
    center: Iterable[float],
    x_axis: Iterable[float],
    y_axis: Iterable[float],
    pixel_spacing: float,
    margin_mm: float = 2.0,
) -> PlaneGrid:
    """Fit a plane grid to the projection of a complete input volume."""
    x_axis = unit(x_axis, "plane x axis")
    y_raw = np.asarray(y_axis, float)
    y_axis = unit(y_raw - np.dot(y_raw, x_axis) * x_axis, "plane y axis")
    center = np.asarray(center, float)
    corners = world_corners(shape, affine)
    px = (corners - center) @ x_axis
    py = (corners - center) @ y_axis
    xmin, xmax = float(px.min() - margin_mm), float(px.max() + margin_mm)
    ymin, ymax = float(py.min() - margin_mm), float(py.max() + margin_mm)
    shifted_center = center + 0.5 * (xmin + xmax) * x_axis + 0.5 * (ymin + ymax) * y_axis
    width = max(2, int(np.ceil((xmax - xmin) / pixel_spacing)) + 1)
    height = max(2, int(np.ceil((ymax - ymin) / pixel_spacing)) + 1)
    return PlaneGrid(shifted_center, x_axis, y_axis, width, height, pixel_spacing)


def resample_plane(
    volume: np.ndarray,
    affine: np.ndarray,
    grid: PlaneGrid,
    order: int = 1,
    cval: float = 0.0,
) -> np.ndarray:
    """Sample a 3-D array on ``grid`` using its own voxel-to-world affine."""
    if volume.ndim != 3:
        raise ValueError(f"volume must be 3-D, got {volume.ndim}-D")
    world = grid.world_grid()
    homogeneous = np.concatenate(
        [world.reshape(-1, 3), np.ones((world.shape[0] * world.shape[1], 1))], axis=1
    )
    voxel = (np.linalg.inv(np.asarray(affine, float)) @ homogeneous.T)[:3]
    sampled = map_coordinates(
        np.asarray(volume), voxel, order=order, mode="constant", cval=cval, prefilter=order > 1
    )
    return sampled.reshape(grid.height, grid.width)


def build_vla_from_axial_points(
    apex: Iterable[float], valve: Iterable[float], axial_normal: Iterable[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the vertical long-axis plane through two axial-view landmarks."""
    apex = np.asarray(apex, float)
    valve = np.asarray(valve, float)
    normal = unit(axial_normal, "axial normal")
    line = valve - apex
    line = line - np.dot(line, normal) * normal
    x_axis = unit(line, "axial apex-to-valve line")
    y_axis = normal
    plane_normal = unit(np.cross(x_axis, y_axis), "VLA normal")
    return x_axis, y_axis, plane_normal


def build_long_axis_plan(
    vla_x_axis: Iterable[float],
    vla_y_axis: Iterable[float],
    apex: Iterable[float],
    valve: Iterable[float],
) -> LongAxisPlan:
    """Build the HLA plane and final short-axis normal from VLA landmarks."""
    vx = unit(vla_x_axis, "VLA x axis")
    vy_raw = np.asarray(vla_y_axis, float)
    vy = unit(vy_raw - np.dot(vy_raw, vx) * vx, "VLA y axis")
    vla_normal = unit(np.cross(vx, vy), "VLA normal")
    apex = np.asarray(apex, float)
    valve = np.asarray(valve, float)
    axis = unit(valve - apex, "VLA apex-to-valve line")
    # Landmark clicks lie in the VLA plane; remove tiny numerical drift.
    axis = unit(axis - np.dot(axis, vla_normal) * vla_normal, "LV long axis")
    hla_x = vla_normal
    hla_y = axis
    hla_normal = unit(np.cross(hla_x, hla_y), "HLA normal")
    return LongAxisPlan(
        vx,
        vy,
        vla_normal,
        hla_x,
        hla_y,
        hla_normal,
        apex,
        valve,
        axis,
    )


def projection_limits(
    shape: tuple[int, int, int], affine: np.ndarray, anchor: Iterable[float], axis: Iterable[float]
) -> tuple[float, float]:
    values = (world_corners(shape, affine) - np.asarray(anchor, float)) @ unit(axis)
    return float(values.min()), float(values.max())


def stack_positions(apex_t: float, base_t: float, spacing_mm: float) -> np.ndarray:
    """Return uniform slice offsets, preserving the requested spacing exactly."""
    if spacing_mm <= 0:
        raise ValueError("slice spacing must be positive")
    if base_t <= apex_t:
        raise ValueError("base boundary must be beyond apex boundary")
    count = int(np.floor((base_t - apex_t) / spacing_mm + 1.0e-7)) + 1
    return float(apex_t) + np.arange(count, dtype=float) * float(spacing_mm)


def generate_short_axis_stack(
    volume: np.ndarray,
    affine: np.ndarray,
    plan: LongAxisPlan,
    apex_t: float,
    base_t: float,
    slice_spacing_mm: float,
    pixel_spacing_mm: float,
    fov_mm: tuple[float, float] | None = None,
    interpolation_order: int = 1,
    reference_x_axis: Iterable[float] | None = None,
) -> StackResult:
    """Resample ``volume`` into an apex-to-base short-axis stack.

    The returned NIfTI-style affine has columns ``(SA x, SA y, long axis)``.
    ``apex_t`` and ``base_t`` are offsets from ``plan.apex`` along the final
    LV long axis.
    """
    if volume.ndim != 3:
        raise ValueError("only 3-D input volumes are supported")
    if pixel_spacing_mm <= 0:
        raise ValueError("pixel spacing must be positive")
    positions = stack_positions(apex_t, base_t, slice_spacing_mm)
    if reference_x_axis is None:
        x_axis = unit(plan.vla_normal, "short-axis x axis")
    else:
        reference = np.asarray(reference_x_axis, dtype=float)
        projected = reference - np.dot(reference, plan.long_axis) * plan.long_axis
        x_axis = unit(projected, "reference short-axis x axis")
    y_axis = unit(np.cross(plan.long_axis, x_axis), "short-axis y axis")
    normal = unit(np.cross(x_axis, y_axis), "short-axis normal")
    if np.dot(normal, plan.long_axis) < 0:
        y_axis = -y_axis
        normal = -normal

    if fov_mm is None:
        corners = world_corners(volume.shape, affine)
        px = corners @ x_axis
        py = corners @ y_axis
        xmin, xmax = float(px.min()), float(px.max())
        ymin, ymax = float(py.min()), float(py.max())
        width = max(2, int(np.ceil((xmax - xmin) / pixel_spacing_mm)) + 1)
        height = max(2, int(np.ceil((ymax - ymin) / pixel_spacing_mm)) + 1)
        x0, y0 = xmin, ymin
    else:
        if len(fov_mm) != 2 or min(fov_mm) <= 0:
            raise ValueError("fov_mm must contain two positive lengths")
        width = max(2, int(np.ceil(float(fov_mm[0]) / pixel_spacing_mm)) + 1)
        height = max(2, int(np.ceil(float(fov_mm[1]) / pixel_spacing_mm)) + 1)
        first_center = plan.apex + positions[0] * normal
        x0 = float(first_center @ x_axis) - (width - 1) * pixel_spacing_mm / 2.0
        y0 = float(first_center @ y_axis) - (height - 1) * pixel_spacing_mm / 2.0

    z0 = float((plan.apex + positions[0] * normal) @ normal)
    origin = x0 * x_axis + y0 * y_axis + z0 * normal
    output_affine = np.eye(4, dtype=float)
    output_affine[:3, 0] = x_axis * pixel_spacing_mm
    output_affine[:3, 1] = y_axis * pixel_spacing_mm
    output_affine[:3, 2] = normal * slice_spacing_mm
    output_affine[:3, 3] = origin

    # Resample one output slice at a time. This keeps peak coordinate-memory
    # proportional to one 2-D image instead of the complete 3-D stack.
    ii, jj = np.meshgrid(np.arange(width), np.arange(height), indexing="ij")
    planar = np.stack([ii.ravel(), jj.ravel(), np.zeros(ii.size), np.ones(ii.size)])
    source_inverse = np.linalg.inv(np.asarray(affine, float))
    output_dtype = np.float32 if interpolation_order > 0 else np.asarray(volume).dtype
    sampled = np.empty((width, height, len(positions)), dtype=output_dtype)
    for k in range(len(positions)):
        planar[2, :] = k
        world = output_affine @ planar
        source_voxels = source_inverse @ world
        sampled[:, :, k] = map_coordinates(
            np.asarray(volume),
            source_voxels[:3],
            order=interpolation_order,
            mode="constant",
            cval=0.0,
            prefilter=interpolation_order > 1,
        ).reshape(width, height)
    return StackResult(sampled, output_affine, positions)


def line_segment_for_level(
    grid: PlaneGrid,
    anchor: Iterable[float],
    axis: Iterable[float],
    level_mm: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Find the pixel-space intersection of a stack plane with a view rectangle."""
    anchor = np.asarray(anchor, float)
    axis = unit(axis)
    p00 = grid.pixel_to_world(0.0, 0.0)
    a = float(np.dot(grid.x_axis * grid.pixel_spacing, axis))
    b = float(np.dot(grid.y_axis * grid.pixel_spacing, axis))
    c = float(np.dot(p00 - anchor, axis) - level_mm)
    xmax, ymax = float(grid.width - 1), float(grid.height - 1)
    points: list[tuple[float, float]] = []
    if abs(b) > EPS:
        for x in (0.0, xmax):
            y = -(a * x + c) / b
            if -EPS <= y <= ymax + EPS:
                points.append((x, min(max(y, 0.0), ymax)))
    if abs(a) > EPS:
        for y in (0.0, ymax):
            x = -(b * y + c) / a
            if -EPS <= x <= xmax + EPS:
                points.append((min(max(x, 0.0), xmax), y))
    unique: list[tuple[float, float]] = []
    for point in points:
        if not any(np.linalg.norm(np.asarray(point) - np.asarray(old)) < 1.0e-5 for old in unique):
            unique.append(point)
    if len(unique) < 2:
        return None
    return np.asarray(unique[0]), np.asarray(unique[1])
