"""Map a registered 3-D coronary centreline mask onto the bullseye plot."""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import nibabel as nib
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage
from skimage.morphology import skeletonize

from ..territory.geometry import normalize_six_segment_relative_angles
from ..territory.models import LandmarkPair, StudyData


@dataclass(frozen=True)
class CardiacGeometry:
    true_apex_world: NDArray[np.float64]
    base_center_world: NDArray[np.float64]
    long_axis: NDArray[np.float64]
    short_axis_x: NDArray[np.float64]
    short_axis_y: NDArray[np.float64]
    short_axis_affine: NDArray[np.float64]
    short_axis_shape: tuple[int, int, int]
    slice_positions_mm: NDArray[np.float64]
    base_height_mm: float
    short_axis_image_path: Path | None = None

    def __post_init__(self) -> None:
        for name in (
            "true_apex_world",
            "base_center_world",
            "long_axis",
            "short_axis_x",
            "short_axis_y",
        ):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError(f"{name} must contain three finite values")
            object.__setattr__(self, name, value)
        affine = np.asarray(self.short_axis_affine, dtype=float)
        if affine.shape != (4, 4) or not np.isfinite(affine).all():
            raise ValueError("short_axis_affine must be a finite 4x4 matrix")
        object.__setattr__(self, "short_axis_affine", affine)
        positions = np.asarray(self.slice_positions_mm, dtype=float)
        if positions.ndim != 1 or positions.size != self.short_axis_shape[2]:
            raise ValueError("slice_positions_mm does not match short_axis_shape")
        if positions.size > 1 and not np.all(np.diff(positions) > 0):
            raise ValueError("slice positions must increase from apex to base")
        object.__setattr__(self, "slice_positions_mm", positions)
        if self.base_height_mm <= 0:
            raise ValueError("base_height_mm must be positive")


@dataclass(frozen=True)
class CoronaryPath:
    label: int
    world_points: NDArray[np.float64]


@dataclass(frozen=True)
class ProjectedCoronaryPath:
    label: int
    xy: NDArray[np.float64]


def _unit(vector: ArrayLike, name: str) -> NDArray[np.float64]:
    value = np.asarray(vector, dtype=float)
    length = float(np.linalg.norm(value))
    if value.shape != (3,) or not np.isfinite(value).all() or length <= 1.0e-8:
        raise ValueError(f"{name} is not a valid 3-D direction")
    return value / length


def load_cardiac_geometry(path: str | Path) -> CardiacGeometry:
    """Load the JSON spatial contract written by Convert2ShortAxis."""

    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "convert2shortaxis.geometry":
        raise ValueError("Geometry JSON was not produced by Convert2ShortAxis")
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("Unsupported Convert2ShortAxis geometry schema version")

    apex = np.asarray(payload["true_apex_world_mm"], dtype=float)
    base = np.asarray(payload["selected_base_center_world_mm"], dtype=float)
    long_axis = _unit(payload["long_axis_apex_to_base"], "long axis")
    base_height = float(np.dot(base - apex, long_axis))
    recorded_height = float(payload["base_plane_offset_mm"])
    if not np.isclose(base_height, recorded_height, atol=1.0e-3, rtol=1.0e-5):
        raise ValueError("Geometry JSON contains inconsistent base coordinates")

    image_path_value = payload.get("short_axis_image_path")
    return CardiacGeometry(
        true_apex_world=apex,
        base_center_world=base,
        long_axis=long_axis,
        short_axis_x=_unit(payload["short_axis_x_world"], "short-axis x"),
        short_axis_y=_unit(payload["short_axis_y_world"], "short-axis y"),
        short_axis_affine=np.asarray(payload["short_axis_affine"], dtype=float),
        short_axis_shape=tuple(int(v) for v in payload["short_axis_shape"]),
        slice_positions_mm=np.asarray(
            payload["slice_positions_from_true_apex_mm"], dtype=float
        ),
        base_height_mm=base_height,
        short_axis_image_path=(
            Path(image_path_value).expanduser().resolve()
            if image_path_value
            else None
        ),
    )


def validate_geometry_against_short_axis_nifti(
    geometry: CardiacGeometry,
    short_axis_mr_path: str | Path,
) -> None:
    image = nib.load(str(Path(short_axis_mr_path)))
    if tuple(image.shape) != geometry.short_axis_shape:
        raise ValueError(
            f"Geometry expects short-axis shape {geometry.short_axis_shape}, "
            f"but MR has {image.shape}"
        )
    if not np.allclose(image.affine, geometry.short_axis_affine, atol=1.0e-4, rtol=1.0e-5):
        raise ValueError("Geometry JSON and short-axis MR affine matrices differ")


def bullseye_slice_radii(geometry: CardiacGeometry) -> NDArray[np.float64]:
    """Map SAX slice heights to Termeer-style bullseye radii 0..4."""

    return 4.0 * geometry.slice_positions_mm / geometry.base_height_mm


def _voxel_graph_paths(skeleton: NDArray[np.bool_]) -> list[NDArray[np.int32]]:
    coordinates = np.argwhere(skeleton)
    if coordinates.size == 0:
        return []
    lookup = {tuple(point): index for index, point in enumerate(coordinates)}
    offsets = [
        np.asarray(offset, dtype=int)
        for offset in np.ndindex(3, 3, 3)
        if offset != (1, 1, 1)
    ]
    offsets = [offset - 1 for offset in offsets]
    neighbours: list[set[int]] = [set() for _ in range(len(coordinates))]

    def has_shorter_bridge(point: NDArray[np.int64], offset: NDArray[np.int64]) -> bool:
        """Avoid diagonal graph edges that cut across an occupied voxel corner."""

        active_axes = np.flatnonzero(offset)
        if active_axes.size < 2:
            return False
        for count in range(1, active_axes.size):
            for selected_axes in combinations(active_axes.tolist(), count):
                intermediate = point.copy()
                intermediate[list(selected_axes)] += offset[list(selected_axes)]
                if tuple(intermediate) in lookup:
                    return True
        return False

    for index, point in enumerate(coordinates):
        for offset in offsets:
            neighbour = lookup.get(tuple(point + offset))
            if (
                neighbour is not None
                and neighbour != index
                and not has_shorter_bridge(point, offset)
            ):
                neighbours[index].add(neighbour)

    visited: set[tuple[int, int]] = set()

    def edge(a: int, b: int) -> tuple[int, int]:
        return (a, b) if a < b else (b, a)

    def trace(start: int, following: int) -> list[int]:
        path = [start]
        previous, current = start, following
        visited.add(edge(previous, current))
        while True:
            path.append(current)
            if len(neighbours[current]) != 2:
                break
            candidates = [value for value in neighbours[current] if value != previous]
            if not candidates:
                break
            next_node = candidates[0]
            if edge(current, next_node) in visited:
                break
            previous, current = current, next_node
            visited.add(edge(previous, current))
        return path

    paths: list[list[int]] = []
    critical = [index for index, values in enumerate(neighbours) if len(values) != 2]
    for start in critical:
        for following in neighbours[start]:
            if edge(start, following) not in visited:
                paths.append(trace(start, following))

    # Any remaining edges belong to closed cycles without endpoints.
    for start, values in enumerate(neighbours):
        for following in values:
            if edge(start, following) not in visited:
                paths.append(trace(start, following))
    return [coordinates[np.asarray(path, dtype=int)].astype(np.int32) for path in paths]


def _polyline_length(points: NDArray[np.float64]) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _resample_polyline(
    points: NDArray[np.float64], step_mm: float
) -> NDArray[np.float64]:
    if len(points) < 2 or step_mm <= 0:
        return points
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(distances)])
    keep = np.concatenate([[True], np.diff(cumulative) > 1.0e-8])
    points = points[keep]
    cumulative = cumulative[keep]
    if len(points) < 2 or cumulative[-1] <= step_mm:
        return points
    samples = np.linspace(0.0, cumulative[-1], int(np.ceil(cumulative[-1] / step_mm)) + 1)
    return np.column_stack(
        [np.interp(samples, cumulative, points[:, dimension]) for dimension in range(3)]
    )


def load_coronary_paths_from_nifti(
    path: str | Path,
    *,
    mask_is_centerline: bool = True,
    min_path_length_mm: float = 2.0,
    resample_step_mm: float = 0.75,
    smoothing_sigma_voxels: float = 0.0,
) -> list[CoronaryPath]:
    """Extract ordered 3-D polylines from a binary or labelled NIfTI mask."""

    image = nib.load(str(Path(path)))
    raw = np.asarray(image.dataobj)
    if raw.ndim != 3:
        raise ValueError("Coronary mask must be a 3-D NIfTI")
    rounded = np.rint(raw)
    if not np.allclose(raw, rounded, atol=1.0e-4, rtol=0.0):
        raise ValueError("Coronary mask contains non-integer values")
    labels = rounded.astype(np.int32)
    output: list[CoronaryPath] = []
    source_labels = sorted(int(value) for value in np.unique(labels) if value != 0)
    binary_input = len(source_labels) == 1
    next_path_label = 1

    for label in source_labels:
        binary = labels == label
        if not mask_is_centerline:
            binary = np.asarray(skeletonize(binary), dtype=bool)
        else:
            binary = np.asarray(binary, dtype=bool)
        if not binary.any():
            continue
        components, component_count = ndimage.label(binary, structure=np.ones((3, 3, 3)))
        for component_index in range(1, component_count + 1):
            component = components == component_index
            for voxel_path in _voxel_graph_paths(component):
                homogeneous = np.column_stack(
                    [voxel_path.astype(float), np.ones(len(voxel_path))]
                )
                world = (np.asarray(image.affine, dtype=float) @ homogeneous.T).T[:, :3]
                if smoothing_sigma_voxels > 0 and len(world) >= 4:
                    smoothed = ndimage.gaussian_filter1d(
                        world, sigma=smoothing_sigma_voxels, axis=0, mode="nearest"
                    )
                    smoothed[0], smoothed[-1] = world[0], world[-1]
                    world = smoothed
                world = _resample_polyline(world, resample_step_mm)
                if _polyline_length(world) >= min_path_length_mm:
                    # A multi-label mask already carries vessel identities. For
                    # a binary mask, assign a separate display label to each
                    # extracted branch/path so it can receive its own colour.
                    display_label = next_path_label if binary_input else label
                    output.append(
                        CoronaryPath(label=display_label, world_points=world)
                    )
                    if binary_input:
                        next_path_label += 1
    if not output:
        raise ValueError("No usable coronary centreline paths were found")
    return output


@dataclass(frozen=True)
class _AngularFrames:
    heights: NDArray[np.float64]
    zero_angles: NDArray[np.float64]
    separations: NDArray[np.float64]
    signs: NDArray[np.float64]


def _landmark_world_point(
    geometry: CardiacGeometry,
    slice_index: int,
    point_xy: tuple[float, float],
) -> NDArray[np.float64]:
    # Internal arrays are (slice, row=voxel axis 0, column=voxel axis 1), so
    # stored annotation (x, y) maps back to NIfTI voxel (y, x, slice).
    voxel = np.asarray([point_xy[1], point_xy[0], float(slice_index), 1.0])
    return (geometry.short_axis_affine @ voxel)[:3]


def _raw_long_axis_angle(
    world_point: NDArray[np.float64], geometry: CardiacGeometry
) -> float:
    displacement = world_point - geometry.true_apex_world
    height = float(np.dot(displacement, geometry.long_axis))
    shortest = displacement - height * geometry.long_axis
    # Match the existing image angle atan2(row, column): internal column points
    # along SAX affine axis 1 and internal row along SAX affine axis 0.
    return float(
        np.arctan2(
            np.dot(shortest, geometry.short_axis_x),
            np.dot(shortest, geometry.short_axis_y),
        )
    )


def _build_angular_frames(
    geometry: CardiacGeometry,
    study: StudyData,
    landmarks: list[LandmarkPair | None],
) -> _AngularFrames:
    if len(landmarks) != study.slice_count:
        raise ValueError("Landmark count does not match the short-axis study")
    if study.slice_count != len(geometry.slice_positions_mm):
        raise ValueError("Geometry slice count does not match the short-axis study")

    valid_heights: list[float] = []
    zero_angles: list[float] = []
    separations: list[float] = []
    signs: list[float] = []
    for slice_index, pair in enumerate(landmarks):
        if pair is None:
            continue
        aip_angle = _raw_long_axis_angle(
            _landmark_world_point(geometry, slice_index, pair.aip), geometry
        )
        iip_angle = _raw_long_axis_angle(
            _landmark_world_point(geometry, slice_index, pair.iip), geometry
        )
        difference = float(
            np.arctan2(np.sin(iip_angle - aip_angle), np.cos(iip_angle - aip_angle))
        )
        if np.isclose(difference, 0.0):
            continue
        valid_heights.append(float(geometry.slice_positions_mm[slice_index]))
        zero_angles.append(aip_angle + difference / 2.0)
        separations.append(abs(difference))
        signs.append(1.0 if difference >= 0 else -1.0)
    if not valid_heights:
        raise ValueError("No valid AIP/IIP pair is available for coronary mapping")
    order = np.argsort(valid_heights)
    return _AngularFrames(
        heights=np.asarray(valid_heights)[order],
        zero_angles=np.unwrap(np.asarray(zero_angles)[order]),
        separations=np.asarray(separations)[order],
        signs=np.asarray(signs)[order],
    )


def _nearest_values(
    query: NDArray[np.float64], positions: NDArray[np.float64], values: NDArray[np.float64]
) -> NDArray[np.float64]:
    insertion = np.searchsorted(positions, query)
    right = np.clip(insertion, 0, len(positions) - 1)
    left = np.clip(insertion - 1, 0, len(positions) - 1)
    choose_right = np.abs(query - positions[right]) < np.abs(query - positions[left])
    return values[np.where(choose_right, right, left)]


def _split_valid_polyline(
    points: NDArray[np.float64], valid: NDArray[np.bool_]
) -> list[NDArray[np.float64]]:
    paths: list[NDArray[np.float64]] = []
    start: int | None = None
    for index, is_valid in enumerate(np.append(valid, False)):
        if is_valid and start is None:
            start = index
        elif not is_valid and start is not None:
            if index - start >= 2:
                paths.append(points[start:index])
            start = None
    return paths


def project_coronary_paths(
    paths: list[CoronaryPath],
    geometry: CardiacGeometry,
    study: StudyData,
    landmarks: list[LandmarkPair | None],
    *,
    six_segment_warp: bool = True,
) -> list[ProjectedCoronaryPath]:
    """Project registered 3-D coronary paths to the current bullseye frame."""

    frames = _build_angular_frames(geometry, study, landmarks)
    projected: list[ProjectedCoronaryPath] = []
    for path in paths:
        points = np.asarray(path.world_points, dtype=float)
        displacement = points - geometry.true_apex_world[None, :]
        heights = displacement @ geometry.long_axis
        shortest = displacement - heights[:, None] * geometry.long_axis[None, :]
        raw_angles = np.arctan2(
            shortest @ geometry.short_axis_x,
            shortest @ geometry.short_axis_y,
        )

        zero = np.interp(heights, frames.heights, frames.zero_angles)
        separation = np.interp(heights, frames.heights, frames.separations)
        signs = _nearest_values(heights, frames.heights, frames.signs)
        canonical = np.mod(raw_angles - zero, 2.0 * np.pi)
        canonical = np.where(signs < 0, np.mod(-canonical, 2.0 * np.pi), canonical)
        if six_segment_warp:
            canonical = normalize_six_segment_relative_angles(canonical, separation)

        radius = 4.0 * heights / geometry.base_height_mm
        plot_angles = canonical + np.pi
        xy = np.column_stack(
            [radius * np.cos(plot_angles), radius * np.sin(plot_angles)]
        )
        # Keep points above the selected basal plane. Their h/H ratio is > 1,
        # hence their bullseye radius naturally extends beyond the outer ring.
        # Points below the true apical plane remain excluded because a negative
        # polar radius would mirror their direction by 180 degrees.
        valid = np.isfinite(xy).all(axis=1) & (heights >= 0.0)
        for segment in _split_valid_polyline(xy, valid):
            projected.append(ProjectedCoronaryPath(label=path.label, xy=segment))
    if not projected:
        raise ValueError("No coronary centreline point lies above the apical plane")
    return projected
