"""Compute exact Euclidean coronary distances in NIfTI world coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize


@dataclass(frozen=True)
class DistanceVolumeResult:
    """Distance-to-nearest-vessel data on the original myocardium grid."""

    distance_mm: NDArray[np.float32]
    nearest_vessel_label: NDArray[np.int16]
    projected_vessel_labels: NDArray[np.int16]
    affine: NDArray[np.float64]
    header: object
    vessel_labels: tuple[int, ...]


def _integer_labels(image: nib.spatialimages.SpatialImage, name: str) -> NDArray[np.int32]:
    raw = np.asarray(image.dataobj)
    if raw.ndim != 3:
        raise ValueError(f"{name} must be a 3-D NIfTI, received {raw.shape}")
    rounded = np.rint(raw)
    if not np.allclose(raw, rounded, atol=1.0e-4, rtol=0.0):
        raise ValueError(f"{name} contains non-integer values")
    return rounded.astype(np.int32)


def _voxel_points_to_world(
    voxel_points: NDArray[np.integer], affine: NDArray[np.float64]
) -> NDArray[np.float64]:
    homogeneous = np.column_stack(
        [np.asarray(voxel_points, dtype=float), np.ones(len(voxel_points))]
    )
    return (affine @ homogeneous.T).T[:, :3]


def compute_distance_volume_nifti(
    coronary_path: str | Path,
    myocardium_path: str | Path,
    *,
    vessel_labels: tuple[int, ...] = (2, 3, 4),
    coronary_is_centerline: bool = False,
) -> DistanceVolumeResult:
    """Project each coronary to myocardium and take its minimum 3-D distance.

    Nearest-neighbour searches use physical NIfTI world coordinates, so the
    returned distances are millimetres even for oblique or anisotropic grids.
    The distance is straight-line 3-D Euclidean distance, matching the EDT
    definition in the reference ``territory.py`` implementation.
    """

    if not vessel_labels or len(set(vessel_labels)) != len(vessel_labels):
        raise ValueError("vessel_labels must contain distinct non-zero labels")
    if any(int(label) == 0 for label in vessel_labels):
        raise ValueError("0 is reserved for background")

    coronary_image = nib.load(str(Path(coronary_path)))
    myocardium_image = nib.load(str(Path(myocardium_path)))
    coronary = _integer_labels(coronary_image, "Coronary mask")
    myocardium_labels = _integer_labels(myocardium_image, "Myocardium mask")
    myocardium = myocardium_labels > 0
    myocardial_voxels = np.argwhere(myocardium)
    if len(myocardial_voxels) == 0:
        raise ValueError("Myocardium mask is empty")

    myocardium_affine = np.asarray(myocardium_image.affine, dtype=float)
    coronary_affine = np.asarray(coronary_image.affine, dtype=float)
    myocardial_world = _voxel_points_to_world(myocardial_voxels, myocardium_affine)
    myocardial_tree = cKDTree(myocardial_world)

    best_distance = np.full(len(myocardial_voxels), np.inf, dtype=float)
    best_label = np.zeros(len(myocardial_voxels), dtype=np.int16)
    projected_volume = np.zeros(myocardium.shape, dtype=np.int16)
    used_labels: list[int] = []

    for label in vessel_labels:
        centerline = coronary == int(label)
        if not centerline.any():
            continue
        if not coronary_is_centerline:
            centerline = np.asarray(skeletonize(centerline), dtype=bool)
        coronary_voxels = np.argwhere(centerline)
        if len(coronary_voxels) == 0:
            continue
        coronary_world = _voxel_points_to_world(coronary_voxels, coronary_affine)

        # This is the reference algorithm's vessel-to-LV projection, performed
        # in world millimetres rather than only using voxel spacing.
        _, projected_indices = myocardial_tree.query(coronary_world, k=1)
        projected_indices = np.unique(np.asarray(projected_indices, dtype=int))
        projected_world = myocardial_world[projected_indices]
        projected_voxels = myocardial_voxels[projected_indices]
        projected_volume[tuple(projected_voxels.T)] = int(label)

        # Compute this vessel separately, then update the voxelwise minimum.
        vessel_tree = cKDTree(projected_world)
        vessel_distance, _ = vessel_tree.query(myocardial_world, k=1)
        closer = vessel_distance < best_distance
        best_distance[closer] = vessel_distance[closer]
        best_label[closer] = int(label)
        used_labels.append(int(label))

    if not used_labels:
        present = sorted(int(v) for v in np.unique(coronary) if v != 0)
        raise ValueError(
            f"None of vessel_labels={tuple(vessel_labels)} is present in the "
            f"coronary mask; present non-zero labels are {present}"
        )

    distance_volume = np.full(myocardium.shape, np.nan, dtype=np.float32)
    nearest_volume = np.zeros(myocardium.shape, dtype=np.int16)
    distance_volume[tuple(myocardial_voxels.T)] = best_distance.astype(np.float32)
    nearest_volume[tuple(myocardial_voxels.T)] = best_label
    return DistanceVolumeResult(
        distance_mm=distance_volume,
        nearest_vessel_label=nearest_volume,
        projected_vessel_labels=projected_volume,
        affine=myocardium_affine,
        header=myocardium_image.header.copy(),
        vessel_labels=tuple(used_labels),
    )


def resample_distance_to_short_axis(
    distance_mm: NDArray[np.floating],
    source_affine: NDArray[np.floating],
    target_shape: tuple[int, int, int],
    target_affine: NDArray[np.floating],
    *,
    target_myocardium: NDArray[np.bool_] | None = None,
) -> NDArray[np.float32]:
    """Linearly resample a scalar distance volume onto the SAX NIfTI grid."""

    source = np.asarray(distance_mm, dtype=float)
    if source.ndim != 3:
        raise ValueError("distance_mm must be a 3-D array")
    if len(target_shape) != 3:
        raise ValueError("target_shape must contain three dimensions")
    if target_myocardium is not None and target_myocardium.shape != target_shape:
        raise ValueError("target_myocardium shape does not match target_shape")

    target_voxels = np.indices(target_shape, dtype=float).reshape(3, -1)
    homogeneous = np.vstack([target_voxels, np.ones(target_voxels.shape[1])])
    source_voxels = (
        np.linalg.inv(np.asarray(source_affine, dtype=float))
        @ np.asarray(target_affine, dtype=float)
        @ homogeneous
    )[:3]

    valid_source = np.isfinite(source)
    numerator = ndimage.map_coordinates(
        np.where(valid_source, source, 0.0),
        source_voxels,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    weight = ndimage.map_coordinates(
        valid_source.astype(float),
        source_voxels,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    sampled = np.divide(
        numerator,
        weight,
        out=np.full_like(numerator, np.nan),
        where=weight > 1.0e-6,
    ).reshape(target_shape)
    if target_myocardium is not None:
        sampled[~np.asarray(target_myocardium, dtype=bool)] = np.nan
    return sampled.astype(np.float32)


def save_distance_nifti(
    path: str | Path,
    distance_mm: NDArray[np.floating],
    affine: NDArray[np.floating],
    header: object | None = None,
) -> Path:
    """Save a floating-point distance map whose finite values are millimetres."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    output_header = header.copy() if header is not None else None
    if output_header is not None:
        output_header.set_data_dtype(np.float32)
    image = nib.Nifti1Image(
        np.asarray(distance_mm, dtype=np.float32),
        np.asarray(affine, dtype=float),
        header=output_header,
    )
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=1)
    nib.save(image, str(path))
    return path
