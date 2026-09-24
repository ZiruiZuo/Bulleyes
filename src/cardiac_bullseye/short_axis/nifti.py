"""NIfTI loading/saving helpers with actionable dependency errors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import unit


@dataclass
class NiftiVolume:
    data: np.ndarray
    affine: np.ndarray
    header: object
    path: Path


def _nibabel():
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError(
            "nibabel is required for NIfTI I/O. Install the project with "
            "'python -m pip install -e E:\\ZZR\\Code\\Tools\\Bulleyes'."
        ) from exc
    return nib


def load_nifti(path: str | Path, dtype=np.float32) -> NiftiVolume:
    nib = _nibabel()
    path = Path(path).expanduser().resolve()
    image = nib.load(str(path))
    if len(image.shape) != 3:
        raise ValueError(f"only 3-D NIfTI images are supported, got shape {image.shape}")
    data = np.asarray(image.dataobj, dtype=dtype)
    return NiftiVolume(data, np.asarray(image.affine, float), image.header.copy(), path)


def constrained_reference_x_axis(
    reference_affine: np.ndarray,
    long_axis: np.ndarray,
    *,
    search_step_deg: float = 0.05,
) -> np.ndarray:
    """Return the closest in-plane axis that keeps the reference view codes.

    The through-plane direction remains ``long_axis``.  A small roll around
    that axis is allowed because Slicer assigns oblique voxel axes to its
    coloured views using a discrete orientation code.  Near an assignment
    boundary, the direct projection can otherwise move the same anatomical
    view to a different pane after a new Apex/Base selection.
    """

    nib = _nibabel()
    affine = np.asarray(reference_affine, dtype=float)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError("reference_affine must be a finite 4x4 matrix")
    if not np.isfinite(search_step_deg) or search_step_deg <= 0:
        raise ValueError("search_step_deg must be positive")

    z_axis = unit(long_axis, "LV long axis")
    reference_x = unit(affine[:3, 0], "reference voxel axis 0")
    projected = reference_x - np.dot(reference_x, z_axis) * z_axis
    base_x = unit(projected, "reference short-axis x axis")
    base_y = unit(np.cross(z_axis, base_x), "reference short-axis y axis")
    target_codes = nib.aff2axcodes(affine)

    step_count = int(np.ceil(180.0 / float(search_step_deg)))
    offsets = [0.0]
    for index in range(1, step_count + 1):
        angle = min(index * float(search_step_deg), 180.0)
        offsets.extend((angle, -angle))

    for offset_deg in offsets:
        angle = np.deg2rad(offset_deg)
        candidate_x = np.cos(angle) * base_x + np.sin(angle) * base_y
        candidate_y = np.cross(z_axis, candidate_x)
        candidate = np.eye(4, dtype=float)
        candidate[:3, :3] = np.column_stack((candidate_x, candidate_y, z_axis))
        if nib.aff2axcodes(candidate) == target_codes:
            return unit(candidate_x, "matched reference short-axis x axis")

    raise ValueError(
        "Cannot preserve the selected LV long axis while matching the "
        f"orientation-reference view codes {target_codes}"
    )


def voxel_grid_center_world(
    shape: tuple[int, int, int], affine: np.ndarray
) -> np.ndarray:
    """Return the physical centre of a three-dimensional voxel-centre grid."""

    sizes = np.asarray(shape, dtype=float)
    matrix = np.asarray(affine, dtype=float)
    if sizes.shape != (3,) or np.any(sizes < 1):
        raise ValueError("shape must contain three positive dimensions")
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("affine must be a finite 4x4 matrix")
    voxel = np.concatenate(((sizes - 1.0) / 2.0, [1.0]))
    return (matrix @ voxel)[:3]


def recenter_affine_to_reference(
    affine: np.ndarray,
    shape: tuple[int, int, int],
    reference_affine: np.ndarray,
    reference_shape: tuple[int, int, int],
) -> np.ndarray:
    """Translate an affine so its grid centre equals the reference centre."""

    result = np.asarray(affine, dtype=float).copy()
    source_center = voxel_grid_center_world(shape, result)
    reference_center = voxel_grid_center_world(reference_shape, reference_affine)
    result[:3, 3] += reference_center - source_center
    return result


def validate_mask_grid(image: NiftiVolume, mask: NiftiVolume) -> None:
    if image.data.shape != mask.data.shape or not np.allclose(
        image.affine, mask.affine, atol=1.0e-4, rtol=1.0e-5
    ):
        raise ValueError(
            "mask must be on the same voxel grid as the image (same shape and affine); "
            "resample the mask to the image grid first"
        )


def save_nifti(
    path: str | Path,
    data: np.ndarray,
    affine: np.ndarray,
    reference_header: object | None = None,
) -> Path:
    nib = _nibabel()
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    header = reference_header.copy() if reference_header is not None else None
    image = nib.Nifti1Image(data, np.asarray(affine, float), header=header)
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=1)
    nib.save(image, str(path))
    return path
