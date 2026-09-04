"""NIfTI loading/saving helpers with actionable dependency errors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


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
