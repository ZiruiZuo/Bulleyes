"""Load a co-gridded short-axis image and nonzero myocardium mask."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from ..territory.models import StudyData


def load_intensity_study_nifti(
    image_path: str | Path,
    mask_path: str | Path,
    *,
    slice_axis: int = 2,
) -> StudyData:
    """Load a full SAX stack while treating every nonzero mask label as myocardium."""

    image = nib.load(str(image_path))
    mask_image = nib.load(str(mask_path))
    if image.shape != mask_image.shape:
        raise ValueError(
            f"Image shape {image.shape} does not match mask shape {mask_image.shape}"
        )
    if not np.allclose(image.affine, mask_image.affine, rtol=0.0, atol=1.0e-4):
        raise ValueError(
            "Image and mask affine matrices differ. Resample the mask onto the "
            "image grid before generating an intensity bullseye."
        )
    if slice_axis not in {0, 1, 2}:
        raise ValueError("slice_axis must be 0, 1, or 2")

    values = np.asarray(image.get_fdata(dtype=np.float32))
    mask = np.asarray(mask_image.dataobj)
    if values.ndim == 4 and values.shape[-1] == 1:
        values = values[..., 0]
        mask = mask[..., 0]
    if values.ndim != 3:
        raise ValueError(f"Expected a 3-D NIfTI volume, received shape {values.shape}")

    values = np.moveaxis(values, slice_axis, 0)
    myocardium = np.moveaxis(mask != 0, slice_axis, 0).astype(np.uint8)
    spacing = float(np.linalg.norm(image.affine[:3, slice_axis]))
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("Short-axis slice spacing must be positive and finite")
    positions = np.arange(values.shape[0], dtype=float) * spacing
    return StudyData(
        np.ascontiguousarray(values),
        np.ascontiguousarray(myocardium),
        positions,
    )
