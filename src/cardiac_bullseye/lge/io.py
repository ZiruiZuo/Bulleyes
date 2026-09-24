"""Load co-gridded short-axis LGE intensity and myocardium masks."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..territory.models import StudyData
from ..territory.aha17 import AHA_SEGMENT_NAMES


@dataclass(frozen=True)
class LgeStudyWithScar:
    """An LGE study with a co-cropped externally generated scar mask."""

    study: StudyData
    scar_mask: NDArray[np.bool_]


def _require_same_geometry(lge_image, mask_image, *, candidate_name="myocardium mask") -> None:
    """Reject geometry differences; this workflow never resamples the mask."""

    checks = (
        ("size", lge_image.GetSize(), mask_image.GetSize(), 0.0),
        ("spacing", lge_image.GetSpacing(), mask_image.GetSpacing(), 1.0e-5),
        ("origin", lge_image.GetOrigin(), mask_image.GetOrigin(), 1.0e-4),
        ("direction", lge_image.GetDirection(), mask_image.GetDirection(), 1.0e-5),
    )
    for name, lge_value, mask_value, tolerance in checks:
        if tolerance == 0.0:
            matches = tuple(lge_value) == tuple(mask_value)
        else:
            matches = np.allclose(
                np.asarray(lge_value, dtype=float),
                np.asarray(mask_value, dtype=float),
                rtol=0.0,
                atol=tolerance,
            )
        if not matches:
            raise ValueError(
                f"LGE and {candidate_name} {name} differ: "
                f"LGE={tuple(lge_value)}, mask={tuple(mask_value)}. "
                "The mask must already be on the LGE short-axis grid; no "
                "automatic resampling is performed."
            )


def _myocardium_from_image(mask_image, sitk) -> NDArray[np.bool_]:
    raw_mask = np.asarray(sitk.GetArrayFromImage(mask_image))
    if raw_mask.ndim == 4:
        return np.any(raw_mask != 0, axis=-1)
    if raw_mask.ndim == 3:
        return raw_mask != 0
    raise ValueError(f"Unsupported myocardium mask array shape: {raw_mask.shape}")


def _myocardium_slice_range(myocardium: NDArray[np.bool_]) -> tuple[int, int]:
    occupied = np.flatnonzero(np.any(myocardium, axis=(1, 2)))
    if occupied.size == 0:
        raise ValueError("The myocardium mask is empty")
    return int(occupied[0]), int(occupied[-1]) + 1


def _study_from_lge_and_myocardium(lge_image, mask_image, sitk) -> tuple[StudyData, NDArray[np.bool_], int, int]:
    if lge_image.GetDimension() != 3 or mask_image.GetDimension() != 3:
        raise ValueError(
            "LGE and myocardium mask must both be three-dimensional images"
        )
    if lge_image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError("LGE must be a scalar intensity image")
    _require_same_geometry(lge_image, mask_image)

    lge = np.asarray(sitk.GetArrayFromImage(lge_image), dtype=np.float32)
    myocardium = _myocardium_from_image(mask_image, sitk)
    if lge.shape != myocardium.shape:
        raise ValueError(
            f"LGE array shape {lge.shape} does not match mask shape {myocardium.shape}"
        )

    first_slice, last_slice = _myocardium_slice_range(myocardium)
    cropped_lge = np.ascontiguousarray(lge[first_slice:last_slice])
    cropped_myocardium = np.ascontiguousarray(
        myocardium[first_slice:last_slice], dtype=np.uint8
    )
    slice_spacing = float(lge_image.GetSpacing()[2])
    if not np.isfinite(slice_spacing) or slice_spacing <= 0.0:
        raise ValueError("LGE slice spacing must be a positive finite value")
    slice_positions = np.arange(cropped_lge.shape[0], dtype=float) * slice_spacing
    return (
        StudyData(cropped_lge, cropped_myocardium, slice_positions),
        myocardium,
        first_slice,
        last_slice,
    )


def load_lge_study(
    lge_path: str | Path,
    myocardium_mask_path: str | Path,
) -> StudyData:
    """Load an apex-to-base LGE stack and crop it to mask-containing slices.

    SimpleITK converts both NIfTI and NRRD physical metadata to its internal
    LPS convention before comparison. Array data are returned in the project's
    ``(slice, row, column)`` order. The original LGE intensity values are kept.
    """

    try:
        import SimpleITK as sitk
    except ImportError as error:
        raise ImportError(
            "LGE input requires SimpleITK. Reinstall the project with "
            "`python -m pip install -e .`."
        ) from error

    lge_path = Path(lge_path)
    mask_path = Path(myocardium_mask_path)
    for description, path in (("LGE", lge_path), ("myocardium mask", mask_path)):
        if not path.is_file():
            raise FileNotFoundError(f"{description} file does not exist: {path}")

    lge_image = sitk.ReadImage(str(lge_path))
    mask_image = sitk.ReadImage(str(mask_path))
    study, _, _, _ = _study_from_lge_and_myocardium(lge_image, mask_image, sitk)
    return study


def load_lge_study_with_scar(
    lge_path: str | Path,
    myocardium_mask_path: str | Path,
    scar_mask_path: str | Path,
) -> LgeStudyWithScar:
    """Load an LGE study and a strictly co-gridded external scar mask."""

    try:
        import SimpleITK as sitk
    except ImportError as error:
        raise ImportError(
            "LGE input requires SimpleITK. Reinstall the project with "
            "`python -m pip install -e .`."
        ) from error

    lge_path = Path(lge_path)
    myocardium_path = Path(myocardium_mask_path)
    scar_path = Path(scar_mask_path)
    for description, path in (
        ("LGE", lge_path),
        ("myocardium mask", myocardium_path),
        ("scar mask", scar_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{description} file does not exist: {path}")

    lge_image = sitk.ReadImage(str(lge_path))
    myocardium_image = sitk.ReadImage(str(myocardium_path))
    scar_image = sitk.ReadImage(str(scar_path))
    study, myocardium, first_slice, last_slice = _study_from_lge_and_myocardium(
        lge_image, myocardium_image, sitk
    )
    if scar_image.GetDimension() != 3:
        raise ValueError("scar mask must be a three-dimensional image")
    if scar_image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError("scar mask must be a scalar image")
    _require_same_geometry(lge_image, scar_image, candidate_name="scar mask")

    scar = np.asarray(sitk.GetArrayFromImage(scar_image)) != 0
    if scar.shape != myocardium.shape:
        raise ValueError(
            f"LGE array shape {myocardium.shape} does not match scar mask shape "
            f"{scar.shape}"
        )
    outside_count = int(np.count_nonzero(scar & ~myocardium))
    if outside_count:
        raise ValueError(
            f"{outside_count} scar voxels are outside the myocardium; "
            "the scar mask must be contained by the myocardium mask"
        )
    return LgeStudyWithScar(
        study=study,
        scar_mask=np.ascontiguousarray(scar[first_slice:last_slice], dtype=bool),
    )


def save_lge_bullseye_npz(path: str | Path, result) -> Path:
    """Save raw-intensity LGE bullseye arrays for downstream analysis."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        intensity=result.intensity,
        radial_positions=result.radial_positions,
        canonical_angles=result.angles,
        apex_intensity=np.asarray(result.apex_intensity, dtype=float),
    )
    return path


def save_lge_scar_mask(
    path: str | Path,
    lge_path: str | Path,
    myocardium_mask_path: str | Path,
    cropped_scar_mask,
) -> Path:
    """Restore a cropped scar mask to the complete native LGE image grid."""

    try:
        import SimpleITK as sitk
    except ImportError as error:
        raise ImportError("Saving an LGE scar mask requires SimpleITK") from error

    lge_image = sitk.ReadImage(str(lge_path))
    myocardium_image = sitk.ReadImage(str(myocardium_mask_path))
    _require_same_geometry(lge_image, myocardium_image)
    raw_myocardium = np.asarray(sitk.GetArrayFromImage(myocardium_image))
    if raw_myocardium.ndim == 4:
        myocardium = np.any(raw_myocardium != 0, axis=-1)
    elif raw_myocardium.ndim == 3:
        myocardium = raw_myocardium != 0
    else:
        raise ValueError(
            f"Unsupported myocardium mask array shape: {raw_myocardium.shape}"
        )
    occupied = np.flatnonzero(np.any(myocardium, axis=(1, 2)))
    if occupied.size == 0:
        raise ValueError("The myocardium mask is empty")
    first_slice = int(occupied[0])
    last_slice = int(occupied[-1]) + 1
    scar = np.asarray(cropped_scar_mask, dtype=bool)
    expected_shape = myocardium[first_slice:last_slice].shape
    if scar.shape != expected_shape:
        raise ValueError(
            f"Cropped scar mask shape {scar.shape} does not match expected "
            f"myocardial extent {expected_shape}"
        )

    full_scar = np.zeros(myocardium.shape, dtype=np.uint8)
    full_scar[first_slice:last_slice] = (
        scar & myocardium[first_slice:last_slice]
    ).astype(np.uint8)
    output_image = sitk.GetImageFromArray(full_scar)
    output_image.CopyInformation(lge_image)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(output_image, str(path))
    return path


def save_aha16_scar_csv(path: str | Path, summary) -> Path:
    """Write native myocardial and scar counts for AHA segments 1 through 16."""

    percentages = np.asarray(summary.scar_percentages, dtype=float)
    myocardial_counts = np.asarray(summary.myocardial_voxel_counts, dtype=np.int64)
    scar_counts = np.asarray(summary.scar_voxel_counts, dtype=np.int64)
    if percentages.shape != (16,) or myocardial_counts.shape != (16,) or scar_counts.shape != (16,):
        raise ValueError("AHA-16 scar summary arrays must each contain 16 values")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file, lineterminator="\n")
        writer.writerow(
            (
                "segment_id",
                "segment_name",
                "myocardium_voxels",
                "scar_voxels",
                "scar_percent",
            )
        )
        for segment_id in range(1, 17):
            percentage = percentages[segment_id - 1]
            writer.writerow(
                (
                    segment_id,
                    AHA_SEGMENT_NAMES[segment_id],
                    int(myocardial_counts[segment_id - 1]),
                    int(scar_counts[segment_id - 1]),
                    "" if not np.isfinite(percentage) else f"{percentage:.6f}",
                )
            )
    return path
