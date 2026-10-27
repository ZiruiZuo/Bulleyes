from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from .aha17 import AHA_SEGMENT_NAMES
from .mapping import (
    TERRITORY_COLORS,
    TERRITORY_NAMES,
    compute_aha17_territory_percentages,
)
from .models import BullseyeResult, LandmarkPair, StudyData


def load_study_npz(path: str | Path) -> StudyData:
    """Load a study stored with the project's slice-first NPZ convention."""

    path = Path(path)
    with np.load(path) as data:
        mr = np.asarray(data["mr"], dtype=float)
        territory_mask = np.asarray(data["territory_mask"], dtype=np.uint8)
        if "slice_positions" in data:
            slice_positions = np.asarray(data["slice_positions"], dtype=float)
        else:
            slice_positions = np.arange(mr.shape[0], dtype=float)

    return StudyData(mr, territory_mask, slice_positions)


def load_study_mat(
    path: str | Path,
    *,
    mr_key: str = "mr",
    territory_key: str = "territory_mask",
    slice_positions_key: str = "slice_positions",
    axis_order: str = "hws",
) -> StudyData:
    """Load a traditional MATLAB MAT file (not MATLAB v7.3/HDF5).

    ``axis_order='hws'`` converts MATLAB ``(height, width, slice)`` arrays to
    the project's ``(slice, height, width)`` convention. Use ``'shw'`` if the
    MAT file is already slice-first.
    """

    path = Path(path)
    payload = loadmat(path)
    if mr_key not in payload or territory_key not in payload:
        available = sorted(key for key in payload if not key.startswith("__"))
        raise KeyError(
            f"Expected keys {mr_key!r} and {territory_key!r}; available keys: "
            f"{available}"
        )

    mr = np.asarray(payload[mr_key], dtype=float)
    territory_mask = np.asarray(payload[territory_key], dtype=np.uint8)
    if axis_order == "hws":
        mr = np.moveaxis(mr, -1, 0)
        territory_mask = np.moveaxis(territory_mask, -1, 0)
    elif axis_order != "shw":
        raise ValueError("axis_order must be 'hws' or 'shw'")

    if slice_positions_key in payload:
        slice_positions = np.asarray(
            payload[slice_positions_key], dtype=float
        ).squeeze()
    else:
        slice_positions = np.arange(mr.shape[0], dtype=float)
    return StudyData(mr, territory_mask, slice_positions)


def load_study(
    path: str | Path,
    *,
    mr_key: str = "mr",
    territory_key: str = "territory_mask",
    slice_positions_key: str = "slice_positions",
    axis_order: str = "hws",
) -> StudyData:
    """Load NPZ or MAT input based on its file extension."""

    path = Path(path)
    if path.suffix.lower() == ".npz":
        return load_study_npz(path)
    if path.suffix.lower() == ".mat":
        return load_study_mat(
            path,
            mr_key=mr_key,
            territory_key=territory_key,
            slice_positions_key=slice_positions_key,
            axis_order=axis_order,
        )
    raise ValueError("Input must be an .npz or traditional .mat file")


def load_study_nifti(
    mr_path: str | Path,
    territory_mask_path: str | Path,
    *,
    slice_axis: int = 2,
    reverse_slice_order: bool = False,
    territory_label_map: dict[int, int] | None = None,
) -> StudyData:
    """Load co-registered 3-D MR and territory-mask NIfTI files.

    NIfTI data are moved to internal ``(slice, row, column)`` order. The code
    deliberately does not apply an automatic anatomical reorientation because
    AIP/IIP are selected on the displayed planes. ``reverse_slice_order`` must
    be set so the resulting stack runs from apex to base.
    """

    try:
        import nibabel as nib
    except ImportError as error:
        raise ImportError(
            "NIfTI input requires nibabel. Install the project dependencies "
            "once with `pip install -e .`, or install nibabel in the active "
            "Python environment."
        ) from error

    mr_path = Path(mr_path)
    territory_mask_path = Path(territory_mask_path)
    mr_image = nib.load(str(mr_path))
    mask_image = nib.load(str(territory_mask_path))

    if mr_image.shape != mask_image.shape:
        raise ValueError(
            f"MR shape {mr_image.shape} does not match mask shape "
            f"{mask_image.shape}"
        )
    if not np.allclose(mr_image.affine, mask_image.affine, rtol=0.0, atol=1e-4):
        raise ValueError(
            "MR and mask affine matrices differ. Resample the mask onto the MR "
            "grid before running this project."
        )

    mr = np.asarray(mr_image.get_fdata(dtype=np.float32))
    raw_mask = np.asarray(mask_image.dataobj)
    if mr.ndim == 4 and mr.shape[-1] == 1:
        mr = mr[..., 0]
        raw_mask = raw_mask[..., 0]
    if mr.ndim != 3:
        raise ValueError(f"Expected a 3-D NIfTI volume, received shape {mr.shape}")
    if slice_axis not in {0, 1, 2}:
        raise ValueError("slice_axis must be 0, 1, or 2")

    rounded_mask = np.rint(raw_mask)
    if not np.allclose(raw_mask, rounded_mask, rtol=0.0, atol=1e-4):
        raise ValueError("The territory mask contains non-integer label values")
    rounded_mask = rounded_mask.astype(np.int32)

    if territory_label_map is None:
        territory_label_map = {1: 1, 2: 2, 3: 3}
    mapped_mask = np.zeros(rounded_mask.shape, dtype=np.uint8)
    known_source_labels = {0}
    for source_label, canonical_label in territory_label_map.items():
        if canonical_label not in {1, 2, 3}:
            raise ValueError("Mapped territory labels must be 1=LAD, 2=LCX, 3=RCA")
        mapped_mask[rounded_mask == int(source_label)] = int(canonical_label)
        known_source_labels.add(int(source_label))

    present_labels = set(np.unique(rounded_mask).tolist())
    unknown_labels = present_labels - known_source_labels
    if unknown_labels:
        raise ValueError(
            f"Mask contains labels absent from TERRITORY_LABEL_MAP: "
            f"{sorted(unknown_labels)}"
        )
    canonical_present = set(np.unique(mapped_mask).tolist()) - {0}
    if len(canonical_present) < 2:
        raise ValueError(
            "The supplied mask contains fewer than two territory classes. A "
            "binary myocardium mask cannot produce an LAD/LCX/RCA overlay."
        )

    mr = np.moveaxis(mr, slice_axis, 0)
    mapped_mask = np.moveaxis(mapped_mask, slice_axis, 0)
    if reverse_slice_order:
        mr = mr[::-1]
        mapped_mask = mapped_mask[::-1]

    slice_spacing = float(np.linalg.norm(mr_image.affine[:3, slice_axis]))
    if not np.isfinite(slice_spacing) or slice_spacing <= 0:
        slice_spacing = 1.0
    slice_positions = np.arange(mr.shape[0], dtype=float) * slice_spacing
    return StudyData(
        np.ascontiguousarray(mr),
        np.ascontiguousarray(mapped_mask),
        slice_positions,
    )


def save_landmarks_json(
    path: str | Path,
    landmarks: list[LandmarkPair | None],
    *,
    deleted_slice_indices: Iterable[int] = (),
    image_shape: tuple[int, ...] | None = None,
    center_overrides: Iterable[tuple[float, float] | None] | None = None,
    annotation_mode: str | None = None,
    reference_slice_index: int | None = None,
) -> None:
    """Save landmarks atomically enough for an interactive annotation session."""

    deleted = {int(index) for index in deleted_slice_indices}
    invalid = sorted(index for index in deleted if index < 0 or index >= len(landmarks))
    if invalid:
        raise ValueError(f"Deleted slice indices are out of range: {invalid}")
    if annotation_mode not in {None, "per-slice", "single-reference"}:
        raise ValueError("Unknown landmark annotation mode")
    if reference_slice_index is not None:
        reference_slice_index = int(reference_slice_index)
        if not 0 <= reference_slice_index < len(landmarks):
            raise ValueError("reference_slice_index is outside the landmark stack")
        if reference_slice_index in deleted:
            raise ValueError("The reference slice cannot be deleted")
    centers = (
        [None] * len(landmarks)
        if center_overrides is None
        else list(center_overrides)
    )
    if len(centers) != len(landmarks):
        raise ValueError("center_overrides must have one entry per slice")

    normalized_centers: list[tuple[float, float] | None] = []
    for index, center in enumerate(centers):
        if center is None or index in deleted:
            normalized_centers.append(None)
            continue
        values = np.asarray(center, dtype=float)
        if values.shape != (2,) or not np.isfinite(values).all():
            raise ValueError("Each center override must contain two finite values")
        x, y = map(float, values)
        if image_shape is not None:
            height, width = image_shape[1:]
            if not (0.0 <= x < width and 0.0 <= y < height):
                raise ValueError(
                    f"Center override on slice {index + 1} is outside the "
                    f"current image shape {tuple(image_shape)}"
                )
        normalized_centers.append((x, y))

    slices = []
    for index, pair in enumerate(landmarks):
        center = normalized_centers[index]
        if index in deleted:
            item = {"status": "deleted"}
        elif pair is None:
            item = None if center is None else {
                "status": "skipped",
                "center": list(center),
            }
        else:
            item = {"aip": list(pair.aip), "iip": list(pair.iip)}
            if center is not None:
                item["center"] = list(center)
        slices.append(item)

    payload = {
        "coordinate_convention": "zero-based (x, y) image pixel coordinates",
        "point_order": ["AIP", "IIP"],
        **({"annotation_mode": annotation_mode} if annotation_mode else {}),
        **(
            {
                "reference_slice_index": reference_slice_index,
                "reference_slice_number": reference_slice_index + 1,
            }
            if reference_slice_index is not None
            else {}
        ),
        **(
            {"image_shape": [int(value) for value in image_shape]}
            if image_shape is not None
            else {}
        ),
        "slices": slices,
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary_path.replace(path)


def load_landmarks_json(
    path: str | Path,
    expected_slice_count: int,
    *,
    expected_image_shape: tuple[int, ...] | None = None,
) -> list[LandmarkPair | None]:
    """Load saved landmarks, or return an empty annotation list."""

    path = Path(path)
    if not path.exists():
        return [None] * expected_slice_count

    payload = json.loads(path.read_text(encoding="utf-8"))
    stored_shape = payload.get("image_shape")
    if (
        expected_image_shape is not None
        and stored_shape is not None
        and tuple(int(value) for value in stored_shape)
        != tuple(int(value) for value in expected_image_shape)
    ):
        raise ValueError(
            f"Landmark image shape {tuple(stored_shape)} does not match "
            f"the current image shape {tuple(expected_image_shape)}"
        )
    slices = payload.get("slices", [])
    if len(slices) != expected_slice_count:
        raise ValueError(
            f"Landmark file contains {len(slices)} slices, but the study has "
            f"{expected_slice_count}"
        )

    result: list[LandmarkPair | None] = []
    for item in slices:
        if item is None:
            result.append(None)
        elif item.get("status") in {"deleted", "skipped"}:
            result.append(None)
        else:
            result.append(
                LandmarkPair(
                    aip=tuple(map(float, item["aip"])),
                    iip=tuple(map(float, item["iip"])),
                )
            )
    if expected_image_shape is not None:
        if len(expected_image_shape) != 3:
            raise ValueError("expected_image_shape must be (slice, height, width)")
        height, width = expected_image_shape[1:]
        for index, pair in enumerate(result):
            if pair is None:
                continue
            for name, (x, y) in (("AIP", pair.aip), ("IIP", pair.iip)):
                if not (0.0 <= x < width and 0.0 <= y < height):
                    raise ValueError(
                        f"Saved {name} on slice {index + 1} is outside the "
                        f"current image shape {tuple(expected_image_shape)}"
                    )
    return result


def load_center_overrides_json(
    path: str | Path,
    expected_slice_count: int,
    *,
    expected_image_shape: tuple[int, ...] | None = None,
) -> list[tuple[float, float] | None]:
    """Load optional per-slice manual LV center overrides."""

    path = Path(path)
    if not path.exists():
        return [None] * expected_slice_count
    payload = json.loads(path.read_text(encoding="utf-8"))
    stored_shape = payload.get("image_shape")
    if (
        expected_image_shape is not None
        and stored_shape is not None
        and tuple(int(value) for value in stored_shape)
        != tuple(int(value) for value in expected_image_shape)
    ):
        raise ValueError(
            f"Landmark image shape {tuple(stored_shape)} does not match "
            f"the current image shape {tuple(expected_image_shape)}"
        )
    slices = payload.get("slices", [])
    if len(slices) != expected_slice_count:
        raise ValueError(
            f"Landmark file contains {len(slices)} slices, but the study has "
            f"{expected_slice_count}"
        )

    result: list[tuple[float, float] | None] = []
    height = width = None
    if expected_image_shape is not None:
        if len(expected_image_shape) != 3:
            raise ValueError("expected_image_shape must be (slice, height, width)")
        height, width = expected_image_shape[1:]
    for index, item in enumerate(slices):
        if not isinstance(item, dict) or item.get("status") == "deleted":
            result.append(None)
            continue
        value = item.get("center")
        if value is None:
            result.append(None)
            continue
        center = np.asarray(value, dtype=float)
        if center.shape != (2,) or not np.isfinite(center).all():
            raise ValueError(
                f"Saved center on slice {index + 1} must contain two finite values"
            )
        x, y = map(float, center)
        if width is not None and not (0.0 <= x < width and 0.0 <= y < height):
            raise ValueError(
                f"Saved center on slice {index + 1} is outside the current "
                f"image shape {tuple(expected_image_shape)}"
            )
        result.append((x, y))
    return result


def load_deleted_slice_indices(
    path: str | Path,
    expected_slice_count: int,
) -> set[int]:
    """Load slice indices explicitly excluded from radial bullseye mapping."""

    path = Path(path)
    if not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    slices = payload.get("slices", [])
    if len(slices) != expected_slice_count:
        raise ValueError(
            f"Landmark file contains {len(slices)} slices, but the study has "
            f"{expected_slice_count}"
        )
    return {
        index
        for index, item in enumerate(slices)
        if isinstance(item, dict) and item.get("status") == "deleted"
    }


def save_bullseye_npz(path: str | Path, result: BullseyeResult) -> None:
    """Save numeric bullseye outputs for later analysis."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        probabilities=result.probabilities,
        territory_labels=result.territory_labels,
        aha_segment_ids=result.aha_segment_ids,
        radial_positions=result.radial_positions,
        canonical_angles=result.angles,
        apex_territory=np.asarray(result.apex_territory, dtype=np.uint8),
    )


def save_aha17_csv(path: str | Path, result: BullseyeResult) -> None:
    """Export patient-specific territory percentages for all AHA segments."""

    table = compute_aha17_territory_percentages(result)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        stream.write(
            "segment_id,segment_name,LAD_percent,LCX_percent,RCA_percent,"
            "dominant_territory\n"
        )
        for segment_id, percentages in enumerate(table, start=1):
            if np.isfinite(percentages).any():
                dominant_id = int(np.nanargmax(percentages)) + 1
                values = [f"{value:.3f}" for value in percentages]
            else:
                dominant_id = 0
                values = ["", "", ""]
            stream.write(
                f"{segment_id},{AHA_SEGMENT_NAMES[segment_id]},"
                f"{','.join(values)},{TERRITORY_NAMES[dominant_id]}\n"
            )
