"""Shared mask-anchored radial coordinates for bullseye products."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .mapping import last_retained_slice_index


def projection_minimum_radius(projection: str) -> float:
    """Translate a public projection name into its apical plot radius."""

    if projection == "1-4":
        return 1.0
    if projection == "0-4":
        return 0.0
    raise ValueError("projection must be '1-4' or '0-4'")


@dataclass(frozen=True)
class MaskAnchoredRadialFrame:
    """Map one retained myocardium-mask extent onto a bullseye radius range."""

    source_radii: NDArray[np.float64]
    minimum_radius: float
    apical_index: int
    basal_index: int
    apical_position: float
    basal_position: float

    def map_positions(self, positions: ArrayLike) -> NDArray[np.float64]:
        """Map physical long-axis positions into this frame's plot radii."""

        values = np.asarray(positions, dtype=float)
        return self.minimum_radius + (4.0 - self.minimum_radius) * (
            values - self.apical_position
        ) / (
            self.basal_position - self.apical_position
        )


def build_mask_anchored_radial_frame(
    mask: ArrayLike,
    slice_positions: ArrayLike,
    deleted_slice_indices: Iterable[int] = (),
    *,
    minimum_radius: float = 1.0,
) -> MaskAnchoredRadialFrame:
    """Anchor the first mask at ``minimum_radius`` and retained base at 4."""

    mask_array = np.asarray(mask)
    positions = np.asarray(slice_positions, dtype=float)
    if mask_array.ndim < 2:
        raise ValueError("mask must have a slice axis and in-plane dimensions")
    if positions.shape != (mask_array.shape[0],):
        raise ValueError("slice_positions must have one value per mask slice")
    if not np.isfinite(positions).all() or (
        positions.size > 1 and not np.all(np.diff(positions) > 0)
    ):
        raise ValueError("slice_positions must be finite and increase apex-to-base")
    if not np.isfinite(minimum_radius) or not 0.0 <= minimum_radius < 4.0:
        raise ValueError("minimum_radius must be finite and in the range [0, 4)")

    in_plane_axes = tuple(range(1, mask_array.ndim))
    occupied = np.flatnonzero(np.any(mask_array != 0, axis=in_plane_axes))
    if occupied.size == 0:
        raise ValueError("mask is empty")
    apical_index = int(occupied[0])
    basal_index = last_retained_slice_index(
        mask_array.shape[0], deleted_slice_indices
    )
    if not np.any(mask_array[basal_index] != 0):
        raise ValueError("final retained basal slice has no mask")
    if basal_index <= apical_index:
        raise ValueError("mask-anchored mapping requires at least two retained slices")

    apical_position = float(positions[apical_index])
    basal_position = float(positions[basal_index])
    source_radii = minimum_radius + (4.0 - minimum_radius) * (
        positions - apical_position
    ) / (basal_position - apical_position)
    return MaskAnchoredRadialFrame(
        source_radii=source_radii,
        minimum_radius=float(minimum_radius),
        apical_index=apical_index,
        basal_index=basal_index,
        apical_position=apical_position,
        basal_position=basal_position,
    )
