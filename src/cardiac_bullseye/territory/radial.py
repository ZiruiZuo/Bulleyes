"""Shared mask-anchored radial coordinates for bullseye products."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .mapping import last_retained_slice_index


@dataclass(frozen=True)
class MaskAnchoredRadialFrame:
    """Map one retained myocardium-mask extent onto bullseye radii 1..4."""

    source_radii: NDArray[np.float64]
    apical_index: int
    basal_index: int
    apical_position: float
    basal_position: float

    def map_positions(self, positions: ArrayLike) -> NDArray[np.float64]:
        """Map physical long-axis positions into this frame's plot radii."""

        values = np.asarray(positions, dtype=float)
        return 1.0 + 3.0 * (values - self.apical_position) / (
            self.basal_position - self.apical_position
        )


def build_mask_anchored_radial_frame(
    mask: ArrayLike,
    slice_positions: ArrayLike,
    deleted_slice_indices: Iterable[int] = (),
) -> MaskAnchoredRadialFrame:
    """Anchor radius 1 at first mask and radius 4 at retained basal mask."""

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
    source_radii = 1.0 + 3.0 * (positions - apical_position) / (
        basal_position - apical_position
    )
    return MaskAnchoredRadialFrame(
        source_radii=source_radii,
        apical_index=apical_index,
        basal_index=basal_index,
        apical_position=apical_position,
        basal_position=basal_position,
    )
