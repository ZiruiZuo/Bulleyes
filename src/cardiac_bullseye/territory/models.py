from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class StudyData:
    """One short-axis stack and its patient-specific territory labels.

    Arrays use slice-first order: ``(slice, y, x)``.
    ``slice_positions`` must increase from apex to base.
    """

    mr: NDArray[np.floating]
    territory_mask: NDArray[np.integer]
    slice_positions: NDArray[np.floating]

    def __post_init__(self) -> None:
        if self.mr.ndim != 3:
            raise ValueError("mr must have shape (slices, height, width)")
        if self.territory_mask.shape != self.mr.shape:
            raise ValueError("territory_mask must have the same shape as mr")
        if self.slice_positions.shape != (self.mr.shape[0],):
            raise ValueError("slice_positions must have one value per slice")
        if not np.all(np.diff(self.slice_positions) > 0):
            raise ValueError("slice_positions must increase from apex to base")

        labels = set(np.unique(self.territory_mask).tolist())
        if not labels.issubset({0, 1, 2, 3}):
            raise ValueError(
                "territory_mask may only contain 0=background, 1=LAD, "
                "2=LCX, and 3=RCA"
            )

    @property
    def slice_count(self) -> int:
        return self.mr.shape[0]


@dataclass(frozen=True)
class LandmarkPair:
    """AIP and IIP in zero-based Matplotlib pixel coordinates (x, y)."""

    aip: tuple[float, float]
    iip: tuple[float, float]

    def as_array(self) -> NDArray[np.float64]:
        return np.asarray([self.aip, self.iip], dtype=float)


@dataclass(frozen=True)
class BullseyeResult:
    """Computed annular bullseye data and its AHA-17 assignments.

    ``probabilities`` has shape ``(radial, angle, 3)`` in LAD, LCX, RCA
    order. ``angles`` are canonical angles before the 180-degree display
    rotation. The apical cap (segment 17) is stored separately.
    """

    probabilities: NDArray[np.floating]
    territory_labels: NDArray[np.integer]
    aha_segment_ids: NDArray[np.integer]
    radial_positions: NDArray[np.floating]
    angles: NDArray[np.floating]
    apex_territory: int

    def __post_init__(self) -> None:
        if self.probabilities.ndim != 3 or self.probabilities.shape[2] != 3:
            raise ValueError("probabilities must have shape (radial, angle, 3)")
        shape = self.probabilities.shape[:2]
        if self.territory_labels.shape != shape:
            raise ValueError("territory_labels shape does not match probabilities")
        if self.aha_segment_ids.shape != shape:
            raise ValueError("aha_segment_ids shape does not match probabilities")
        if self.radial_positions.shape != (shape[0],):
            raise ValueError("radial_positions length does not match probabilities")
        if self.angles.shape != (shape[1],):
            raise ValueError("angles length does not match probabilities")
        if self.apex_territory not in {0, 1, 2, 3}:
            raise ValueError("apex_territory must be 0, 1, 2, or 3")
