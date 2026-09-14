# Standalone LGE Scar Mask Tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a portable three-argument tool that interactively selects per-slice normal-myocardium ROIs and writes a native-grid 5-SD LGE scar mask.

**Architecture:** The sibling folder `E:/ZZR/Code/Tools/lgeMask` contains independent calculation, image I/O, Matplotlib annotation, and CLI orchestration modules. It does not import the `Bulleyes` project's `cardiac_bullseye` package; tests import the folder directly and verify calculation, interaction state, cancellation, geometry preservation, and CLI behavior.

**Tech Stack:** Python 3.10+, NumPy, Matplotlib, PyQt5, SimpleITK, pytest (development only)

**Spec:** `docs/superpowers/specs/2026-09-14-standalone-lge-scar-mask-tool-design.md`

## Global Constraints

- The executable interface has exactly three required user-facing options: `--lge`, `--mask`, and `--output`.
- The threshold is fixed at per-slice `mean + 5 * sample standard deviation` (`ddof=1`) and uses strict `>` comparison.
- `Accept` requires at least 20 finite myocardium ROI voxels and keeps the user on the current slice when validation fails.
- `Skip` excludes the current slice and leaves that output slice entirely zero.
- Every run starts from blank ROIs; annotations are neither loaded nor persisted for reuse.
- Closing the window or selecting `Save & Quit` cancels without creating or replacing the requested output.
- The output is a binary `uint8` `.nii` or `.nii.gz` image on the complete original LGE grid with copied physical metadata.
- The standalone folder must not import `cardiac_bullseye`.
- Existing `cardiac-lge-bullseye` behavior is out of scope for this phase.

---

### Task 1: Pure Per-Slice Scar Calculation

**Files:**
- Create: `E:/ZZR/Code/Tools/lgeMask/scar.py`
- Create: `E:/ZZR/Code/Tools/lgeMask/tests/conftest.py`
- Create: `E:/ZZR/Code/Tools/lgeMask/tests/test_scar.py`

**Interfaces:**
- Consumes: NumPy image, myocardium, and ROI arrays.
- Produces: `ReferenceStatistics`, `compute_reference_statistics(image_slice, myocardium_mask, roi_mask)`, and `compute_scar_mask(lge, myocardium, roi_masks)`.

- [ ] **Step 1: Add test import setup and failing statistics tests**

```python
# conftest.py
from pathlib import Path
import sys

TOOL_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_DIRECTORY))
```

```python
# test_scar.py
import numpy as np
import pytest

from scar import compute_reference_statistics


def test_reference_statistics_use_sample_sd_and_fixed_five_sd_threshold():
    image = np.asarray([[8.0, 9.0, 10.0], [10.0, 11.0, 12.0]])
    myocardium = np.ones_like(image, dtype=bool)
    roi = np.ones_like(image, dtype=bool)

    result = compute_reference_statistics(image, myocardium, roi, min_voxels=6)

    assert result.count == 6
    assert result.mean == pytest.approx(10.0)
    assert result.std == pytest.approx(np.sqrt(2.0))
    assert result.threshold == pytest.approx(10.0 + 5.0 * np.sqrt(2.0))


def test_reference_statistics_reject_fewer_than_twenty_valid_voxels():
    image = np.ones((5, 5), dtype=float)
    roi = np.zeros((5, 5), dtype=bool)
    roi.ravel()[:19] = True

    with pytest.raises(ValueError, match="19.*20"):
        compute_reference_statistics(image, np.ones_like(roi), roi)
```

- [ ] **Step 2: Run the new tests and verify import failure**

Run: `python -m pytest E:/ZZR/Code/Tools/lgeMask/tests/test_scar.py -v`

Expected: FAIL because `scar.py` or its public functions do not exist.

- [ ] **Step 3: Implement immutable statistics and validation**

```python
@dataclass(frozen=True)
class ReferenceStatistics:
    count: int
    mean: float
    std: float
    threshold: float


def compute_reference_statistics(image_slice, myocardium_mask, roi_mask, *, min_voxels=20):
    image = np.asarray(image_slice)
    myocardium = np.asarray(myocardium_mask, dtype=bool)
    roi = np.asarray(roi_mask, dtype=bool)
    if image.shape != myocardium.shape or image.shape != roi.shape:
        raise ValueError("image, myocardium, and ROI must share a shape")
    selected = myocardium & roi & np.isfinite(image)
    count = int(np.count_nonzero(selected))
    if count < min_voxels:
        raise ValueError(f"ROI contains {count} valid myocardial voxels; at least {min_voxels} are required")
    values = np.asarray(image[selected], dtype=float)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    return ReferenceStatistics(count, mean, std, mean + 5.0 * std)
```

- [ ] **Step 4: Run statistics tests and verify they pass**

Run: `python -m pytest E:/ZZR/Code/Tools/lgeMask/tests/test_scar.py -v`

Expected: 2 passed.

- [ ] **Step 5: Add failing whole-volume tests for accepted, skipped, empty, finite, and strict-threshold behavior**

```python
from scar import compute_scar_mask


def test_scar_mask_thresholds_each_accepted_slice_and_zeros_skipped_slice():
    lge = np.zeros((2, 5, 5), dtype=float)
    myocardium = np.ones_like(lge, dtype=bool)
    first_roi = np.zeros((5, 5), dtype=bool)
    first_roi.ravel()[:20] = True
    lge[0, 4, 4] = 1.0

    result = compute_scar_mask(lge, myocardium, [first_roi, None])

    assert result.scar_mask[0, 4, 4]
    assert not result.scar_mask[1].any()
    assert result.statistics[0].threshold == 0.0
    assert result.statistics[1] is None


def test_scar_requires_finite_myocardium_voxel_strictly_above_threshold():
    lge = np.zeros((1, 5, 5), dtype=float)
    myocardium = np.ones_like(lge, dtype=bool)
    myocardium[0, 4, 3] = False
    roi = np.ones((5, 5), dtype=bool)
    roi[4, :] = False
    lge[0, 4, 2] = np.nan
    lge[0, 4, 3] = 2.0
    lge[0, 4, 4] = 0.0

    result = compute_scar_mask(lge, myocardium, [roi])

    assert not result.scar_mask[0, 4, 2]
    assert not result.scar_mask[0, 4, 3]
    assert not result.scar_mask[0, 4, 4]
```

- [ ] **Step 6: Run the new tests and verify missing-volume-function failure**

Run: `python -m pytest E:/ZZR/Code/Tools/lgeMask/tests/test_scar.py -v`

Expected: FAIL because `compute_scar_mask` does not exist.

- [ ] **Step 7: Implement volume result and fixed-threshold computation**

```python
@dataclass(frozen=True)
class ScarMaskResult:
    scar_mask: np.ndarray
    statistics: tuple[ReferenceStatistics | None, ...]


def compute_scar_mask(lge, myocardium, roi_masks):
    image = np.asarray(lge)
    mask = np.asarray(myocardium, dtype=bool)
    if image.shape != mask.shape or image.ndim != 3:
        raise ValueError("LGE and myocardium must be matching 3-D arrays")
    if len(roi_masks) != image.shape[0]:
        raise ValueError("There must be one accepted ROI or Skip entry per slice")
    scar = np.zeros(image.shape, dtype=bool)
    statistics = []
    for index, roi in enumerate(roi_masks):
        if roi is None or not mask[index].any():
            statistics.append(None)
            continue
        item = compute_reference_statistics(image[index], mask[index], roi)
        scar[index] = mask[index] & np.isfinite(image[index]) & (image[index] > item.threshold)
        statistics.append(item)
    return ScarMaskResult(scar, tuple(statistics))
```

- [ ] **Step 8: Run calculation tests and inspect the task files**

Run: `python -m pytest E:/ZZR/Code/Tools/lgeMask/tests/test_scar.py -v`

Expected: all tests pass.

Inspect the three task files after the tests pass. The target folder is outside
the existing `Bulleyes` Git repository, so this plan does not initialize or
modify version-control metadata for `lgeMask`.

---

### Task 2: Exact-Grid Image Loading and Native-Grid Saving

**Files:**
- Create: `E:/ZZR/Code/Tools/lgeMask/image_io.py`
- Create: `E:/ZZR/Code/Tools/lgeMask/tests/test_image_io.py`

**Interfaces:**
- Consumes: LGE path, myocardium-mask path, destination path, and a cropped boolean scar array.
- Produces: `LoadedStudy`, `load_study(lge_path, mask_path)`, `validate_output_path(path)`, and `save_scar_mask(path, study, cropped_scar_mask)`.

- [ ] **Step 1: Write failing tests for loading, cropping, and mismatch rejection**

```python
import numpy as np
import pytest
import SimpleITK as sitk

from image_io import load_study


def write_pair(tmp_path, *, different_spacing=False):
    lge = np.zeros((4, 5, 6), dtype=np.float32)
    mask = np.zeros_like(lge, dtype=np.uint8)
    mask[1:3, 1:4, 1:5] = 1
    lge_image = sitk.GetImageFromArray(lge)
    lge_image.SetSpacing((0.8, 0.9, 6.0))
    lge_image.SetOrigin((1.0, 2.0, 3.0))
    mask_image = sitk.GetImageFromArray(mask)
    mask_image.CopyInformation(lge_image)
    if different_spacing:
        mask_image.SetSpacing((0.8, 0.9, 7.0))
    lge_path = tmp_path / "lge.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    sitk.WriteImage(lge_image, str(lge_path))
    sitk.WriteImage(mask_image, str(mask_path))
    return lge_path, mask_path


def test_load_study_crops_to_first_and_last_nonempty_myocardium_slice(tmp_path):
    lge_path, mask_path = write_pair(tmp_path)
    study = load_study(lge_path, mask_path)
    assert study.lge.shape == (2, 5, 6)
    assert study.first_slice == 1
    assert study.full_shape == (4, 5, 6)


def test_load_study_rejects_geometry_mismatch_without_resampling(tmp_path):
    lge_path, mask_path = write_pair(tmp_path, different_spacing=True)
    with pytest.raises(ValueError, match="spacing differ"):
        load_study(lge_path, mask_path)
```

- [ ] **Step 2: Run I/O tests and verify import failure**

Run: `python -m pytest E:/ZZR/Code/Tools/lgeMask/tests/test_image_io.py -v`

Expected: FAIL because `image_io.py` does not exist.

- [ ] **Step 3: Implement geometry validation and cropped study loading**

Implement `LoadedStudy` with cropped `lge`, cropped boolean `myocardium`,
`first_slice`, `full_shape`, and the original SimpleITK `reference_image`.
Compare size exactly and spacing/origin/direction with absolute tolerances
`1e-5`, `1e-4`, and `1e-5`. Reject empty masks and unsupported dimensions.

- [ ] **Step 4: Run loading tests and verify they pass**

Run: `python -m pytest E:/ZZR/Code/Tools/lgeMask/tests/test_image_io.py -v`

Expected: 2 passed.

- [ ] **Step 5: Write failing tests for extensions, full-grid restoration, metadata, and overwrite**

```python
from image_io import save_scar_mask, validate_output_path


@pytest.mark.parametrize("name", ["scar.nii", "scar.nii.gz"])
def test_validate_output_path_accepts_nifti_extensions(name, tmp_path):
    assert validate_output_path(tmp_path / name).name == name


def test_validate_output_path_rejects_non_nifti_extension(tmp_path):
    with pytest.raises(ValueError, match=".nii"):
        validate_output_path(tmp_path / "scar.nrrd")


def test_save_restores_complete_grid_metadata_and_overwrites(tmp_path):
    lge_path, mask_path = write_pair(tmp_path)
    study = load_study(lge_path, mask_path)
    cropped = np.zeros(study.lge.shape, dtype=bool)
    cropped[0, 2, 2] = True
    output = tmp_path / "nested" / "scar.nii"
    output.parent.mkdir()
    output.write_bytes(b"old")

    save_scar_mask(output, study, cropped)

    saved_image = sitk.ReadImage(str(output))
    saved = sitk.GetArrayFromImage(saved_image)
    assert saved.dtype == np.uint8
    assert saved.shape == study.full_shape
    assert saved[study.first_slice, 2, 2] == 1
    assert saved_image.GetSpacing() == study.reference_image.GetSpacing()
    assert saved_image.GetOrigin() == study.reference_image.GetOrigin()
    assert saved_image.GetDirection() == study.reference_image.GetDirection()
```

- [ ] **Step 6: Run tests and verify missing save-function failures**

Run: `python -m pytest E:/ZZR/Code/Tools/lgeMask/tests/test_image_io.py -v`

Expected: FAIL because output validation and saving are not implemented.

- [ ] **Step 7: Implement NIfTI validation and full-grid binary saving**

Validate `.nii` and compound `.nii.gz` suffixes, create the parent directory,
insert the cropped scar into a zero-filled `uint8` full grid at `first_slice`,
intersect with cropped myocardium, copy the reference image information, and
write through SimpleITK to the requested path.

- [ ] **Step 8: Run I/O tests and inspect the task files**

Run: `python -m pytest E:/ZZR/Code/Tools/lgeMask/tests/test_image_io.py -v`

Expected: all tests pass.

Inspect `image_io.py` and its tests after the tests pass.

---

### Task 3: Fresh Brush ROI Annotation with Accept and Skip

**Files:**
- Create: `E:/ZZR/Code/Tools/lgeMask/roi_selector.py`
- Create: `E:/ZZR/Code/Tools/lgeMask/tests/test_roi_selector.py`

**Interfaces:**
- Consumes: cropped LGE and myocardium arrays.
- Produces: `AnnotationCancelled`, `apply_brush_stroke(mask, start_xy, end_xy, radius, value)`, `zoom_limits_about_point(...)`, and `select_reference_rois_interactively(lge, myocardium) -> list[np.ndarray | None]`.

- [ ] **Step 1: Write failing pure interaction tests**

```python
import numpy as np
import pytest

from roi_selector import apply_brush_stroke, zoom_limits_about_point


def test_brush_paints_continuous_path_and_erases():
    mask = np.zeros((15, 15), dtype=bool)
    apply_brush_stroke(mask, (2.0, 7.0), (12.0, 7.0), radius=1.0, value=True)
    assert mask[7, 2:13].all()
    apply_brush_stroke(mask, (7.0, 7.0), (7.0, 7.0), radius=1.0, value=False)
    assert not mask[7, 7]


def test_zoom_keeps_cursor_on_same_image_coordinate():
    xlim, ylim = zoom_limits_about_point(
        (-0.5, 99.5), (79.5, -0.5), (74.5, 19.5), scale=0.5,
        image_shape=(80, 100),
    )
    assert xlim == pytest.approx((37.0, 87.0))
    assert ylim == pytest.approx((49.5, 9.5))
```

- [ ] **Step 2: Run pure interaction tests and verify import failure**

Run: `python -m pytest E:/ZZR/Code/Tools/lgeMask/tests/test_roi_selector.py -v`

Expected: FAIL because `roi_selector.py` does not exist.

- [ ] **Step 3: Implement brush and cursor-centered zoom helpers**

Port the existing tested line-segment distance brush algorithm and bounded
cursor-centered zoom math into the standalone module without parent-package
imports.

- [ ] **Step 4: Run pure interaction tests and verify they pass**

Run: `python -m pytest E:/ZZR/Code/Tools/lgeMask/tests/test_roi_selector.py -v`

Expected: 2 passed.

- [ ] **Step 5: Add failing simulated GUI tests for invalid Accept, valid Accept, Skip, and fresh state**

Use fake Matplotlib `Button` objects and dispatch mouse events through the
figure canvas. The invalid-Accept test clicks `Accept` with 0 or 19 valid
voxels, verifies a visible warning and verifies that the event loop has not
advanced. The valid-Accept test paints at least 20 myocardium pixels and
verifies the returned mask. The Skip test verifies a `None` entry and the next
slice opens. A two-run test verifies the second invocation begins blank rather
than reusing the first returned masks.

- [ ] **Step 6: Run GUI tests and verify missing-selector failure**

Run: `python -m pytest E:/ZZR/Code/Tools/lgeMask/tests/test_roi_selector.py -v`

Expected: FAIL because the interactive selector is not implemented.

- [ ] **Step 7: Implement the Matplotlib annotation window**

Build one reusable figure with LGE display, myocardium outline, semi-transparent
ROI overlay, brush-radius slider, and `Accept`, `Skip`, `Undo`, `Clear`, `Fit`,
and `Save & Quit` controls. Maintain per-slice blank masks only in memory. Clip
every brush update to finite myocardium. `Accept` calls
`compute_reference_statistics`; on a count error it sets a red warning message
and stays on the slice. `Skip` stores `None`. Window close or `Save & Quit`
raises `AnnotationCancelled` after the event loop ends.

- [ ] **Step 8: Run ROI tests and inspect the task files**

Run: `python -m pytest E:/ZZR/Code/Tools/lgeMask/tests/test_roi_selector.py -v`

Expected: all tests pass.

Inspect `roi_selector.py` and its tests after the tests pass.

---

### Task 4: CLI Orchestration, Cancellation Safety, and Portable Documentation

**Files:**
- Create: `E:/ZZR/Code/Tools/lgeMask/main.py`
- Create: `E:/ZZR/Code/Tools/lgeMask/requirements.txt`
- Create: `E:/ZZR/Code/Tools/lgeMask/README.md`
- Create: `E:/ZZR/Code/Tools/lgeMask/tests/test_main.py`

**Interfaces:**
- Consumes: exactly `--lge`, `--mask`, and `--output` paths.
- Produces: `run(lge_path, mask_path, output_path)`, `main(argv=None) -> int`, console summaries, and the final NIfTI scar mask.

- [ ] **Step 1: Write failing CLI dispatch, summary, and cancellation tests**

```python
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import main as module


def test_main_requires_and_forwards_exactly_three_paths(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(module, "run", lambda lge, mask, output: calls.append((lge, mask, output)))
    status = module.main([
        "--lge", str(tmp_path / "lge.nii.gz"),
        "--mask", str(tmp_path / "mask.nii.gz"),
        "--output", str(tmp_path / "scar.nii"),
    ])
    assert status == 0
    assert calls == [(tmp_path / "lge.nii.gz", tmp_path / "mask.nii.gz", tmp_path / "scar.nii")]


def test_run_does_not_save_when_annotation_is_cancelled(monkeypatch, tmp_path):
    output = tmp_path / "scar.nii"
    output.write_bytes(b"existing")
    study = SimpleNamespace(
        lge=np.zeros((1, 5, 5)), myocardium=np.ones((1, 5, 5), dtype=bool),
    )
    monkeypatch.setattr(module, "load_study", lambda *_: study)
    monkeypatch.setattr(
        module, "select_reference_rois_interactively",
        lambda *_: (_ for _ in ()).throw(module.AnnotationCancelled()),
    )
    save_calls = []
    monkeypatch.setattr(module, "save_scar_mask", lambda *args: save_calls.append(args))

    assert module.main(["--lge", "a.nii", "--mask", "b.nii", "--output", str(output)]) == 1
    assert output.read_bytes() == b"existing"
    assert save_calls == []
```

- [ ] **Step 2: Run CLI tests and verify import failure**

Run: `python -m pytest E:/ZZR/Code/Tools/lgeMask/tests/test_main.py -v`

Expected: FAIL because `main.py` does not exist.

- [ ] **Step 3: Implement orchestration and guarded main entry point**

`run` validates the output path, loads the study, gets fresh ROI-or-Skip entries,
computes the scar mask, prints accepted/skipped per-slice summaries, and saves
only after the entire workflow completes. `main` catches expected file,
validation, runtime, and cancellation errors, prints `Error: ...` to stderr,
and returns 1; otherwise it returns 0. The module ends with:

```python
if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run CLI tests and verify they pass**

Run: `python -m pytest E:/ZZR/Code/Tools/lgeMask/tests/test_main.py -v`

Expected: all tests pass.

- [ ] **Step 5: Add exact standalone dependencies and usage documentation**

Create `requirements.txt` with:

```text
numpy>=1.26
matplotlib>=3.8
PyQt5>=5.15
SimpleITK>=2.3
```

Document Python 3.10+ virtual-environment creation, dependency installation,
the exact three-argument command, supported input formats and geometry,
annotation controls, fixed 20-voxel and 5-SD rules, Skip semantics,
cancellation semantics, and output interpretation in `README.md`.

- [ ] **Step 6: Add and run portability assertions**

Extend `test_main.py` to parse every standalone source file with `ast` and
assert that no import starts with `cardiac_bullseye`. Assert the requirements
file has exactly the four declared runtime dependencies.

Run: `python -m pytest E:/ZZR/Code/Tools/lgeMask/tests/test_main.py -v`

Expected: all tests pass.

- [ ] **Step 7: Run full verification and command help smoke test**

Run: `python -m pytest E:/ZZR/Code/Tools/lgeMask/tests -v`

Expected: all standalone tests pass.

Run from `E:/ZZR/Code/Tools/lgeMask`: `python main.py --help`

Expected: exit 0 and help text listing exactly `--lge`, `--mask`, and
`--output` as required tool-specific options.

Run: `python -m compileall -q E:/ZZR/Code/Tools/lgeMask`

Expected: exit 0 with no syntax errors.

- [ ] **Step 8: Review the completed standalone tool scope**

Confirm no existing `cardiac_bullseye` source file changed in this phase and
inspect all standalone files under `E:/ZZR/Code/Tools/lgeMask`.
