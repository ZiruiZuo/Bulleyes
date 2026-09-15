# External LGE Scar Mask Bullseye Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `cardiac-lge-mask-bullseye`, which accepts co-gridded LGE, myocardium mask, and external scar mask inputs, asks only for fresh AIP/IIP landmarks, and writes exactly the landmarks plus intensity and scar bullseye PNGs.

**Architecture:** Add a separate `[lge_mask_bullseye]` configuration contract and workflow. Load and strictly validate all three images before opening the GUI, then use a shared non-interactive LGE product builder so both the new command and the existing complete LGE command use identical AHA 1-16 mapping and plotting code.

**Tech Stack:** Python 3.10+, dataclasses, NumPy, SimpleITK, Matplotlib/PyQt5, TOML (`tomllib`/`tomli`), pytest.

**Spec:** `docs/superpowers/specs/2026-09-15-lge-mask-bullseye-workflow-design.md`

## Global Constraints

- The new console command is exactly `cardiac-lge-mask-bullseye --config <file>`.
- The new TOML section is exactly `[lge_mask_bullseye]`; `[lge_bullseye]` keeps its current meaning.
- Required image fields are `lge`, `mask`, and `lge_scar_mask`.
- All three images must share size, spacing, origin, and direction; no automatic resampling is permitted.
- Any nonzero scar voxel outside the myocardium is a fatal validation error, and the message must include the violating voxel count.
- AIP/IIP annotation always starts with `resume_existing=False`.
- Segment 17 is not calculated or displayed.
- The new workflow writes exactly `<case>_lge_landmarks.json`, `<case>_lge_intensity_bullseye.png`, and `<case>_lge_scar_bullseye.png`.
- The existing `cardiac-lge-bullseye` configuration, interaction sequence, output names, seven-file return contract, and scar thresholding remain compatible.
- Reuse the existing atomic PNG replacement implementation.
- Preserve all pre-existing working-tree changes. Before each commit, inspect `git diff --cached --name-only` and do not stage unrelated files or hunks.

---

## File Map

- Modify `src/cardiac_bullseye/config.py`: define and validate the new configuration contract.
- Modify `src/cardiac_bullseye/lge/io.py`: load three co-gridded images and enforce scar containment.
- Modify `src/cardiac_bullseye/lge/__init__.py`: export the new loader result and function.
- Create `src/cardiac_bullseye/workflows/lge_common.py`: build shared intensity/scar products and save the two PNGs.
- Modify `src/cardiac_bullseye/workflows/lge.py`: route the complete workflow through the shared product builder without changing outputs.
- Create `src/cardiac_bullseye/workflows/lge_mask_bullseye.py`: orchestrate the external-scar workflow.
- Modify `src/cardiac_bullseye/workflows/__init__.py`: export the new workflow API.
- Modify `src/cardiac_bullseye/territory/landmarks.py`: optionally turn Save & Quit/window close into a workflow cancellation error.
- Modify `src/cardiac_bullseye/cli.py`: expose the new runner.
- Modify `pyproject.toml`: register the new console script.
- Modify `config.example.toml` and `README.md`: document the independent configuration and three-part workflow.
- Create `toml/lge_mask_bullseye.example.toml`: provide a copyable focused example.
- Modify/create tests under `tests/config`, `tests/lge`, `tests/territory`, `tests/workflows`, and `tests/cli` beside the code they protect.

---

### Task 1: Add the Independent Configuration Contract

**Files:**
- Modify: `src/cardiac_bullseye/config.py`
- Modify: `tests/config/test_config.py`

**Interfaces:**
- Consumes: existing `CaseConfig`, `_path`, `_input_error`, numeric validators, and `load_config` conventions.
- Produces: `LgeMaskBullseyeConfig`; `AppConfig.lge_mask_bullseye`; workflow literal `"lge-mask-bullseye"`.

- [ ] **Step 1: Write failing parsing and validation tests**

Add tests that create all three input files and verify the independent section and defaults:

```python
def test_lge_mask_bullseye_config_parses_three_inputs_and_defaults(tmp_path):
    lge = tmp_path / "lge.nii.gz"
    myocardium = tmp_path / "myocardium.seg.nrrd"
    scar = tmp_path / "scar.nii.gz"
    for path in (lge, myocardium, scar):
        path.touch()
    config_path = write_config(
        tmp_path,
        f"""
[case]
id = "032602"
output_directory = "results"
[lge_mask_bullseye]
lge = '{lge.as_posix()}'
mask = '{myocardium.as_posix()}'
lge_scar_mask = '{scar.as_posix()}'
""",
    )

    config = load_config(config_path, "lge-mask-bullseye")

    assert config.lge_mask_bullseye.lge == lge
    assert config.lge_mask_bullseye.mask == myocardium
    assert config.lge_mask_bullseye.lge_scar_mask == scar
    assert config.lge_mask_bullseye.angle_count == 360
    assert config.lge_mask_bullseye.radial_count == 300
    assert config.lge_mask_bullseye.six_segment_warp is True
    assert config.case.output_directory == tmp_path / "results"


def test_lge_mask_bullseye_reports_all_missing_inputs(tmp_path):
    config_path = write_config(
        tmp_path,
        """
[case]
id = "case"
[lge_mask_bullseye]
lge = "missing_lge.nii.gz"
mask = "missing_mask.seg.nrrd"
lge_scar_mask = "missing_scar.nii.gz"
""",
    )

    with pytest.raises(ConfigError) as caught:
        load_config(config_path, "lge-mask-bullseye")

    message = str(caught.value)
    assert "lge_mask_bullseye.lge" in message
    assert "lge_mask_bullseye.mask" in message
    assert "lge_mask_bullseye.lge_scar_mask" in message
```

- [ ] **Step 2: Run the two tests and verify the red state**

Run:

```powershell
python -m pytest tests/config/test_config.py -k "lge_mask_bullseye" -v
```

Expected: both tests fail because `"lge-mask-bullseye"` and `AppConfig.lge_mask_bullseye` do not exist.

- [ ] **Step 3: Add the dataclass, parser, and workflow validation**

In `config.py`, extend the workflow literal and add:

```python
WorkflowName = Literal[
    "short-axis",
    "bullseye",
    "pipeline",
    "lge-bullseye",
    "lge-mask-bullseye",
]


@dataclass(frozen=True)
class LgeMaskBullseyeConfig:
    lge: Path | None
    mask: Path | None
    lge_scar_mask: Path | None
    angle_count: int = 360
    radial_count: int = 300
    angular_smoothing_sigma_deg: float = 1.5
    radial_smoothing_sigma: float = 1.5
    six_segment_warp: bool = True
```

Add this defaulted member to `AppConfig`:

```python
lge_mask_bullseye: LgeMaskBullseyeConfig = field(
    default_factory=lambda: LgeMaskBullseyeConfig(
        lge=None,
        mask=None,
        lge_scar_mask=None,
    )
)
```

Parse `data.get("lge_mask_bullseye", {})`, resolve all three paths through `_path`, and copy the five mapping options with the documented defaults.

```python
lge_mask_bullseye_data = data.get("lge_mask_bullseye", {})

# Inside AppConfig(...)
lge_mask_bullseye=LgeMaskBullseyeConfig(
    lge=_path(lge_mask_bullseye_data.get("lge"), base_directory),
    mask=_path(lge_mask_bullseye_data.get("mask"), base_directory),
    lge_scar_mask=_path(
        lge_mask_bullseye_data.get("lge_scar_mask"),
        base_directory,
    ),
    angle_count=lge_mask_bullseye_data.get("angle_count", 360),
    radial_count=lge_mask_bullseye_data.get("radial_count", 300),
    angular_smoothing_sigma_deg=lge_mask_bullseye_data.get(
        "angular_smoothing_sigma_deg",
        1.5,
    ),
    radial_smoothing_sigma=lge_mask_bullseye_data.get(
        "radial_smoothing_sigma",
        1.5,
    ),
    six_segment_warp=lge_mask_bullseye_data.get("six_segment_warp", True),
),
```

Add a private validator to avoid diverging defaults between the two LGE configurations:

```python
def _validate_lge_mapping_options(errors, prefix, options) -> None:
    _positive_integer(errors, f"{prefix}.angle_count", options.angle_count)
    _positive_integer(errors, f"{prefix}.radial_count", options.radial_count)
    _nonnegative_number(
        errors,
        f"{prefix}.angular_smoothing_sigma_deg",
        options.angular_smoothing_sigma_deg,
    )
    _nonnegative_number(
        errors,
        f"{prefix}.radial_smoothing_sigma",
        options.radial_smoothing_sigma,
    )
```

Call it for both LGE sections. For the new workflow, call `_input_error` for exactly the three required paths. Keep the existing scar-threshold parameter checks restricted to `LgeBullseyeConfig`.

- [ ] **Step 4: Run focused configuration tests**

Run:

```powershell
python -m pytest tests/config/test_config.py tests/config/test_example_config.py -v
```

Expected: all configuration tests pass; existing `lge-bullseye` defaults and scar-parameter validation remain unchanged.

- [ ] **Step 5: Commit the configuration contract**

```powershell
git add src/cardiac_bullseye/config.py tests/config/test_config.py
git diff --cached --name-only
git commit -m "feat: add external scar bullseye config"
```

Expected staged paths: only the two paths listed above. If either file already contains unrelated unstaged changes, stage only the task's hunks and recheck the cached diff before committing.

---

### Task 2: Load and Strictly Validate the External Scar Mask

**Files:**
- Modify: `src/cardiac_bullseye/lge/io.py`
- Modify: `src/cardiac_bullseye/lge/__init__.py`
- Modify: `tests/lge/test_lge.py`

**Interfaces:**
- Consumes: `StudyData` and the physical-grid rules already used by `load_lge_study`.
- Produces: `LgeStudyWithScar(study: StudyData, scar_mask: NDArray[np.bool_])`; `load_lge_study_with_scar(lge_path, myocardium_mask_path, scar_mask_path) -> LgeStudyWithScar`.

- [ ] **Step 1: Add failing happy-path, crop, empty-scar, and containment tests**

Extend the imports in `tests/lge/test_lge.py`, then add:

```python
def test_three_image_loader_crops_scar_with_myocardium(tmp_path):
    lge = np.arange(5 * 9 * 9, dtype=np.float32).reshape(5, 9, 9)
    myocardium = np.zeros_like(lge, dtype=np.uint8)
    myocardium[1:4, 2:7, 2:7] = 1
    scar = np.zeros_like(myocardium)
    scar[2, 4, 4] = 1
    lge_path = tmp_path / "lge.nii.gz"
    myocardium_path = tmp_path / "myocardium.seg.nrrd"
    scar_path = tmp_path / "scar.nii.gz"
    _write_image(lge_path, lge)
    _write_image(myocardium_path, myocardium)
    _write_image(scar_path, scar)

    loaded = load_lge_study_with_scar(lge_path, myocardium_path, scar_path)

    assert loaded.study.mr.shape == (3, 9, 9)
    np.testing.assert_array_equal(loaded.study.mr, lge[1:4])
    np.testing.assert_array_equal(loaded.scar_mask, scar[1:4].astype(bool))


def test_three_image_loader_accepts_empty_scar(tmp_path):
    lge = np.zeros((3, 9, 9), dtype=np.float32)
    myocardium = np.zeros_like(lge, dtype=np.uint8)
    myocardium[:, 2:7, 2:7] = 1
    scar = np.zeros_like(myocardium)
    paths = [tmp_path / name for name in ("lge.nii.gz", "myo.nrrd", "scar.nii.gz")]
    for path, array in zip(paths, (lge, myocardium, scar), strict=True):
        _write_image(path, array)

    loaded = load_lge_study_with_scar(*paths)

    assert not loaded.scar_mask.any()


def test_three_image_loader_rejects_scar_outside_myocardium_with_count(tmp_path):
    lge = np.zeros((3, 9, 9), dtype=np.float32)
    myocardium = np.zeros_like(lge, dtype=np.uint8)
    myocardium[:, 2:7, 2:7] = 1
    scar = np.zeros_like(myocardium)
    scar[0, 0, 0] = 1
    scar[1, 8, 8] = 1
    paths = [tmp_path / name for name in ("lge.nii.gz", "myo.nrrd", "scar.nii.gz")]
    for path, array in zip(paths, (lge, myocardium, scar), strict=True):
        _write_image(path, array)

    with pytest.raises(ValueError, match=r"2.*outside.*myocardium"):
        load_lge_study_with_scar(*paths)
```

- [ ] **Step 2: Add failing geometry and scalar-scar tests**

Add a parameterized geometry test that changes one SimpleITK property at a time for the scar image and expects the property name in the error. Add a vector scar image with `sitk.GetImageFromArray(array, isVector=True)` and assert that the error contains `scar mask must be a scalar`.

Representative assertion:

```python
with pytest.raises(ValueError, match="scar mask origin"):
    load_lge_study_with_scar(lge_path, myocardium_path, scar_path)
```

- [ ] **Step 3: Run the loader tests and verify the red state**

Run:

```powershell
python -m pytest tests/lge/test_lge.py -k "three_image_loader" -v
```

Expected: collection or import fails because `load_lge_study_with_scar` is absent.

- [ ] **Step 4: Implement named grid comparison and the three-image loader**

Refactor the private grid helper without changing existing default error wording:

```python
def _require_same_geometry(
    reference_image,
    candidate_image,
    *,
    reference_name: str = "LGE",
    candidate_name: str = "myocardium mask",
) -> None:
    checks = (
        ("size", reference_image.GetSize(), candidate_image.GetSize(), 0.0),
        ("spacing", reference_image.GetSpacing(), candidate_image.GetSpacing(), 1.0e-5),
        ("origin", reference_image.GetOrigin(), candidate_image.GetOrigin(), 1.0e-4),
        (
            "direction",
            reference_image.GetDirection(),
            candidate_image.GetDirection(),
            1.0e-5,
        ),
    )
    for name, reference_value, candidate_value, tolerance in checks:
        matches = (
            tuple(reference_value) == tuple(candidate_value)
            if tolerance == 0.0
            else np.allclose(
                np.asarray(reference_value, dtype=float),
                np.asarray(candidate_value, dtype=float),
                rtol=0.0,
                atol=tolerance,
            )
        )
        if not matches:
            raise ValueError(
                f"{reference_name} and {candidate_name} {name} differ: "
                f"{reference_name}={tuple(reference_value)}, "
                f"{candidate_name}={tuple(candidate_value)}. "
                "All inputs must already share the LGE short-axis grid; "
                "no automatic resampling is performed."
            )
```

Add the public result type:

```python
@dataclass(frozen=True)
class LgeStudyWithScar:
    study: StudyData
    scar_mask: NDArray[np.bool_]
```

Implement:

```python
def load_lge_study_with_scar(
    lge_path: str | Path,
    myocardium_mask_path: str | Path,
    scar_mask_path: str | Path,
) -> LgeStudyWithScar:
    import SimpleITK as sitk

    paths = (
        ("LGE", Path(lge_path)),
        ("myocardium mask", Path(myocardium_mask_path)),
        ("scar mask", Path(scar_mask_path)),
    )
    for description, path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"{description} file does not exist: {path}")

    lge_image = sitk.ReadImage(str(paths[0][1]))
    myocardium_image = sitk.ReadImage(str(paths[1][1]))
    scar_image = sitk.ReadImage(str(paths[2][1]))
    for description, image in (
        ("LGE", lge_image),
        ("myocardium mask", myocardium_image),
        ("scar mask", scar_image),
    ):
        if image.GetDimension() != 3:
            raise ValueError(f"{description} must be a three-dimensional image")
    if lge_image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError("LGE must be a scalar intensity image")
    if scar_image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError("scar mask must be a scalar image")

    _require_same_geometry(lge_image, myocardium_image)
    _require_same_geometry(
        lge_image,
        scar_image,
        candidate_name="scar mask",
    )

    lge = np.asarray(sitk.GetArrayFromImage(lge_image), dtype=np.float32)
    raw_myocardium = np.asarray(sitk.GetArrayFromImage(myocardium_image))
    myocardium = (
        np.any(raw_myocardium != 0, axis=-1)
        if raw_myocardium.ndim == 4
        else raw_myocardium != 0
    )
    if myocardium.ndim != 3:
        raise ValueError(
            f"Unsupported myocardium mask array shape: {raw_myocardium.shape}"
        )
    scar = np.asarray(sitk.GetArrayFromImage(scar_image)) != 0
    if lge.shape != myocardium.shape or lge.shape != scar.shape:
        raise ValueError("LGE, myocardium mask, and scar mask arrays must share a shape")

    occupied = np.flatnonzero(np.any(myocardium, axis=(1, 2)))
    if occupied.size == 0:
        raise ValueError("The myocardium mask is empty")
    outside_count = int(np.count_nonzero(scar & ~myocardium))
    if outside_count:
        raise ValueError(
            f"Scar mask contains {outside_count} foreground voxels outside "
            "the myocardium mask"
        )

    first_slice = int(occupied[0])
    last_slice = int(occupied[-1]) + 1
    spacing = float(lge_image.GetSpacing()[2])
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("LGE slice spacing must be a positive finite value")
    study = StudyData(
        np.ascontiguousarray(lge[first_slice:last_slice]),
        np.ascontiguousarray(myocardium[first_slice:last_slice], dtype=np.uint8),
        np.arange(last_slice - first_slice, dtype=float) * spacing,
    )
    return LgeStudyWithScar(
        study=study,
        scar_mask=np.ascontiguousarray(scar[first_slice:last_slice], dtype=bool),
    )
```

Keep `load_lge_study` behavior intact, sharing private array/crop helpers where that removes duplication. Export `LgeStudyWithScar` and `load_lge_study_with_scar` from `lge/__init__.py`.

- [ ] **Step 5: Run all LGE I/O tests**

Run:

```powershell
python -m pytest tests/lge/test_lge.py -v
```

Expected: all existing two-image loader tests and all new three-image loader tests pass.

- [ ] **Step 6: Commit the validated loader**

```powershell
git add src/cardiac_bullseye/lge/io.py src/cardiac_bullseye/lge/__init__.py tests/lge/test_lge.py
git diff --cached --name-only
git commit -m "feat: validate external LGE scar masks"
```

---

### Task 3: Extract Shared LGE Bullseye Product Computation

**Files:**
- Create: `src/cardiac_bullseye/workflows/lge_common.py`
- Create: `tests/workflows/test_lge_common.py`
- Modify: `src/cardiac_bullseye/workflows/lge.py`
- Modify: `tests/workflows/test_lge_workflow.py`

**Interfaces:**
- Consumes: `StudyData`, `LandmarkPair`, `compute_lge_bullseye`, `compute_aha16_segment_ids`, `compute_aha16_scar_percentages`, and the two existing plotting functions.
- Produces: `LgeBullseyeProducts`; `compute_lge_bullseye_products(...)`; `save_lge_bullseye_figures(...)`.

- [ ] **Step 1: Write a failing shared-product test**

Create `tests/workflows/test_lge_common.py` with a three-slice ring study, known landmarks, and a scar mask. Verify shapes, scar counts, and both saved files:

```python
def test_shared_products_compute_intensity_and_aha16_scar(tmp_path):
    study, landmarks, scar = make_three_slice_ring_study()

    products = compute_lge_bullseye_products(
        study,
        landmarks,
        scar,
        angle_count=24,
        radial_count=9,
        angular_smoothing_sigma_deg=0.0,
        radial_smoothing_sigma=0.0,
        six_segment_warp=False,
        deleted_slice_indices=(),
    )

    assert products.intensity.intensity.shape == (9, 24)
    assert products.scar.scar_percentages.shape == (16,)
    assert int(products.scar.scar_voxel_counts.sum()) == int(scar.sum())

    intensity_path, scar_path = save_lge_bullseye_figures(
        products,
        tmp_path / "intensity.png",
        tmp_path / "scar.png",
    )
    assert intensity_path.is_file()
    assert scar_path.is_file()
```

`make_three_slice_ring_study()` must construct myocardium at all three AHA levels and place each scar voxel inside the ring so every scar voxel receives a segment ID.

- [ ] **Step 2: Run the new test and verify the red state**

Run:

```powershell
python -m pytest tests/workflows/test_lge_common.py -v
```

Expected: import fails because `workflows.lge_common` does not exist.

- [ ] **Step 3: Implement the shared product builder and saver**

Create `lge_common.py` with these exact interfaces:

```python
@dataclass(frozen=True)
class LgeBullseyeProducts:
    intensity: LgeBullseyeResult
    scar: Aha16ScarSummary


def compute_lge_bullseye_products(
    study: StudyData,
    landmarks: list[LandmarkPair | None],
    scar_mask: ArrayLike,
    *,
    angle_count: int,
    radial_count: int,
    angular_smoothing_sigma_deg: float,
    radial_smoothing_sigma: float,
    six_segment_warp: bool,
    deleted_slice_indices: Iterable[int] = (),
) -> LgeBullseyeProducts:
    scar = np.asarray(scar_mask, dtype=bool)
    if scar.shape != study.mr.shape:
        raise ValueError("scar_mask must have the same shape as the LGE study")
    if np.any(scar & ~(study.territory_mask > 0)):
        raise ValueError("scar_mask must be contained within myocardium")
    segment_ids = compute_aha16_segment_ids(
        study,
        landmarks,
        six_segment_warp=six_segment_warp,
        deleted_slice_indices=deleted_slice_indices,
    )
    summary = compute_aha16_scar_percentages(scar, segment_ids)
    intensity = compute_lge_bullseye(
        study,
        landmarks,
        angle_count=angle_count,
        radial_count=radial_count,
        angular_smoothing_sigma_deg=angular_smoothing_sigma_deg,
        radial_smoothing_sigma=radial_smoothing_sigma,
        six_segment_warp=six_segment_warp,
        deleted_slice_indices=deleted_slice_indices,
    )
    return LgeBullseyeProducts(intensity=intensity, scar=summary)


def save_lge_bullseye_figures(
    products: LgeBullseyeProducts,
    intensity_path: str | Path,
    scar_path: str | Path,
) -> tuple[Path, Path]:
    intensity_path = Path(intensity_path)
    scar_path = Path(scar_path)
    plot_lge_bullseye(products.intensity, intensity_path, show=False)
    plot_lge_scar_bullseye(products.scar, scar_path, show=False)
    return intensity_path, scar_path
```

- [ ] **Step 4: Run the shared-product test**

Run:

```powershell
python -m pytest tests/workflows/test_lge_common.py -v
```

Expected: pass, and the two PNGs are nonempty.

- [ ] **Step 5: Refactor the complete workflow through the shared builder**

In `workflows/lge.py`, replace its direct AHA/intensity computation with one call to `compute_lge_bullseye_products`. Save the complete workflow's existing NPZ and CSV from `products.intensity` and `products.scar`, and call `save_lge_bullseye_figures` for its PNGs. Keep `LgeBullseyeOutputs` fields, seven filenames, fresh ROI/landmark behavior, and saved scar mask unchanged.

Representative replacement:

```python
products = compute_lge_bullseye_products(
    study,
    landmarks,
    scar_result.scar_mask,
    angle_count=config.lge_bullseye.angle_count,
    radial_count=config.lge_bullseye.radial_count,
    angular_smoothing_sigma_deg=config.lge_bullseye.angular_smoothing_sigma_deg,
    radial_smoothing_sigma=config.lge_bullseye.radial_smoothing_sigma,
    six_segment_warp=config.lge_bullseye.six_segment_warp,
    deleted_slice_indices=deleted,
)
saved_numeric = save_lge_bullseye_npz(numeric_path, products.intensity)
saved_scar_segments = save_aha16_scar_csv(scar_segments_path, products.scar)
save_lge_bullseye_figures(products, figure_path, scar_figure_path)
```

- [ ] **Step 6: Run full-workflow regression tests**

Run:

```powershell
python -m pytest tests/workflows/test_lge_common.py tests/workflows/test_lge_workflow.py tests/lge/test_scar.py -v
```

Expected: all tests pass; the complete workflow still creates all seven outputs and still contains three scar voxels in its synthetic saved mask.

- [ ] **Step 7: Commit the shared core and refactor**

```powershell
git add src/cardiac_bullseye/workflows/lge_common.py src/cardiac_bullseye/workflows/lge.py tests/workflows/test_lge_common.py tests/workflows/test_lge_workflow.py
git diff --cached --name-only
git commit -m "refactor: share LGE bullseye product generation"
```

---

### Task 4: Make Landmark Cancellation Explicit for the New Workflow

**Files:**
- Modify: `src/cardiac_bullseye/territory/landmarks.py`
- Modify: `tests/territory/test_landmarks.py`

**Interfaces:**
- Consumes: existing `select_landmarks_interactively` behavior and GUI callbacks.
- Produces: optional keyword `cancel_is_error: bool = False`; `RuntimeError("AIP/IIP annotation was cancelled")` when the new workflow opts in.

- [ ] **Step 1: Write a failing opt-in cancellation test**

Use the existing fake-button/event-loop pattern in `tests/territory/test_landmarks.py`. Capture the `Save && Quit` callback, invoke it from the fake event loop, and assert:

```python
with pytest.raises(RuntimeError, match="AIP/IIP annotation was cancelled"):
    select_landmarks_interactively(
        study,
        tmp_path / "landmarks.json",
        resume_existing=False,
        cancel_is_error=True,
    )
```

Also retain or add a test showing that omitting `cancel_is_error` preserves the existing complete-workflow behavior.

- [ ] **Step 2: Run the focused test and verify the red state**

Run:

```powershell
python -m pytest tests/territory/test_landmarks.py -k "cancel" -v
```

Expected: failure with an unexpected keyword argument for `cancel_is_error`.

- [ ] **Step 3: Implement the opt-in behavior**

Extend the signature:

```python
def select_landmarks_interactively(
    study: StudyData,
    output_path: str | Path,
    *,
    revisit_completed: bool = False,
    resume_existing: bool = True,
    transpose_for_display: bool = False,
    cancel_is_error: bool = False,
) -> list[LandmarkPair | None]:
```

In the existing quit/close branch, preserve progress saving and cleanup, then add:

```python
if cancel_is_error:
    raise RuntimeError("AIP/IIP annotation was cancelled")
return landmarks
```

Do not change mouse controls, zoom, pan, interpolation, Delete semantics, or the default behavior used by existing callers.

- [ ] **Step 4: Run all landmark tests**

Run:

```powershell
python -m pytest tests/territory/test_landmarks.py -v
```

Expected: all tests pass, including default compatibility and opt-in cancellation.

- [ ] **Step 5: Commit the cancellation contract**

```powershell
git add src/cardiac_bullseye/territory/landmarks.py tests/territory/test_landmarks.py
git diff --cached --name-only
git commit -m "feat: expose landmark annotation cancellation"
```

---

### Task 5: Add the External-Scar Workflow and Console Command

**Files:**
- Create: `src/cardiac_bullseye/workflows/lge_mask_bullseye.py`
- Modify: `src/cardiac_bullseye/workflows/__init__.py`
- Modify: `src/cardiac_bullseye/cli.py`
- Modify: `pyproject.toml`
- Create: `tests/workflows/test_lge_mask_bullseye_workflow.py`
- Modify: `tests/cli/test_cli.py`

**Interfaces:**
- Consumes: `load_lge_study_with_scar`, `select_landmarks_interactively`, `load_deleted_slice_indices`, `compute_lge_bullseye_products`, and `save_lge_bullseye_figures`.
- Produces: `LgeMaskBullseyeOutputs`; `generate_lge_mask_bullseye(config: AppConfig) -> LgeMaskBullseyeOutputs`; `lge_mask_bullseye_main(argv=None) -> int`.

- [ ] **Step 1: Add failing workflow and exact-output tests**

Create a synthetic three-slice LGE, `.seg.nrrd` myocardium, and external scar NIfTI. Monkeypatch landmark selection to assert the fresh/cancellation arguments and write its fake landmarks JSON with `save_landmarks_json`:

```python
def select_fresh_landmarks(
    study,
    output_path,
    *,
    resume_existing,
    cancel_is_error,
):
    assert resume_existing is False
    assert cancel_is_error is True
    landmarks = [landmark] * study.slice_count
    save_landmarks_json(output_path, landmarks)
    return landmarks
```

Protect against accidental delegation to the complete workflow:

```python
def forbidden(*_args, **_kwargs):
    raise AssertionError("external-scar workflow must not select ROIs or compute scar")

monkeypatch.setattr(
    "cardiac_bullseye.workflows.lge.select_reference_rois_interactively",
    forbidden,
)
monkeypatch.setattr(
    "cardiac_bullseye.workflows.lge.compute_lge_scar",
    forbidden,
)
```

Run the new workflow and assert:

```python
outputs = generate_lge_mask_bullseye(config)

assert outputs.landmarks.name == "032602_lge_landmarks.json"
assert outputs.intensity_figure.name == "032602_lge_intensity_bullseye.png"
assert outputs.scar_figure.name == "032602_lge_scar_bullseye.png"
assert {path.name for path in output_directory.iterdir()} == {
    "032602_lge_landmarks.json",
    "032602_lge_intensity_bullseye.png",
    "032602_lge_scar_bullseye.png",
}
```

Add another workflow test with an empty scar mask and assert successful creation of the same three outputs.

- [ ] **Step 2: Add failing CLI dispatch/help tests**

Extend the CLI parameter tables with:

```python
(
    "lge_mask_bullseye_main",
    "generate_lge_mask_bullseye",
    "lge-mask-bullseye",
)
```

Include `lge_mask_bullseye_main` in the `--help` test list. Add a pyproject assertion if the tests do not currently inspect console scripts:

```python
assert scripts["cardiac-lge-mask-bullseye"] == (
    "cardiac_bullseye.cli:lge_mask_bullseye_main"
)
```

Add a direct error-path assertion for the new public entry point:

```python
def test_lge_mask_bullseye_command_returns_one_on_validation_error(
    monkeypatch,
    tmp_path,
    capsys,
):
    from cardiac_bullseye import cli

    config_path = tmp_path / "case.toml"
    config_path.touch()
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda *_args: (_ for _ in ()).throw(ValueError("scar outside myocardium")),
    )

    assert cli.lge_mask_bullseye_main(["--config", str(config_path)]) == 1
    assert "scar outside myocardium" in capsys.readouterr().err
```

- [ ] **Step 3: Run the new workflow and CLI tests and verify the red state**

Run:

```powershell
python -m pytest tests/workflows/test_lge_mask_bullseye_workflow.py tests/cli/test_cli.py -v
```

Expected: import/parameter-table failures because the workflow and CLI entry point do not exist.

- [ ] **Step 4: Implement the workflow**

Create `workflows/lge_mask_bullseye.py`:

```python
@dataclass(frozen=True)
class LgeMaskBullseyeOutputs:
    landmarks: Path
    intensity_figure: Path
    scar_figure: Path


def generate_lge_mask_bullseye(config: AppConfig) -> LgeMaskBullseyeOutputs:
    validate_config(config, "lge-mask-bullseye")
    options = config.lge_mask_bullseye
    if options.lge is None or options.mask is None or options.lge_scar_mask is None:
        raise ValueError(
            "External-scar bullseye generation requires LGE, myocardium mask, "
            "and LGE scar mask inputs"
        )

    loaded = load_lge_study_with_scar(
        options.lge,
        options.mask,
        options.lge_scar_mask,
    )
    output_directory = config.case.output_directory
    output_directory.mkdir(parents=True, exist_ok=True)
    landmarks_path = config.case.output_path("lge_landmarks.json")
    intensity_path = config.case.output_path("lge_intensity_bullseye.png")
    scar_path = config.case.output_path("lge_scar_bullseye.png")

    landmarks = select_landmarks_interactively(
        loaded.study,
        landmarks_path,
        resume_existing=False,
        cancel_is_error=True,
    )
    deleted = load_deleted_slice_indices(
        landmarks_path,
        loaded.study.slice_count,
    )
    if not any(pair is not None for pair in landmarks):
        raise RuntimeError("No slice has a valid AIP/IIP pair")

    products = compute_lge_bullseye_products(
        loaded.study,
        landmarks,
        loaded.scar_mask,
        angle_count=options.angle_count,
        radial_count=options.radial_count,
        angular_smoothing_sigma_deg=options.angular_smoothing_sigma_deg,
        radial_smoothing_sigma=options.radial_smoothing_sigma,
        six_segment_warp=options.six_segment_warp,
        deleted_slice_indices=deleted,
    )
    save_lge_bullseye_figures(products, intensity_path, scar_path)
    return LgeMaskBullseyeOutputs(
        landmarks=landmarks_path,
        intensity_figure=intensity_path,
        scar_figure=scar_path,
    )
```

Export the dataclass and function from `workflows/__init__.py`.

- [ ] **Step 5: Register CLI and package entry points**

Import `generate_lge_mask_bullseye` in `cli.py`, then add:

```python
def lge_mask_bullseye_main(argv: Sequence[str] | None = None) -> int:
    """Render LGE bullseyes from an existing scar mask."""
    return _run(
        "lge-mask-bullseye",
        "cardiac-lge-mask-bullseye",
        generate_lge_mask_bullseye,
        argv,
    )
```

Register in `pyproject.toml`:

```toml
cardiac-lge-mask-bullseye = "cardiac_bullseye.cli:lge_mask_bullseye_main"
```

- [ ] **Step 6: Run workflow, CLI, and complete-LGE regression tests**

Run:

```powershell
python -m pytest tests/workflows/test_lge_mask_bullseye_workflow.py tests/workflows/test_lge_workflow.py tests/cli/test_cli.py -v
```

Expected: all tests pass. The new workflow directory contains exactly three outputs, while the complete workflow still returns seven.

- [ ] **Step 7: Commit the command and workflow**

```powershell
git add pyproject.toml src/cardiac_bullseye/cli.py src/cardiac_bullseye/workflows/__init__.py src/cardiac_bullseye/workflows/lge_mask_bullseye.py tests/cli/test_cli.py tests/workflows/test_lge_mask_bullseye_workflow.py
git diff --cached --name-only
git commit -m "feat: add external scar mask bullseye command"
```

---

### Task 6: Document the Three-Part LGE Workflow

**Files:**
- Modify: `README.md`
- Modify: `config.example.toml`
- Create: `toml/lge_mask_bullseye.example.toml`
- Modify: `tests/config/test_example_config.py`

**Interfaces:**
- Consumes: the finalized command, TOML field names, output names, and validation rules from Tasks 1-5.
- Produces: a copyable example and user-facing explanation of when to use each of the three capabilities.

- [ ] **Step 1: Write failing example-contract tests**

Extend `test_example_config_exercises_all_config_sections`:

```python
assert {
    "case",
    "short_axis",
    "bullseye",
    "lge_bullseye",
    "lge_mask_bullseye",
} <= raw_config.keys()
```

Add a test for the focused example:

```python
def test_lge_mask_bullseye_example_is_parseable(monkeypatch):
    example = Path(__file__).parents[2] / "toml" / "lge_mask_bullseye.example.toml"
    monkeypatch.setattr(Path, "is_file", lambda _path: True)

    config = load_config(example, "lge-mask-bullseye")

    assert config.case.id == "patient_043"
    assert config.lge_mask_bullseye.lge is not None
    assert config.lge_mask_bullseye.mask is not None
    assert config.lge_mask_bullseye.lge_scar_mask is not None
```

- [ ] **Step 2: Run the example tests and verify the red state**

Run:

```powershell
python -m pytest tests/config/test_example_config.py -v
```

Expected: failure because the new section and focused example are absent.

- [ ] **Step 3: Update the main example and README**

Add `[lge_mask_bullseye]` to `config.example.toml` with the three required paths and five mapping options.

Update `README.md` to:

- list five installed commands instead of four;
- show `cardiac-lge-mask-bullseye --config ...`;
- distinguish standalone `lgeMask`, external-scar bullseye, and complete bullseye;
- state that the new command asks only for fresh AIP/IIP annotations;
- document strict co-grid and scar-containment validation;
- list exactly the three new-command outputs;
- state that the complete command retains its full ROI/scar workflow and seven outputs.

- [ ] **Step 4: Create the focused TOML example**

Create `toml/lge_mask_bullseye.example.toml` with:

```toml
[case]
id = 'patient_043'
output_directory = '../outputs/patient_043_lge'

[lge_mask_bullseye]
lge = '../inputs/patient_043_lge_sax.nii.gz'
mask = '../inputs/patient_043_lge_myocardium.seg.nrrd'
lge_scar_mask = '../inputs/patient_043_lge_scar_mask.nii.gz'
angle_count = 360
radial_count = 300
angular_smoothing_sigma_deg = 1.5
radial_smoothing_sigma = 1.5
six_segment_warp = true
```

- [ ] **Step 5: Run documentation contract tests**

Run:

```powershell
python -m pytest tests/config/test_example_config.py tests/cli/test_cli.py -v
```

Expected: all tests pass and both example files parse under their intended workflow.

- [ ] **Step 6: Commit documentation**

```powershell
git add README.md config.example.toml toml/lge_mask_bullseye.example.toml tests/config/test_example_config.py
git diff --cached --name-only
git commit -m "docs: explain split LGE bullseye workflows"
```

---

### Task 7: Run Final Regression and Packaging Verification

**Files:**
- Verify: all production and test files changed in Tasks 1-6.

**Interfaces:**
- Consumes: the complete implementation.
- Produces: evidence that the package metadata, targeted workflows, and complete test suite pass.

- [ ] **Step 1: Verify package metadata contains the new command**

Run:

```powershell
python -c "import pathlib,tomllib; p=tomllib.loads(pathlib.Path('pyproject.toml').read_text(encoding='utf-8')); assert p['project']['scripts']['cardiac-lge-mask-bullseye']=='cardiac_bullseye.cli:lge_mask_bullseye_main'"
```

Expected: exit code `0` with no output.

- [ ] **Step 2: Run the complete LGE-focused suite**

Run:

```powershell
python -m pytest tests/lge tests/territory/test_landmarks.py tests/workflows/test_lge_common.py tests/workflows/test_lge_workflow.py tests/workflows/test_lge_mask_bullseye_workflow.py tests/config/test_config.py tests/config/test_example_config.py tests/cli/test_cli.py -v
```

Expected: all selected tests pass with no failures.

- [ ] **Step 3: Run the complete project test suite**

Run:

```powershell
python -m pytest -q
```

Expected: exit code `0`; no failed or errored tests.

- [ ] **Step 4: Inspect the final diff and output contract**

Run:

```powershell
git status --short
git diff --stat
rg -n "cardiac-lge-mask-bullseye|lge_mask_bullseye|lge_scar_mask" README.md config.example.toml pyproject.toml src tests toml
```

Expected: the new command and configuration names are consistent; no code path writes NPZ, CSV, ROI JSON, or a copied scar mask in `lge_mask_bullseye.py`; pre-existing unrelated working-tree changes remain untouched.

- [ ] **Step 5: Commit any narrowly scoped verification fixes**

If Steps 1-4 required a correction, stage only the corrected task files, inspect the cached diff, and commit:

```powershell
git diff --cached --name-only
git commit -m "test: complete external scar bullseye coverage"
```

If no correction was required, leave the tree unchanged and record the passing commands in the implementation handoff.
