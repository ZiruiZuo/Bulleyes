# Short-axis Source-view Controls Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three switchable source-volume views and display-only 90-degree rotations/flips to stage 1 of `cardiac-short-axis`, while preserving exact original voxel/world coordinates and leaving stages 2 and 3 unchanged.

**Architecture:** Put all 2-D extraction and reversible coordinate mapping in a small immutable `SourceViewTransform` component. `ShortAxisPlanner` will keep one transform per source view, render MR/mask/points through that component, and use the selected source axis normal when constructing the initial VLA. Saved planning geometry stays in world coordinates; only optional audit metadata records the accepted stage-1 display state.

**Tech Stack:** Python 3, NumPy, Matplotlib widgets, NiBabel, pytest

**Spec:** `docs/designs/2026-10-21-short-axis-view-controls-design.md`

## Global Constraints

- The UI labels are `View 1`, `View 2`, and `View 3`; they map to source voxel slice axes 2, 1, and 0 and must not be called anatomical axial/sagittal/coronal views.
- Rotations and flips are display-only: do not modify source arrays, headers, affines, world landmarks, saved short-axis orientation, or resampling geometry.
- Stage 2 VLA refinement and stage 3 VLA/HLA coverage planning retain their current behavior.
- Each source view independently remembers slice index, quarter-turn rotation, and horizontal/vertical flips.
- Switching view or slice clears unconfirmed stage-1 points; rotate, flip, and reset-view actions preserve them.
- MR and mask always use the same transform; mask interpolation is not introduced in stage 1.
- The UI shows one-based slice numbers, while saved `initial_view_display.slice_index` is the zero-based NIfTI array index.
- Old saved-point JSON remains loadable, and `reuse_saved_points = true` continues to bypass the GUI and ignore display audit metadata.
- Preserve existing public `build_vla_from_axial_points` behavior as a compatibility wrapper.
- Add no new runtime dependency.

## Review Focus

- Non-square planes rotated 90°/270° must swap display dimensions and still round-trip fractional coordinates exactly; Task 1 pins this.
- Oblique or sheared NIfTI affines must yield the same world location for the same source voxel under every display orientation; Task 4 pins this.
- Fractional boundary clicks are accepted only inside `[0, width-1] × [0, height-1]`, and out-of-image clicks fail without adding a point; Tasks 1 and 4 pin this.
- A source axis containing one slice must produce a usable fixed slider and must not lose that view's state; Task 3 pins this.
- Image/mask source-shape mismatch or transform divergence must be rejected instead of silently drawing a displaced overlay; Tasks 1 and 3 pin this.

---

### Task 1: Reversible source-view transform

**Files:**
- Create: `src/cardiac_bullseye/short_axis/source_view.py`
- Create: `tests/short_axis/test_source_view.py`

**Interfaces:**
- Consumes: a three-dimensional source shape, source slice axis/index, rotation count, flip flags, and a three-dimensional array.
- Produces: immutable `SourceViewTransform`, `VIEW_SLICE_AXES`, `display_shape`, `extract_plane()`, `voxel_to_display()`, `display_to_voxel()`, `contains_display_point()`, and `audit_dict()` for later planner tasks.

- [ ] **Step 1: Write failing construction and extraction tests**

Create tests that construct:

```python
SourceViewTransform(
    source_shape=(5, 7, 9),
    slice_axis=2,
    slice_index=4,
    rotation_quadrants=0,
    flip_horizontal=False,
    flip_vertical=False,
)
```

Assert that `VIEW_SLICE_AXES == (2, 1, 0)`, the base plane equals `np.take(volume, 4, axis=2).T`, `display_shape` is `(7, 5)`, and a volume whose voxels contain unique IDs places each ID at `voxel_to_display()`'s returned display coordinate. Parameterize all three slice axes.

- [ ] **Step 2: Run the extraction tests to verify they fail**

Run: `python -m pytest tests/short_axis/test_source_view.py -k "construction or extraction" -v`

Expected: FAIL because `cardiac_bullseye.short_axis.source_view` does not exist.

- [ ] **Step 3: Implement the immutable transform and base extraction**

Implement:

```python
VIEW_SLICE_AXES: tuple[int, int, int] = (2, 1, 0)

@dataclass(frozen=True)
class SourceViewTransform:
    source_shape: tuple[int, int, int]
    slice_axis: int
    slice_index: int
    rotation_quadrants: int = 0
    flip_horizontal: bool = False
    flip_vertical: bool = False

    @property
    def display_shape(self) -> tuple[int, int]: ...  # rows, columns
    def extract_plane(self, volume: np.ndarray) -> np.ndarray: ...
    def voxel_to_display(self, voxel_xyz: Iterable[float]) -> np.ndarray: ...
    def display_to_voxel(self, x: float, y: float) -> np.ndarray: ...
    def contains_display_point(self, x: float, y: float) -> bool: ...
    def audit_dict(self, view_number: int) -> dict[str, int | bool]: ...
```

Validate a positive three-dimensional shape, `slice_axis in {0, 1, 2}`, an in-range slice index, and exact volume/source shape equality. Normalize `rotation_quadrants` modulo 4. Extract with `np.take(..., axis=slice_axis).T`, then apply `np.rot90`, `np.fliplr`, and `np.flipud` in that order.

- [ ] **Step 4: Write failing exhaustive coordinate tests**

Parameterize axes `0, 1, 2`, rotations `0..3`, and both boolean flip flags. For a non-square `(5, 7, 9)` source, assert:

- `display_to_voxel(*voxel_to_display(voxel))` returns the original continuous voxel coordinate;
- 90°/270° swaps the base plane rows/columns;
- integer voxel locations agree with the value produced by `extract_plane()`;
- coordinates exactly on all four edges are accepted and `-0.01` or `last + 0.01` is rejected;
- invalid axis/index/shape and a mismatched array shape raise `ValueError`.

Use the following pre-flip coordinate rules, where `(W, H)` are the base display width/height:

```text
k=0: (x, y) -> (x,         y),         size (W, H)
k=1: (x, y) -> (y,         W - 1 - x), size (H, W)
k=2: (x, y) -> (W - 1 - x, H - 1 - y), size (W, H)
k=3: (x, y) -> (H - 1 - y, x),         size (H, W)
```

Apply horizontal and vertical reflection after rotation; inverse mapping reverses flips first and rotation second.

- [ ] **Step 5: Implement coordinate conversion, bounds checks, and audit serialization**

`display_to_voxel()` must raise `ValueError` for non-finite or out-of-display coordinates. `voxel_to_display()` must reject non-finite or out-of-source coordinates and must require the coordinate on the selected axis to match `slice_index` within `1e-6`. `audit_dict(view_number)` returns exactly `view`, `slice_axis`, `slice_index`, `rotation_quadrants`, `flip_horizontal`, and `flip_vertical` with native JSON scalar types.

- [ ] **Step 6: Run the complete transform tests**

Run: `python -m pytest tests/short_axis/test_source_view.py -v`

Expected: all tests PASS.

- [ ] **Step 7: Commit the transform component**

```bash
git add src/cardiac_bullseye/short_axis/source_view.py tests/short_axis/test_source_view.py
git commit -m "feat: add reversible short-axis source views"
```

### Task 2: Generalize first-stage VLA geometry

**Files:**
- Modify: `src/cardiac_bullseye/short_axis/geometry.py:164-176`
- Modify: `src/cardiac_bullseye/short_axis/__init__.py:1-22`
- Create: `tests/short_axis/test_source_plane_geometry.py`

**Interfaces:**
- Consumes: two stage-1 world points and the active source plane's world normal.
- Produces: `build_vla_from_plane_points(apex, valve, plane_normal)`; existing `build_vla_from_axial_points(apex, valve, axial_normal)` remains public and delegates to it.

- [ ] **Step 1: Write failing generalized-geometry tests**

For source axes 0, 1, and 2 from an oblique affine, call `voxel_axis_world_direction()` and assert that `build_vla_from_plane_points()` returns unit, mutually orthogonal axes; its x-axis follows the apex-to-valve line projected into the selected plane; and its y-axis equals the normalized selected plane normal. Add a compatibility test asserting element-for-element equality between the generalized function and `build_vla_from_axial_points()` for the current axis-2 case.

- [ ] **Step 2: Run the geometry tests to verify they fail**

Run: `python -m pytest tests/short_axis/test_source_plane_geometry.py -v`

Expected: FAIL because `build_vla_from_plane_points` is not defined.

- [ ] **Step 3: Implement and export the generalized function**

Add:

```python
def build_vla_from_plane_points(
    apex: Iterable[float],
    valve: Iterable[float],
    plane_normal: Iterable[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
```

Move the existing projection/cross-product operation into this function with generalized error labels. Retain `build_vla_from_axial_points()` as a thin compatibility wrapper and export both names from `short_axis.__init__`.

- [ ] **Step 4: Run focused geometry tests**

Run: `python -m pytest tests/short_axis/test_source_plane_geometry.py -v`

Expected: all tests PASS.

- [ ] **Step 5: Run existing short-axis geometry consumers**

Run: `python -m pytest tests/short_axis/test_orientation_reference.py tests/short_axis/test_saved_points.py -v`

Expected: all tests PASS.

- [ ] **Step 6: Commit generalized geometry**

```bash
git add src/cardiac_bullseye/short_axis/geometry.py src/cardiac_bullseye/short_axis/__init__.py tests/short_axis/test_source_plane_geometry.py
git commit -m "refactor: generalize initial VLA source plane"
```

### Task 3: Add independent planner view state and transformed rendering

**Files:**
- Modify: `src/cardiac_bullseye/short_axis/planner.py:67-255,376-425,465-491`
- Create: `tests/short_axis/test_source_view_planner.py`

**Interfaces:**
- Consumes: `VIEW_SLICE_AXES`, `SourceViewTransform`, and `build_vla_from_plane_points()` from Tasks 1 and 2.
- Produces: planner state `active_source_view`, `source_views`, `accepted_initial_view_display`; transformed stage-1 rendering/click mapping; and generalized initial VLA construction.

- [ ] **Step 1: Write failing planner state and rendering tests**

Build small `NiftiVolume` fixtures with a non-square shape and matching labeled mask. Assert a newly constructed planner has View 1 active and three transforms at their respective middle slices. Switch among views and assert each preserves its own slice/orientation state. For every view, assert the displayed image and mask arrays equal `transform.extract_plane()` output and remain pixel-aligned.

Add the Review Focus cases: a one-voxel source axis yields `slice_slider.valmin == slice_slider.valmax == 0` without crashing, and construction still rejects an image/mask grid mismatch through `validate_mask_grid()`.

- [ ] **Step 2: Run the planner state tests to verify they fail**

Run: `python -m pytest tests/short_axis/test_source_view_planner.py -k "initial or state or rendering or single_slice or mask_grid" -v`

Expected: FAIL because the planner has only `axial_index` and `_render_axial()`.

- [ ] **Step 3: Replace fixed axial state with three source-view transforms**

Use stage name `"source"`, `active_source_view: int = 1`, and:

```python
self.source_views: dict[int, SourceViewTransform] = {
    view: SourceViewTransform(image.data.shape, axis, image.data.shape[axis] // 2)
    for view, axis in enumerate(VIEW_SLICE_AXES, start=1)
}
```

Rename `axial_points` to `source_points`. Replace `_render_axial()` with `_render_source_selection()`, using the active transform for image, mask, point projection, one-based title text, and slice slider synchronization. Remove `invert_xaxis()` because every displayed orientation is represented explicitly in the transform.

Add a `_syncing_slice_slider` guard. `_sync_source_slider()` updates label to `Slice`, `valmin`, `valmax`, axes limits, and current value without triggering point clearing; a one-slice axis remains fixed at zero.

- [ ] **Step 4: Write failing point-lifecycle and navigation tests**

Assert that changing source view or slice clears stage-1 points and sets an explanatory message; changing rotation/flip fields and reset-view preserves points; returning Back from stage 2 restores the accepted source view, slice, orientation, and stage-1 points. Assert advancing from each view calls the generalized geometry with `voxel_axis_world_direction(image.affine, active_transform.slice_axis)` and stores `accepted_initial_view_display` from the accepted transform.

- [ ] **Step 5: Implement source navigation, point lifecycle, and stage transition**

Add helpers `_active_source_transform()`, `_replace_active_source_transform(**changes)`, `_set_source_view(view_number)`, and `_source_stage_title()`. The title reports active view, one-based slice number, rotation in degrees, and each enabled flip exactly as specified. View/slice handlers clear only unconfirmed stage-1 points; display-orientation handlers do not. `_on_next()` stores `source_points`, records `active_transform.audit_dict(active_source_view)`, and uses `build_vla_from_plane_points()` with the active slice-axis normal. `_on_back()` from stage 2 restores `source_points` and calls `_render_source_selection()`.

- [ ] **Step 6: Run planner and existing regression tests**

Run: `python -m pytest tests/short_axis/test_source_view_planner.py tests/short_axis/test_planner.py tests/short_axis/test_reference_save.py -v`

Expected: all tests PASS.

- [ ] **Step 7: Commit planner state and rendering**

```bash
git add src/cardiac_bullseye/short_axis/planner.py tests/short_axis/test_source_view_planner.py
git commit -m "feat: render switchable short-axis source views"
```

### Task 4: Add stage-1 controls and prove click invariance

**Files:**
- Modify: `src/cardiac_bullseye/short_axis/planner.py:127-159,228-316,376-463`
- Modify: `tests/short_axis/test_source_view_planner.py`

**Interfaces:**
- Consumes: the planner view state and transform methods from Task 3.
- Produces: visible View 1/2/3, Rotate Left/Right, Flip H/V, Reset View, and Clear Points controls in stage 1 only; exact display-click-to-world conversion.

- [ ] **Step 1: Write failing control visibility and behavior tests**

Assert stage 1 exposes buttons labeled exactly `View 1`, `View 2`, `View 3`, `Rotate Left`, `Rotate Right`, `Flip H`, `Flip V`, `Reset View`, and `Clear Points`. Invoke their handlers and assert view state changes as specified. After moving to stage 2 or 3, assert all source-view button axes and the source slider are hidden; Back to stage 1 restores them. Confirm the existing stage-2 point reset and stage-3 boundary reset behavior remains available through the existing stage-specific reset action.

- [ ] **Step 2: Run control tests to verify they fail**

Run: `python -m pytest tests/short_axis/test_source_view_planner.py -k "controls or button or visibility" -v`

Expected: FAIL because the new widgets and handlers do not exist.

- [ ] **Step 3: Create and wire source-view widgets**

Create the nine stage-1 buttons in a dedicated collection so `_set_source_controls_visible(bool)` can show/hide them together. Add handlers `_on_rotate_left`, `_on_rotate_right`, `_on_flip_horizontal`, `_on_flip_vertical`, `_on_reset_view`, and `_on_clear_points`. Quarter turns are modulo 4; Reset View sets only rotation to 0 and both flips to false; Clear Points changes no view state. Keep stages 2/3 controls and logic unchanged.

- [ ] **Step 4: Write failing affine and boundary click tests**

With a non-square volume and a nontrivial affine containing rotation, unequal scale, shear, and translation, choose one valid continuous source voxel on each view. For all rotations and flip combinations, convert it to display coordinates, synthesize an event on `ax_left`, and assert `_click_to_world()` equals `affine @ voxel` within `1e-8`. Assert clicks just outside each displayed edge return `None`, set an error status, and do not append a landmark.

- [ ] **Step 5: Implement transform-based click and marker conversion**

In stage `"source"`, `_click_to_world()` calls `active_transform.display_to_voxel(event.xdata, event.ydata)` and then applies the original image affine. Catch coordinate `ValueError`, show a concise invalid-click status, and return `None`. Point rendering applies the inverse source affine and `voxel_to_display()`; it must never transform or overwrite the saved world point.

- [ ] **Step 6: Run stage-1 interaction tests**

Run: `python -m pytest tests/short_axis/test_source_view_planner.py -v`

Expected: all tests PASS.

- [ ] **Step 7: Commit interactive controls**

```bash
git add src/cardiac_bullseye/short_axis/planner.py tests/short_axis/test_source_view_planner.py
git commit -m "feat: add short-axis source view controls"
```

### Task 5: Save optional display audit metadata

**Files:**
- Modify: `src/cardiac_bullseye/short_axis/saved_points.py:45-101`
- Modify: `src/cardiac_bullseye/short_axis/planner.py:518-595`
- Modify: `tests/short_axis/test_saved_points.py`
- Modify: `tests/short_axis/test_reference_save.py`

**Interfaces:**
- Consumes: `accepted_initial_view_display: dict[str, int | bool]` recorded when stage 1 is accepted.
- Produces: optional `initial_view_display` in newly saved point JSON; loader and noninteractive reuse remain display-state agnostic.

- [ ] **Step 1: Write failing metadata persistence and compatibility tests**

Call `save_short_axis_points(..., initial_view_display={...})` with the exact six fields from the spec and assert the JSON object is preserved with zero-based `slice_index`. Save/load another file without the new argument and assert it still loads. Add an unknown `initial_view_display` object to an old-style fixture and assert `load_short_axis_points()` and `convert_to_short_axis_from_saved_points()` ignore it.

In the planner save test, set `accepted_initial_view_display`, call `_save()`, and assert it reaches the point JSON unchanged.

- [ ] **Step 2: Run saved-point tests to verify they fail**

Run: `python -m pytest tests/short_axis/test_saved_points.py tests/short_axis/test_reference_save.py -v`

Expected: FAIL because `save_short_axis_points()` does not accept or write `initial_view_display`.

- [ ] **Step 3: Add the optional save-only argument and planner pass-through**

Extend the keyword-only signature with:

```python
initial_view_display: Mapping[str, int | bool] | None = None
```

When non-`None`, write a shallow JSON-compatible copy under `initial_view_display`; otherwise omit the key entirely. Do not add display state to `SavedShortAxisPoints`, and do not consume it in regeneration. Pass the accepted planner snapshot from `_save()`.

- [ ] **Step 4: Run saved-point tests**

Run: `python -m pytest tests/short_axis/test_saved_points.py tests/short_axis/test_reference_save.py -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit metadata support**

```bash
git add src/cardiac_bullseye/short_axis/saved_points.py src/cardiac_bullseye/short_axis/planner.py tests/short_axis/test_saved_points.py tests/short_axis/test_reference_save.py
git commit -m "feat: record initial short-axis display state"
```

### Task 6: Document controls and run complete verification

**Files:**
- Modify: `README.md:45-125`

**Interfaces:**
- Consumes: the completed UI behavior and compatibility rules.
- Produces: user-facing instructions for the first stage of `cardiac-short-axis`.

- [ ] **Step 1: Update the short-axis README section**

Document that the first stage can switch among three source voxel-axis views; state the exact mapping View 1/2/3 → axes 2/1/0; explain Slice, Rotate Left/Right, Flip H/V, Reset View, and Clear Points; emphasize that rotation/flips affect display only; and state that changing view/slice clears the pending Apex/Mitral pair. Preserve the existing `reuse_saved_points` instructions and explain that reuse bypasses all interactive view controls.

- [ ] **Step 2: Run formatting/static checks configured by the project**

Inspect `pyproject.toml` for configured format/lint commands and run only those present. If none are configured, run:

```bash
python -m compileall -q src
```

Expected: exit code 0.

- [ ] **Step 3: Run the complete short-axis suite**

Run: `python -m pytest tests/short_axis -v`

Expected: all tests PASS.

- [ ] **Step 4: Run the complete project suite**

Run: `python -m pytest -q`

Expected: all tests PASS with no regression in territory or LGE workflows.

- [ ] **Step 5: Manually exercise the GUI smoke path**

Using a small local NIfTI and matching mask, open `cardiac-short-axis`, verify all three views and transforms redraw both layers together, place points before/after transforms, navigate Next/Back, and close without saving unwanted outputs. Record the tested command and observations in the implementation handoff.

- [ ] **Step 6: Review the final diff for scope and saved-data compatibility**

Confirm there are no source-array/affine mutations, no anatomical naming claims, no stage-2/3 behavioral changes, and no edits to unrelated existing working-tree changes. Confirm the accepted stage-1 metadata is audit-only and old point JSON still works.

- [ ] **Step 7: Commit documentation**

```bash
git add README.md
git commit -m "docs: explain short-axis source view controls"
```
