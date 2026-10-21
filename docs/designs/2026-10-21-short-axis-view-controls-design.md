# Short-axis source-view controls

## Purpose

`cardiac-short-axis` currently begins planning from a fixed view of the third
NIfTI voxel axis. Acquisition protocols may store the heart with different
axis orders and display orientations, making the initial landmark selection
awkward or misleading.

The first planning stage will allow the user to switch among the three
orthogonal voxel views and adjust the two-dimensional display with lossless
90-degree rotations and flips before selecting the initial apex and mitral
centre. These controls affect presentation only. They must not modify source
arrays, NIfTI affines, world coordinates, or final short-axis orientation.

## Scope

The feature applies only to stage 1 of `ShortAxisPlanner`, where the initial
apex and mitral centre are selected on a source-volume slice.

Stage 2 VLA landmark refinement, stage 3 VLA/HLA coverage adjustment,
short-axis resampling, and the meaning of `orientation_reference` remain
unchanged.

The views will be named `View 1`, `View 2`, and `View 3`. They correspond to
slicing the source array along voxel axes 2, 1, and 0 respectively. The UI
will not claim that these are anatomical axial, sagittal, or coronal planes.

## Display state

Each source view owns an independent display state:

- source slice axis;
- current slice index;
- counter-clockwise rotation in quarter turns, from 0 through 3;
- horizontal flip flag;
- vertical flip flag.

The initial state uses View 1, the middle slice of each view, zero rotation,
and no flips. Switching away from a view and returning restores that view's
slice and display state.

The original `NiftiVolume.data`, mask data, headers, and affines are never
modified.

## Coordinate transformation

A dedicated display-coordinate component will be responsible for both image
rendering and coordinate conversion. It must define one invertible transform
for a given source shape and display state.

Forward display flow:

1. Extract a two-dimensional plane with `numpy.take` along the selected slice
   axis.
2. Transpose the plane into Matplotlib row/column display convention.
3. Apply the selected number of `numpy.rot90` quarter turns.
4. Apply horizontal and vertical flips.
5. Apply the identical operations to the optional mask.

World-point display flow:

1. Transform a world point through the inverse source affine to obtain its
   original continuous voxel coordinates.
2. Project the two in-plane voxel coordinates through the forward discrete
   display transform.
3. Plot the marker in the transformed view.

Click-to-world flow:

1. Reject clicks outside the displayed image.
2. Undo vertical and horizontal flips.
3. Undo the quarter-turn rotation.
4. Undo the base transpose and insert the active slice coordinate.
5. Validate the resulting continuous voxel coordinate against the original
   source bounds.
6. Transform the voxel coordinate through the original source affine.

Rotation and flip operations therefore alter only the appearance of the
plane. A click on the same source location must resolve to the same world
coordinate under every display orientation.

## Planner geometry

The current first-stage geometry assumes source voxel axis 2 and calls
`build_vla_from_axial_points`. This operation will be generalized to construct
the initial VLA from points selected on any source plane.

When stage 1 is accepted:

1. Use the two saved world points as the in-plane apex-to-mitral direction.
2. Obtain the selected plane normal with
   `voxel_axis_world_direction(image.affine, active_slice_axis)`.
3. Build the VLA grid from that direction and plane normal.
4. Continue into the existing VLA refinement stage.

The numerical operation is the existing plane construction with generalized
naming and an arbitrary source-plane normal. The old View 1 path must remain
numerically equivalent to the current implementation.

## User interface

Stage 1 will expose:

- `View 1`, `View 2`, and `View 3` buttons;
- a shared `Slice` slider whose bounds and value follow the active view;
- `Rotate Left` and `Rotate Right` buttons;
- `Flip H` and `Flip V` buttons;
- `Reset View`, which resets only the active view's rotation and flips;
- `Clear Points`, which clears the initial apex and mitral selections;
- the existing `Next` action.

The stage title will include the active view, one-based slice number, rotation,
and enabled flips. For example:

```text
Step 1/3 — View 2 — Slice 38 — Rotate 90° — Flip H
```

Changing the active view or slice clears unconfirmed landmark points and
shows a status message explaining why. Rotating, flipping, or resetting the
display preserves selected points and redraws them at their transformed
locations.

All source-view controls are hidden in stages 2 and 3. Returning from stage 2
with `Back` restores the previously active source view and its display state.

## Saved points and compatibility

Planning points continue to be saved in world coordinates. The points JSON
will gain an optional audit-only object:

```json
{
  "initial_view_display": {
    "view": 2,
    "slice_axis": 1,
    "slice_index": 38,
    "rotation_quadrants": 1,
    "flip_horizontal": true,
    "flip_vertical": false
  }
}
```

The display object does not participate in noninteractive regeneration from
saved points. `reuse_saved_points = true` continues to bypass the UI and use
the saved world-space plan. Loaders must accept both old files without this
object and new files containing it.

## Error handling

- MR and mask must use the same display transform at all times.
- A click that cannot map to a valid original continuous voxel coordinate is
  rejected with a status message.
- View and slice changes clear unconfirmed points so that a landmark pair
  cannot accidentally span two planes.
- Existing validation continues to reject coincident points and degenerate
  long-axis geometry.
- No display operation writes to a source NIfTI array or affine.

## Testing

Automated tests will cover:

- forward and inverse coordinate conversion for all three views;
- all quarter-turn rotations and horizontal/vertical flip combinations;
- non-square source planes, where rotation changes displayed dimensions;
- invariance of world coordinates under display-only transforms;
- identical MR and mask transformations;
- independent state restoration for each view;
- point clearing after view or slice changes;
- point preservation after rotation, flipping, and view reset;
- initial VLA construction from source axes 0, 1, and 2;
- numerical compatibility of the existing View 1 planning path;
- loading old saved-point JSON files;
- persistence of the optional display audit metadata;
- the complete project test suite.

## Documentation

The README will describe the stage-1 controls, emphasize that the view names
refer to source voxel axes rather than asserted anatomical planes, and state
that all rotations and flips are display-only.
