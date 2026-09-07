# Matched MRA/LGE Bullseye Workflow Design

Date: 2026-09-07

## Objective

Add a reusable workflow that produces anatomically normalized, mutually
comparable bullseye maps from an MRA short-axis stack and an LGE short-axis
stack. The two stacks do not need voxelwise or slice-by-slice registration.
They are matched through a common normalized apex-to-base coordinate and
independently annotated AIP/IIP angular frames.

The workflow must generate:

- continuous LAD/LCX/RCA territory probabilities;
- a coronary projection overlay;
- distance-to-coronary values in millimetres;
- continuous raw LGE intensity;
- persistent MRA and LGE AIP/IIP landmarks;
- numeric arrays as well as figures for later responsibility-vessel analysis.

No LGE thresholding or scar classification is included in this phase.

## Inputs for case 020501

MRA inputs:

- `outputs/020501_mra/020501_mra_short_axis.nii.gz`
- `outputs/020501_mra/020501_mra_short_axis_mask.nii.gz` (binary myocardium)
- `outputs/020501_mra/020501_mra_geometry.json`
- `Data/HospitalData/ZhongNan/Labels/label_020501.nii.gz` (coronaries)
- `Data/HospitalData/ZhongNan/MyoLabels/label_020501.nii.gz` (world myocardium)

LGE inputs:

- `ANTsRegistration/outputs/020501/prepared/lge_sax.nii.gz`
- `Data/HospitalData/ZhongNan/LGE/020501-liu hong kui/lge_mask_020501.seg.nrrd`

Coronary labels for this case are `1=LAD`, `2=LCX`, and `3=RCA`.

## Territory contract

Territory partitioning must call the existing function
`E:/ZZR/Code/Tools/territory.py::compute_territory`. Its skeletonization,
vessel-to-myocardium projection, physical-distance calculation, and nearest
territory assignment are authoritative for territory generation and must not
be copied into Bulleyes. The existing Bulleyes projection and distance modules
remain responsible for their own projection overlay and distance-map outputs;
they do not replace or modify the territory returned by `compute_territory`.

Bulleyes will provide a small adapter that:

1. loads coronary and myocardium images as SimpleITK `(z, y, x)` arrays;
2. verifies identical source geometry;
3. constructs `{1: coronary == 1, 2: coronary == 2, 3: coronary == 3}`;
4. calls `compute_territory(..., spacing=image.GetSpacing()[::-1])`;
5. saves the returned world-space territory with the myocardium geometry;
6. samples those hard labels at the MRA SAX voxel-centre world coordinates
   described by the existing short-axis geometry, using nearest-neighbour
   interpolation.

Existing Bulleyes angular and longitudinal smoothing then converts those hard
labels to continuous LAD/LCX/RCA probabilities. The external function remains
the only implementation of the territory partition.

## MRA map workflow

1. Validate the MRA SAX image, binary SAX myocardium, and geometry JSON,
   including the SAX array shape and affine reconstructed from the JSON.
2. Generate world and SAX territory labels through the territory adapter.
3. Compute the existing world-space distance map and linearly resample it to
   MRA SAX, restricted to the binary myocardium.
4. Persist MRA AIP/IIP annotations to `<case>_mra_landmarks.json`; resume
   unfinished annotations on later runs.
5. Use the existing territory interpolation to generate a common
   `(radial_count, angle_count)` probability grid.
6. Use the same landmarks and grid for the distance bullseye.
7. Use the existing world-space coronary path projection and the same angular
   frame to draw the coronary overlay.

## LGE intensity workflow

1. Load the LGE NIfTI and the Slicer segmentation NRRD through SimpleITK.
2. Reduce all myocardium segments/layers to one binary mask. When its grid is
   not identical to LGE but has valid physical metadata, resample it to the LGE
   grid with nearest-neighbour interpolation; fail if the transformed mask has
   no overlap with the LGE volume.
3. Crop the internal study to the first-through-last slices containing
   myocardium. A configurable `lge_reverse_slice_order` flag controls whether
   this range must be reversed to run from apex to base.
4. Persist LGE AIP/IIP annotations to `<case>_lge_landmarks.json`; resume them
   on later runs.
5. For each valid slice and angular bin, calculate the mean raw LGE intensity
   of myocardial pixels in that bin. Apply the same optional angular smoothing
   and six-segment AIP/IIP warp used by the MRA maps.
6. Interpolate the slice profiles onto the exact radial positions and angles
   of the MRA territory bullseye. Do not threshold, classify, or normalize the
   intensity values.

The LGE result is therefore matched to MRA in normalized cardiac coordinates,
not in scanner world coordinates.

## Configuration and command

Add a separate `[matched_bullseye]` configuration section and a
`cardiac-matched-bullseye` command so the existing `cardiac-bullseye` contract
continues to work unchanged for callers that already have a territory mask.

The new section contains the paths listed above, the external
`territory_module_path`, `vessel_labels = [1, 2, 3]`,
`lge_reverse_slice_order`, common angle/radial counts and smoothing settings,
and display controls.

## Outputs

The workflow writes at least:

- `<case>_territory_world.nii.gz`
- `<case>_territory_short_axis.nii.gz`
- `<case>_projected_vessels_world.nii.gz`
- `<case>_distance_world.nii.gz`
- `<case>_distance_short_axis.nii.gz`
- `<case>_mra_landmarks.json`
- `<case>_lge_landmarks.json`
- `<case>_territory_bullseye.npz`
- `<case>_distance_bullseye.npz`
- `<case>_lge_intensity_bullseye.npz`
- `<case>_territory_bullseye.png`
- `<case>_distance_bullseye.png`
- `<case>_lge_intensity_bullseye.png`
- `<case>_matched_bullseyes.png`
- `<case>_matched_coordinate_report.json`

The NPZ files include the common radial positions and canonical angles. The
matched report records input paths, label convention, slice ranges, completed
landmark counts, array shapes, and confirmation that all maps share the same
bullseye coordinate arrays.

## Failure handling and quality control

The command stops before annotation if source geometries are incompatible,
required labels are missing, or MRA geometry does not match its SAX image.
After annotation it stops if no valid AIP/IIP pair exists. It warns when too
few slices are annotated or when LGE and MRA effective apex/base coverage is
clearly incomplete.

The combined figure displays territory with coronary overlay, distance, and
LGE intensity using the same AHA boundaries. This is a normalized-coordinate
QC view; it must not claim voxelwise MRA/LGE registration.

## Testing

Tests will cover:

- dynamic reuse of `territory.py::compute_territory` with labels 1/2/3;
- nearest-neighbour territory resampling and linear distance resampling;
- raw LGE angular means on synthetic myocardium;
- radial interpolation onto an existing MRA bullseye grid;
- AIP/IIP landmark persistence and resume behavior;
- equality of MRA/LGE radial and angular coordinate arrays;
- configuration validation and command dispatch;
- an end-to-end noninteractive synthetic workflow with saved landmarks.
