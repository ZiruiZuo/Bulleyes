# Standalone LGE Scar Mask Tool Design

## Goal

Create a self-contained, portable command-line tool that loads a short-axis
LGE volume and a co-gridded myocardium mask, lets the user paint a normal
myocardium reference ROI on each analyzed slice, applies a per-slice
`mean + 5 * sample standard deviation` threshold, and writes a binary LGE scar
mask.

This phase does not change `cardiac-lge-bullseye`. Compatibility between the
new scar mask and that command will be implemented separately when requested.

## Location and Portability

The tool will live as a sibling of `Bulleyes` under the shared Tools directory:

```text
E:/ZZR/Code/Tools/lgeMask/
├── main.py
├── image_io.py
├── roi_selector.py
├── scar.py
├── requirements.txt
├── README.md
└── tests/
```

The tool must not import `cardiac_bullseye` or require the `Bulleyes` project to
be installed. Copying this one folder to another Windows computer, installing
its `requirements.txt`, and running `main.py` must be sufficient.

## Command-Line Interface

The executable entry point is `main.py` with the standard guard:

```python
if __name__ == "__main__":
    raise SystemExit(main())
```

The command accepts exactly three required user-facing parameters:

```powershell
python main.py --lge lge_sax.nii.gz --mask myocardium_mask.nii.gz --output lge_scar_mask.nii
```

- `--lge`: scalar three-dimensional short-axis LGE image.
- `--mask`: three-dimensional myocardium mask on the exact LGE grid. Any
  nonzero value is myocardium. A vector mask is accepted only when its spatial
  grid matches the LGE; nonzero components are combined.
- `--output`: destination `.nii` or `.nii.gz` binary scar mask. An existing
  file is overwritten.

The 5-SD multiplier and the minimum 20-voxel ROI size are fixed clinical
workflow constants in this version and are not exposed as additional command
parameters.

## Input Validation and Image Geometry

`SimpleITK` will read both images. Before annotation, the tool will verify:

- both inputs exist and can be read;
- both images are three-dimensional;
- the LGE is a scalar image;
- image size, spacing, origin, and direction match within the same tolerances
  used by the existing LGE workflow;
- the myocardium mask is not empty;
- the output filename ends in `.nii` or `.nii.gz`.

The tool will not resample the mask. Geometry mismatch produces a concise error
and a nonzero process exit code.

Only the continuous slice range from the first through the last slice that
contains myocardium is shown to the user. The saved scar mask is restored to
the complete original LGE grid.

## ROI Annotation Interaction

Every run starts with blank ROIs. The tool does not load a previous ROI file
and does not persist ROI annotations for reuse.

Each nonempty myocardium slice is displayed in sequence. The display uses the
same tested orientation transform as the existing annotation interface so the
image and mask overlay remain aligned.

Controls:

- left-button drag paints the normal-myocardium ROI;
- right-button drag erases it;
- mouse wheel zooms around the cursor;
- middle-button drag pans;
- a slider changes the brush radius;
- `Undo` restores the previous brush stroke;
- `Clear` removes the current slice ROI;
- `Fit` restores the full-image view;
- `Accept` validates and accepts the current slice;
- `Skip` excludes the current slice from scar analysis;
- `Save & Quit` stops the run without producing a final scar mask.

Painting is clipped continuously to finite LGE voxels inside the myocardium
mask. The user therefore cannot accidentally include blood pool, background,
or nonfinite image values in the reference ROI.

When `Accept` is clicked, the tool counts valid selected myocardial voxels. If
the count is below 20, it displays a clear warning containing the current count
and the required minimum, remains on the current slice, and lets the user add
or correct the ROI. A valid `Accept` advances to the next slice.

When `Skip` is clicked, no threshold is computed for that slice. The slice is
excluded from analysis, its output scar mask is all zero, and annotation moves
to the next slice. `Skip` is allowed on any displayed slice, including slices
near the base.

Empty myocardium slices inside the continuous cropped range do not open an
annotation window and remain zero in the output.

## Scar Calculation

For each accepted slice, let the valid reference ROI values be
`x_1, ..., x_n`, with `n >= 20`. The tool computes:

```text
mean_i = arithmetic mean(x)
std_i = sample standard deviation(x), using ddof=1
threshold_i = mean_i + 5 * std_i
```

A voxel is scar only when all of these conditions hold:

- it lies inside the myocardium mask;
- its LGE value is finite;
- the slice was accepted rather than skipped;
- its LGE value is strictly greater than `threshold_i`.

The computation is independent for every accepted short-axis slice. No AIP,
IIP, AHA segment assignment, or neighboring-slice interpolation is required.

## Output

After every required slice has been accepted or skipped, the tool writes one
binary `uint8` NIfTI image:

- `1` indicates threshold-defined scar;
- `0` indicates other myocardium, background, an empty slice, or a skipped
  slice.

The output has the complete original LGE dimensions and copies the LGE spacing,
origin, and direction. The parent output directory is created when necessary.
If the output already exists, it is replaced by the new result.

The console prints a short per-slice summary for accepted slices containing ROI
voxel count, mean, sample standard deviation, threshold, and scar voxel count.
Skipped slices are explicitly reported as skipped. A final line reports the
saved output path.

## Dependencies and Installation

`requirements.txt` contains only runtime dependencies needed by this folder:

```text
numpy>=1.26
matplotlib>=3.8
PyQt5>=5.15
SimpleITK>=2.3
```

The README documents creating a Python 3.10 or newer virtual environment,
installing with `python -m pip install -r requirements.txt`, running the tool,
GUI controls, the fixed 5-SD method, and input/output requirements.

## Error and Cancellation Behavior

Validation and calculation errors are printed as `Error: <message>` to stderr,
and the command exits with status 1. Successful completion exits with status 0.

Closing the annotation window or clicking `Save & Quit` cancels the workflow.
Cancellation does not create or overwrite the requested output scar mask. This
prevents a partial annotation session from being mistaken for a completed
result.

## Testing Strategy

Automated tests will cover:

- command-line parsing and exit codes;
- exact-grid validation and mismatch rejection;
- fixed per-slice mean plus five sample-SD calculation;
- strict greater-than thresholding inside myocardium only;
- accepted, skipped, and empty-slice behavior;
- ROI clipping and continuous paint/erase brush strokes;
- rejection of `Accept` when fewer than 20 valid voxels are selected;
- advancement after a valid `Accept` and after `Skip`;
- cancellation preserving any existing output file;
- restoration of the complete image grid and physical metadata;
- overwriting an existing `.nii` or `.nii.gz` output;
- independence from imports in the parent `cardiac_bullseye` package.

GUI tests will simulate Matplotlib events using a noninteractive backend where
possible. A final smoke check will run command help from inside the standalone
folder.
