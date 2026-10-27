"""Console entry points for cardiac bullseye workflows."""

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path
import sys

from .config import AppConfig, ConfigError, WorkflowName, load_config
from .workflows import (
    generate_bullseye,
    generate_intensity_bullseye,
    generate_lge_bullseye,
    generate_lge_mask_bullseye,
    generate_short_axis,
    run_complete_pipeline,
)


def _run(
    workflow_name: WorkflowName,
    program_name: str,
    runner: Callable[[AppConfig], object],
    argv: Sequence[str] | None,
) -> int:
    parser = argparse.ArgumentParser(prog=program_name)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, workflow_name)
        runner(config)
    except (ConfigError, OSError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


def short_axis_main(argv: Sequence[str] | None = None) -> int:
    """Run the short-axis workflow command."""
    return _run("short-axis", "cardiac-short-axis", generate_short_axis, argv)


def bullseye_main(argv: Sequence[str] | None = None) -> int:
    """Run the bullseye workflow command."""

    parser = argparse.ArgumentParser(prog="cardiac-bullseye")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--mode",
        choices=("per-slice", "single-reference"),
        default=None,
        help=(
            "AIP/IIP annotation mode: select every slice or propagate one "
            "reference slice (default: per-slice)"
        ),
    )
    parser.add_argument(
        "--projection",
        choices=("1-4", "0-4"),
        default=None,
        help=(
            "Radial projection: first myocardial mask slice at radius 1 "
            "or radius 0 (default: 1-4)"
        ),
    )
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, "bullseye")
        if args.mode is None and args.projection is None:
            generate_bullseye(config)
        elif args.mode is None:
            generate_bullseye(config, projection=args.projection)
        elif args.projection is None:
            generate_bullseye(config, landmark_mode=args.mode)
        else:
            generate_bullseye(
                config,
                landmark_mode=args.mode,
                projection=args.projection,
            )
    except (ConfigError, OSError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


def intensity_bullseye_main(argv: Sequence[str] | None = None) -> int:
    """Run the continuous image-intensity bullseye command."""

    parser = argparse.ArgumentParser(prog="cardiac-intensity-bullseye")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--mode",
        choices=("per-slice", "single-reference"),
        default=None,
        help=(
            "AIP/IIP annotation mode: select every slice or propagate one "
            "reference slice (default: per-slice)"
        ),
    )
    parser.add_argument(
        "--projection",
        choices=("1-4", "0-4"),
        default=None,
        help=(
            "Radial projection: first myocardial mask slice at radius 1 "
            "or radius 0 (default: 1-4)"
        ),
    )
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, "intensity-bullseye")
        if args.mode is None and args.projection is None:
            generate_intensity_bullseye(config)
        elif args.mode is None:
            generate_intensity_bullseye(config, projection=args.projection)
        elif args.projection is None:
            generate_intensity_bullseye(config, landmark_mode=args.mode)
        else:
            generate_intensity_bullseye(
                config,
                landmark_mode=args.mode,
                projection=args.projection,
            )
    except (ConfigError, OSError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


def pipeline_main(argv: Sequence[str] | None = None) -> int:
    """Run the complete pipeline command."""
    return _run("pipeline", "cardiac-pipeline", run_complete_pipeline, argv)


def lge_bullseye_main(argv: Sequence[str] | None = None) -> int:
    """Run the standalone raw-intensity LGE bullseye command."""

    return _run(
        "lge-bullseye",
        "cardiac-lge-bullseye",
        generate_lge_bullseye,
        argv,
    )


def lge_mask_bullseye_main(argv: Sequence[str] | None = None) -> int:
    """Render LGE bullseyes from an existing scar mask."""

    return _run(
        "lge-mask-bullseye",
        "cardiac-lge-mask-bullseye",
        generate_lge_mask_bullseye,
        argv,
    )
