"""Console entry points for cardiac bullseye workflows."""

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path
import sys

from .config import AppConfig, ConfigError, WorkflowName, load_config
from .workflows import (
    generate_bullseye,
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
    return _run("bullseye", "cardiac-bullseye", generate_bullseye, argv)


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
