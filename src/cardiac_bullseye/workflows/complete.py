"""Complete short-axis-to-bullseye workflow."""

from ..config import AppConfig, validate_config
from .bullseye import BullseyeOutputs, generate_bullseye
from .short_axis import generate_short_axis


def run_complete_pipeline(config: AppConfig) -> BullseyeOutputs:
    """Run both interactive workflows with validation before the first GUI."""
    validate_config(config, "pipeline")
    short_axis_outputs = generate_short_axis(config)
    return generate_bullseye(config, short_axis_outputs)
