"""Reference implementation - pure NumPy, zero external optimization libraries."""

from .stats import compute_instance_stats_ref, validate_inputs
from .schedule import generate_schedule_ref
from .annealing import simulated_annealing_ref, AnnealingResult

__all__ = [
    "compute_instance_stats_ref",
    "generate_schedule_ref",
    "simulated_annealing_ref",
    "AnnealingResult",
    "validate_inputs",
]
