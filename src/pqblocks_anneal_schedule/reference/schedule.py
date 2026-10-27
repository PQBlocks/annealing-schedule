"""Reference schedule generator - pure NumPy implementation."""
import math
import time
import warnings
from typing import List, Tuple

import numpy as np


def generate_schedule_ref(
    stats: Tuple[float, float, float],
    n: int,
    max_degree: int,
    budget_steps: int = 128,
    budget_wallclock_ms: float = 5000.0,
) -> List[Tuple[int, float, float]]:
    """Generate adaptive annealing schedule [(step, T(step), Gamma(step))].

    Inputs:
        stats: (sigma_c, H_d, f_r) from compute_instance_stats.
        n: instance size len(h).
        max_degree: maximum graph degree max_i(d_i).
        budget_steps: total annealing steps S > 0.
        budget_wallclock_ms: total wall-clock budget W > 0.
    Returns:
        List of (step, T, Gamma) tuples.
    Raises:
        ValueError on non-finite, non-positive budgets or negative dimensions.
    """
    if not (isinstance(budget_steps, int) and budget_steps > 0):
        raise ValueError(f"budget_steps must be int > 0, got {budget_steps}")
    if not (np.isfinite(budget_wallclock_ms) and budget_wallclock_ms > 0):
        raise ValueError(f"budget_wallclock_ms must be finite > 0, got {budget_wallclock_ms}")
    if n < 0 or max_degree < 0:
        raise ValueError(f"n and max_degree must be non-negative, got n={n}, max_degree={max_degree}")

    if n == 0:
        return []

    t0_wall = time.perf_counter()
    sigma_c, H_d, f_r = stats

    T0 = max(1e-10, 0.5 * sigma_c * math.sqrt(max(n, 1)))
    Gamma0 = max(1e-10, 1.0 * sigma_c * math.sqrt(max(n, 1)))
    quench = 0.8 + 0.2 * H_d / math.log2(max(max_degree, 1) + 1)

    sched = []
    for step in range(budget_steps):
        t = step / budget_steps
        T = T0 * math.exp(-(t ** quench))
        G = Gamma0 * (1.0 - t)
        sched.append((step, float(T), float(G)))

    elapsed_ms = (time.perf_counter() - t0_wall) * 1000.0
    if elapsed_ms > 0.1 * budget_wallclock_ms:
        warnings.warn(
            f"schedule construction {elapsed_ms:.1f}ms exceeds 10% of budget_wallclock_ms={budget_wallclock_ms}",
            RuntimeWarning,
        )
    return sched