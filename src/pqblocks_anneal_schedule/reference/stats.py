"""Reference statistics engine - pure NumPy implementation."""
import math
from typing import Tuple

import numpy as np


def validate_inputs(J: np.ndarray, h: np.ndarray) -> None:
    """Validate coupling matrix J and external field h per Section 2.1."""
    # Type checks
    if not isinstance(J, np.ndarray):
        raise ValueError(
            f"Coupling matrix J must be a numpy.ndarray, got {type(J).__name__}"
        )
    if not isinstance(h, np.ndarray):
        raise ValueError(
            f"External field h must be a numpy.ndarray, got {type(h).__name__}"
        )
    # J must be 2-D square array
    if J.ndim != 2 or J.shape[0] != J.shape[1]:
        raise ValueError(
            f"Coupling matrix J must be a square 2D array, got shape {J.shape}"
        )

    n = J.shape[0]

    # h must be 1-D array with matching length
    if h.ndim != 1 or len(h) != n:
        raise ValueError(
            f"External field h must be 1D array of length {n}, got shape {h.shape}"
        )

    # All entries must be finite
    if not np.all(np.isfinite(J)):
        raise ValueError("Coupling matrix J contains non-finite values")

    if not np.all(np.isfinite(h)):
        raise ValueError("External field h contains non-finite values")

    # Symmetry check
    if not np.allclose(J, J.T, rtol=1e-14, atol=1e-14):
        raise ValueError("Coupling matrix J must be symmetric")

    # Zero diagonal check
    if not np.all(np.diag(J) == 0):
        raise ValueError("Coupling matrix J must have zero diagonal")

    # Coefficient bounds
    max_coeff = 1e12
    if np.any(np.abs(J) > max_coeff):
        raise ValueError(
            f"Coupling matrix J entries must satisfy |J_ij| <= {max_coeff}"
        )

    if np.any(np.abs(h) > max_coeff):
        raise ValueError(
            f"External field h entries must satisfy |h_i| <= {max_coeff}"
        )


def compute_instance_stats_ref(J: np.ndarray, h: np.ndarray) -> Tuple[float, float, float]:
    """Compute instance statistics (sigma_c, H_d, f_r).

    Inputs:
        J: n x n symmetric float ndarray, zero diagonal, finite.
        h: n float ndarray, finite.
    Returns:
        (sigma_c, H_d, f_r) float64 triple.
    Raises:
        ValueError on illegal inputs per Section 2.1.
    """
    validate_inputs(J, h)
    n = len(h)
    if n == 0:
        return (0.0, 0.0, 0.0)

    # Nonzero edges
    edges = [(i, j, J[i, j]) for i in range(n) for j in range(i + 1, n) if J[i, j] != 0]
    if not edges:
        return (1e-10, 0.0, 0.0)

    # Coupling standard deviation sigma_c
    weights = [w for (_, _, w) in edges]
    mean_w = sum(weights) / len(weights)
    mean_w2 = sum(w * w for w in weights) / len(weights)
    var_w = max(0.0, mean_w2 - mean_w * mean_w)
    sigma_c = max(1e-10, math.sqrt(var_w))

    # Degree distribution entropy H_d
    deg = [0] * n
    for (i, j, _) in edges:
        deg[i] += 1
        deg[j] += 1
    counts = {}
    for d in deg:
        counts[d] = counts.get(d, 0) + 1
    H_d = -sum((cnt / n) * math.log2(cnt / n) for cnt in counts.values() if cnt > 0)

    # Frustration density f_r
    frustr = 0.0
    for (i, j, w) in edges:
        sign_w = 1.0 if w >= 0 else -1.0
        prod_h = h[i] * h[j]
        sign_h = 1.0 if prod_h >= 0 else -1.0
        frustr += 0.5 * (1.0 - sign_w * sign_h)
    f_r = frustr / len(edges)

    return (float(sigma_c), float(H_d), float(f_r))