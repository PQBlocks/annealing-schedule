"""
Optimized implementation - Numba JIT + CSR sparse matrix.
"""
import math
import time
import os
import warnings
from typing import List, Tuple, Optional
import numpy as np
from numba import njit

# Use absolute import matching test path (sys.path has src/), then re-export
# so isinstance checks work regardless of import path
from pqblocks_anneal_schedule import annealing_types as _annealing_types
from pqblocks_anneal_schedule.annealing_types import AnnealingResult as _AR
from pqblocks_anneal_schedule.reference.stats import validate_inputs

# Re-export AnnealingResult so isinstance works when imported via 'import optimized'
AnnealingResult = _AR


@njit(fastmath=True, nogil=True)
def _compute_instance_stats_opt(J_data: np.ndarray, J_indices: np.ndarray, J_indptr: np.ndarray, h: np.ndarray) -> Tuple[float, float, float]:
    n = len(h)
    if n == 0:
        return 0.0, 0.0, 0.0

    num_edges = 0
    sum_w = 0.0
    sum_w2 = 0.0
    frustr = 0.0
    deg = np.zeros(n, dtype=np.int32)

    # Iterate through CSR format, counting each edge exactly once (i < j)
    # For symmetric matrices, CSR stores both (i,j) and (j,i) in different rows
    for i in range(n):
        for ptr in range(J_indptr[i], J_indptr[i+1]):
            j = J_indices[ptr]
            w = J_data[ptr]
            # Only process each edge once (when i < j)
            if i < j and w != 0:
                num_edges += 1
                sum_w += w
                sum_w2 += w * w

                sign_w = 1.0 if w >= 0 else -1.0
                prod_h = h[i] * h[j]
                sign_h = 1.0 if prod_h >= 0 else -1.0
                frustr += 0.5 * (1.0 - sign_w * sign_h)
                
                # Count degrees correctly: each edge contributes 1 to both i and j
                deg[i] += 1
                deg[j] += 1

    if num_edges == 0:
        return 1e-10, 0.0, 0.0

    mean_w = sum_w / num_edges
    mean_w2 = sum_w2 / num_edges
    var_w = max(0.0, mean_w2 - mean_w * mean_w)
    sigma_c = max(1e-10, math.sqrt(var_w))

    f_r = frustr / num_edges

    # Compute degree entropy from degree array
    counts = {}
    for i in range(n):
        d = int(deg[i])
        if d in counts:
            counts[d] += 1
        else:
            counts[d] = 1

    H_d = 0.0
    for d, cnt in counts.items():
        if cnt > 0:
            p = cnt / n
            H_d -= p * math.log2(p)

    return float(sigma_c), float(H_d), float(f_r)


def compute_instance_stats_opt(J: np.ndarray, h: np.ndarray) -> Tuple[float, float, float]:
    validate_inputs(J, h)
    from scipy.sparse import csr_matrix
    J_csr = csr_matrix(J)
    return _compute_instance_stats_opt(J_csr.data, J_csr.indices, J_csr.indptr, h)


def generate_schedule_opt(
    stats: Tuple[float, float, float],
    n: int,
    max_degree: int,
    budget_steps: int = 128,
    budget_wallclock_ms: float = 5000.0,
) -> List[Tuple[int, float, float]]:
    from pqblocks_anneal_schedule.reference import generate_schedule_ref
    return generate_schedule_ref(stats, n, max_degree, budget_steps, budget_wallclock_ms)


@njit(fastmath=True, nogil=True)
def _evaluate_energy_opt(s: np.ndarray, J_data: np.ndarray, J_indices: np.ndarray, J_indptr: np.ndarray, h: np.ndarray) -> float:
    n = len(s)
    if n == 0:
        return 0.0

    E = 0.0
    for i in range(n):
        E += h[i] * s[i]
        for ptr in range(J_indptr[i], J_indptr[i+1]):
            j = J_indices[ptr]
            if i < j:
                E += J_data[ptr] * s[i] * s[j]
    return float(E)


@njit(fastmath=True, nogil=True)
def _sa_kernel(
    n: int,
    J_data: np.ndarray,
    J_indices: np.ndarray,
    J_indptr: np.ndarray,
    h: np.ndarray,
    schedule_T: np.ndarray,
    seed: int,
    num_runs: int,
    target_energy: float
) -> Tuple[float, np.ndarray, int, int]:
    """
    Optimized SA kernel that exactly matches the reference implementation.
    Uses the same SplitMix64 PRNG and Metropolis acceptance logic.
    """
    best_overall_energy = np.inf
    best_overall_config = np.ones(n, dtype=np.int8)
    successes = 0

    num_steps = len(schedule_T)

    for r in range(num_runs):
        # Match reference: SplitMix64 seeded with base_seed + run_idx
        rng_state = (seed + r) & 0xFFFFFFFFFFFFFFFF

        # Initialize random spin configuration: s[i] = 1 if next_float() < 0.5 else -1
        s = np.zeros(n, dtype=np.int8)
        for i in range(n):
            # next_u64()
            rng_state = (rng_state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
            z = rng_state
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
            z = (z ^ (z >> 27)) * 0x94D049BB133111EB
            z = z ^ (z >> 31)
            rng_state = z & 0xFFFFFFFFFFFFFFFF
            # next_float() = u64 / (1 << 64)
            # next_float() < 0.5  <=>  u64 < (1 << 63)  <=>  MSBit == 0
            s[i] = -1 if (z >> 63) else 1

        current_energy = _evaluate_energy_opt(s, J_data, J_indices, J_indptr, h)

        for step_idx in range(min(num_steps, len(schedule_T))):
            T = schedule_T[step_idx]

            # Single spin flip Metropolis (exactly like reference)
            for _ in range(n):
                # i = rng.next_int(n)
                rng_state = (rng_state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
                z = rng_state
                z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
                z = (z ^ (z >> 27)) * 0x94D049BB133111EB
                z = z ^ (z >> 31)
                i = (z & 0xFFFFFFFFFFFFFFFF) % n

                si_old = s[i]
                si_new = -si_old

                # Compute energy difference (exactly like reference)
                delta_E = 0.0
                for j in range(n):
                    if i != j:
                        # Check if J[i,j] != 0 by looking in CSR format
                        for ptr in range(J_indptr[i], J_indptr[i+1]):
                            if J_indices[ptr] == j:
                                delta_E += 2.0 * float(J_data[ptr]) * float(si_new) * float(s[j])
                                break
                delta_E += 2.0 * float(h[i]) * float(si_new)

                # Metropolis acceptance with overflow protection (exactly like reference)
                if delta_E <= 0:
                    accept = True
                else:
                    beta_delta = delta_E / T
                    # Clamp to prevent overflow (exp(-700) ~ 1e-304, exp(-710) underflows)
                    if beta_delta > 700:
                        accept = False
                    else:
                        # accept = rng.next_float() < math.exp(-beta_delta)
                        rng_state = (rng_state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
                        z = rng_state
                        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
                        z = (z ^ (z >> 27)) * 0x94D049BB133111EB
                        z = z ^ (z >> 31)
                        rng_state = z & 0xFFFFFFFFFFFFFFFF
                        # next_float() = u64 / (1 << 64) — must match reference exactly
                        random_val = z / 18446744073709551616.0
                        accept = random_val < math.exp(-beta_delta)

                if accept:
                    s[i] = si_new
                    current_energy += delta_E

        if current_energy < best_overall_energy:
            best_overall_energy = current_energy
            best_overall_config = s.copy()

        if current_energy <= target_energy:
            successes += 1

    return float(best_overall_energy), best_overall_config, int(rng_state), int(successes)


def simulated_annealing_opt(
    J: np.ndarray,
    h: np.ndarray,
    schedule: List[Tuple[int, float, float]],
    seed: Optional[int] = None,
    num_steps: int = 128,
    num_runs: int = 1,
) -> "_AR":
    import platform
    from scipy.sparse import csr_matrix
    validate_inputs(J, h)

    if seed is None:
        seed = int.from_bytes(os.urandom(8), "little") & 0x7FFFFFFFFFFFFFFF

    n = len(h)
    stats = compute_instance_stats_opt(J, h)

    t0_wall = time.perf_counter()

    if n == 0:
        best_overall_energy = 0.0
        best_overall_config = np.array([], dtype=np.int8)
        successes = num_runs
    else:
        J_csr = csr_matrix(J)
        schedule_T = np.array([T for _, T, _ in schedule], dtype=np.float64)

        best_overall_energy, best_overall_config, _, successes = _sa_kernel(
            n, J_csr.data, J_csr.indices, J_csr.indptr, h, schedule_T, seed, num_runs, -float("inf")
        )

    t1_wall = time.perf_counter()
    wallclock_ms = (t1_wall - t0_wall) * 1000.0

    success_prob = successes / max(1, num_runs)
    if success_prob == 0.0:
        tts = float('inf')
        success = False
    elif success_prob == 1.0:
        tts = (wallclock_ms / 1000.0) / max(1, num_runs)
        success = True
    else:
        tts = (wallclock_ms / 1000.0) * math.log(1 - 0.99) / math.log(1 - success_prob)
        success = True

    return _AR(
        best_energy=float(best_overall_energy),
        best_config=best_overall_config,
        tts=tts,
        success=success,
        success_prob=success_prob,
        num_runs=num_runs,
        num_steps=len(schedule),
        stats=stats,
        schedule=schedule,
        wallclock_ms=wallclock_ms,
        hardware=platform.node()
    )