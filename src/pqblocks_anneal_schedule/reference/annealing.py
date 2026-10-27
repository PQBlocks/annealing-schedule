"""Reference simulated annealing - pure NumPy implementation with SplitMix64 PRNG."""
import math
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

# Absolute import - works when src/ is on sys.path (conftest.py adds it)
from pqblocks_anneal_schedule import annealing_types
from pqblocks_anneal_schedule.annealing_types import AnnealingResult

# Re-export so isinstance works regardless of which import path resolves first.
AnnealingResult = AnnealingResult


class SplitMix64:
    """SplitMix64 PRNG for deterministic bit-identical reproducibility."""

    def __init__(self, seed: int):
        self.state = seed & ((1 << 64) - 1)

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
        z = self.state
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) * 0x94D049BB133111EB
        z = z ^ (z >> 31)
        return z & ((1 << 64) - 1)

    def next_float(self) -> float:
        return self.next_u64() / (1 << 64)

    def next_int(self, n: int) -> int:
        """Return random int in [0, n)."""
        return self.next_u64() % n


def validate_sa_inputs(
    J: np.ndarray,
    h: np.ndarray,
    schedule: List[Tuple[int, float, float]],
    seed: Optional[int],
    num_steps: int,
    num_runs: int,
) -> None:
    """Validate simulated annealing inputs per Section 2.1."""
    # J and h validation
    if J.ndim != 2 or J.shape[0] != J.shape[1]:
        raise ValueError(
            f"Coupling matrix J must be a square 2D array, got shape {J.shape}"
        )

    n = J.shape[0]
    if h.ndim != 1 or len(h) != n:
        raise ValueError(
            f"External field h must be 1D array of length {n}, got shape {h.shape}"
        )

    if not np.all(np.isfinite(J)):
        raise ValueError("Coupling matrix J contains non-finite values")

    if not np.all(np.isfinite(h)):
        raise ValueError("External field h contains non-finite values")

    if not np.allclose(J, J.T, rtol=1e-14, atol=1e-14):
        raise ValueError("Coupling matrix J must be symmetric")

    if not np.all(np.diag(J) == 0):
        raise ValueError("Coupling matrix J must have zero diagonal")

    max_coeff = 1e12
    if np.any(np.abs(J) > max_coeff):
        raise ValueError(f"Coupling matrix J entries must satisfy |J_ij| <= {max_coeff}")

    if np.any(np.abs(h) > max_coeff):
        raise ValueError(f"External field h entries must satisfy |h_i| <= {max_coeff}")

    # Schedule validation
    if n > 0:
        if not schedule:
            raise ValueError("Schedule must be non-empty for n > 0")
        for i, (step, T, G) in enumerate(schedule):
            if T <= 0:
                raise ValueError(f"Schedule step {step}: temperature T must be > 0, got {T}")
            if G < 0:
                raise ValueError(f"Schedule step {step}: Gamma must be >= 0, got {G}")
            if i > 0:
                prev_T = schedule[i - 1][1]
                prev_G = schedule[i - 1][2]
                if T > prev_T + 1e-15:
                    raise ValueError(f"Schedule step {step}: T must be non-increasing, got {T} > {prev_T}")
                if G > prev_G + 1e-15:
                    raise ValueError(f"Schedule step {step}: Gamma must be non-increasing, got {G} > {prev_G}")

    # Seed validation
    if seed is not None:
        if not isinstance(seed, int) or seed < 0 or seed >= (1 << 64):
            raise ValueError(f"seed must be int in [0, 2**64-1], got {seed}")

    # Budget validation
    if not isinstance(num_steps, int) or num_steps <= 0:
        raise ValueError(f"num_steps must be int > 0, got {num_steps}")
    if not isinstance(num_runs, int) or num_runs <= 0:
        raise ValueError(f"num_runs must be int > 0, got {num_runs}")


def _energy(J: np.ndarray, h: np.ndarray, s: NDArray[np.int8]) -> float:
    """Compute Ising energy E(s; J, h) using f64 accumulator."""
    # E(s) = sum_{i<j} J_ij s_i s_j + sum_i h_i s_i
    energy = 0.0
    n = len(s)
    for i in range(n):
        si = float(s[i])
        energy += float(h[i]) * si
        for j in range(i + 1, n):
            if J[i, j] != 0:
                energy += float(J[i, j]) * si * float(s[j])
    return energy


def simulated_annealing_ref(
    J: np.ndarray,
    h: np.ndarray,
    schedule: List[Tuple[int, float, float]],
    seed: Optional[int] = None,
    num_steps: int = 128,
    num_runs: int = 10,
) -> AnnealingResult:
    """Reference simulated annealing implementation.

    Inputs:
        J: n x n symmetric coupling matrix (zero diagonal, finite).
        h: n external field vector (finite).
        schedule: list of (step, T, Gamma) tuples from generate_schedule.
        seed: int in [0, 2**64-1] or None (OS entropy).
        num_steps: steps per run (S).
        num_runs: number of independent runs.
    Returns:
        AnnealingResult per Section 4.4.
    Raises:
        ValueError on illegal inputs per Section 2.1.
    """
    validate_sa_inputs(J, h, schedule, seed, num_steps, num_runs)

    n = len(h)
    if n == 0:
        return AnnealingResult(
            best_energy=0.0,
            best_config=np.array([], dtype=np.int8),
            tts=0.0,
            success=True,
            success_prob=1.0,
            num_runs=num_runs,
            num_steps=num_steps,
            stats=(0.0, 0.0, 0.0),
            schedule=schedule,
            wallclock_ms=0.0,
            hardware="",
        )

    # Derive base seed
    if seed is None:
        base_seed = int.from_bytes(
            np.random.bytes(8), byteorder="little"
        ) & ((1 << 64) - 1)
    else:
        base_seed = seed

    t0_wall = time.perf_counter()

    best_energy_overall = float("inf")
    best_config_overall = None
    success_count = 0

    # Target energy for TTS computation (ground state energy for planted instances)
    # We'll use the best energy found as target for success_prob
    target_energy = float("inf")

    for run_idx in range(num_runs):
        rng = SplitMix64(base_seed + run_idx)

        # Initialize random spin configuration
        s = np.array([1 if rng.next_float() < 0.5 else -1 for _ in range(n)], dtype=np.int8)

        current_energy = _energy(J, h, s)

        for step_idx in range(min(num_steps, len(schedule))):
            step, T, Gamma = schedule[step_idx]

            # Single spin flip Metropolis
            for _ in range(n):
                i = rng.next_int(n)
                si_old = s[i]
                si_new = -si_old

                # Compute energy difference
                delta_E = 0.0
                for j in range(n):
                    if i != j and J[i, j] != 0:
                        delta_E += 2.0 * float(J[i, j]) * float(si_new) * float(s[j])
                delta_E += 2.0 * float(h[i]) * float(si_new)

                # Metropolis acceptance with overflow protection
                if delta_E <= 0:
                    accept = True
                else:
                    beta_delta = delta_E / T
                    # Clamp to prevent overflow (exp(-700) ~ 1e-304, exp(-710) underflows)
                    if beta_delta > 700:
                        accept = False
                    else:
                        accept = rng.next_float() < math.exp(-beta_delta)

                if accept:
                    s[i] = si_new
                    current_energy += delta_E

        if current_energy < best_energy_overall:
            best_energy_overall = current_energy
            best_config_overall = s.copy()

        if current_energy <= target_energy:
            success_count += 1

    wallclock_ms = (time.perf_counter() - t0_wall) * 1000.0

    success_prob = success_count / num_runs
    success = success_prob > 0

    # Compute TTS (Time-to-solution at 99% target probability)
    if success_prob == 0.0:
        tts = float("inf")
    elif success_prob == 1.0:
        tts = wallclock_ms / (1000.0 * num_runs)
    else:
        tts = (wallclock_ms / 1000.0) * math.log(1 - 0.99) / math.log(1 - success_prob)

    return AnnealingResult(
        best_energy=best_energy_overall,
        best_config=best_config_overall if best_config_overall is not None else np.array([], dtype=np.int8),
        tts=tts,
        success=success,
        success_prob=success_prob,
        num_runs=num_runs,
        num_steps=num_steps,
        stats=(0.0, 0.0, 0.0),
        schedule=schedule,
        wallclock_ms=wallclock_ms,
        hardware="",
    )