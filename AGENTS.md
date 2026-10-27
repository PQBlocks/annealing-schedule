# Agent Integration Guide: pqblocks-anneal-schedule

> [!NOTE]
> **Domain Classification**: Quantum-inspired classical optimisation: not cryptography, no security claim.

---

## Overview for Autonomous Agents

This document specifies API contracts, data types, error behaviors, and invocation patterns for autonomous coding agents integrating `pqblocks-anneal-schedule`.

---

## Core Module Exports

```python
from pqblocks_anneal_schedule import (
    compute_instance_stats,
    generate_schedule,
    simulated_annealing,
    schedule_to_msgpack,
    schedule_from_msgpack,
    AnnealingResult,
)
```

---

## Function Signatures & Behaviors

### 1. `compute_instance_stats(J: np.ndarray, h: np.ndarray) -> Tuple[float, float, float]`

Computes structural graph metrics from the Ising Hamiltonian parameters.

- **Parameters**:
  - `J`: Symmetric 2D NumPy array of shape `(n, n)` with dtype `float64`. Diagonal must be zero.
  - `h`: 1D NumPy array of shape `(n,)` with dtype `float64`.
- **Returns**:
  - `(sigma_c, H_d, f_r)`:
    - `sigma_c` (`float`): Coupling weight standard deviation.
    - `H_d` (`float`): Degree distribution Shannon entropy in bits.
    - `f_r` (`float`): Frustration density in range $[0.0, 1.0]$.
- **Raises**:
  - `ValueError`: If $J$ is not square, not symmetric, has non-zero diagonal, contains `NaN`/`Inf`, or if lengths of $J$ and $h$ mismatch.

---

### 2. `generate_schedule(stats, n, max_degree, budget_steps=128, budget_wallclock_ms=5000.0) -> List[Tuple[int, float, float]]`

Synthesizes an adaptive annealing schedule.

- **Parameters**:
  - `stats`: `(sigma_c, H_d, f_r)` tuple from `compute_instance_stats`.
  - `n` (`int`): Number of spins ($n \ge 0$).
  - `max_degree` (`int`): Maximum node degree in the coupling graph ($\Delta \ge 0$).
  - `budget_steps` (`int`): Number of annealing steps ($S > 0$, default 128).
  - `budget_wallclock_ms` (`float`): Target wall-clock budget in milliseconds (default 5000.0).
- **Returns**:
  - List of `(step, T, Gamma)` tuples, where `step` is `0 <= step < budget_steps`, `T` is temperature ($T > 0$), and `Gamma` is transverse field intensity.

---

### 3. `simulated_annealing(J, h, schedule, seed=None, num_steps=None, num_runs=1, target_energy=-inf) -> AnnealingResult`

Executes simulated annealing across one or more independent runs.

- **Parameters**:
  - `J`: Coupling matrix `(n, n)`.
  - `h`: External field vector `(n,)`.
  - `schedule`: Schedule tuple list from `generate_schedule` or `schedule_from_msgpack`.
  - `seed` (`int`, optional): Random seed for reproducible execution. If `None`, seeds from OS entropy.
  - `num_steps` (`int`, optional): Number of steps to execute. Defaults to `len(schedule)`.
  - `num_runs` (`int`): Number of independent annealing runs (default 1).
  - `target_energy` (`float`): Energy threshold for determining success (default $-\infty$).
- **Returns**:
  - `AnnealingResult` dataclass instance.

---

## Result Data Structure

```python
@dataclass
class AnnealingResult:
    best_energy: float                       # Minimum energy found across all runs
    best_config: NDArray[np.int8]            # Optimal spin configuration in {-1, +1}^n
    tts: float                               # Time-to-solution TTS(0.99) in seconds
    success: bool                            # True iff best_energy <= target_energy
    success_prob: float                      # Empirical success probability
    num_runs: int                            # Total runs evaluated
    num_steps: int                           # Steps per run
    stats: Tuple[float, float, float]        # (sigma_c, H_d, f_r) instance metrics
    schedule: List[Tuple[int, float, float]] # Schedule sequence consumed
    wallclock_ms: float                      # Wall-clock duration in milliseconds
    hardware: str                            # Host CPU and environment fingerprint
```

---

## Agent Guidelines & Traps

1. **Matrix Symmetry Trap**: Always symmetrize coupling matrices before passing to `compute_instance_stats`:
   ```python
   J = 0.5 * (J + J.T)
   np.fill_diagonal(J, 0.0)
   ```
2. **Dimension Mismatch**: Never pass 1D vectors for $J$. $J$ must have `ndim == 2`.
3. **Empty Instance Safety**: For $n=0$, `compute_instance_stats` returns `(0.0, 0.0, 0.0)` and `generate_schedule` returns an empty list `[]`.
