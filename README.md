# pqblocks-anneal-schedule

> Quantum-inspired instance-adaptive simulated annealing with statistics-driven schedule generation for Ising and QUBO optimization problems.

---

> [!NOTE]
> **Domain Classification**: Quantum-inspired classical optimisation: not cryptography, no security claim.

---

## What It Is

`pqblocks-anneal-schedule` implements an instance-adaptive simulated annealing optimizer that dynamically constructs annealing schedules based on structural graph statistics extracted from the problem instance.

Traditional simulated annealing typically relies on static geometric or linear cooling schedules (e.g., $T_{k+1} = \alpha T_k$) that do not account for instance-specific topology, coupling variance, or frustration. `pqblocks-anneal-schedule` analyzes the problem's coupling matrix $J$ and local field vector $h$ to derive three fundamental instance characteristics:
1. **Coupling Dispersion ($\sigma_c$)**: Standard deviation of non-zero coupling weights, governing initial temperature $T_0$.
2. **Degree Distribution Entropy ($H_d$)**: Shannon entropy of vertex degrees, determining the quench exponent.
3. **Frustration Density ($f_r$)**: Proportion of frustrated edges, informing transverse field $\Gamma_0$ scaling.

Using these statistics, the package generates adaptive temperature trajectories $T(t) = T_0 \exp(-t^{\text{quench}})$ and transverse field schedules $\Gamma(t) = \Gamma_0 (1 - t)$ optimized for structured sparse instances.

---

## When It Wins / When It Loses

- **Wins on structured sparse instances**: Excels on heterogeneous, sparse graphs (such as MaxCut and planted partition problems) where coupling variance and degree irregularity allow the adaptive schedule to navigate energy barriers effectively, achieving up to 2.04x faster Time-To-Solution (TTS) compared to naive geometric schedules.
- **Loses on dense, uniform instances**: On large, dense, uniform random graphs (e.g., $n=512, p=0.5$), the overhead of graph statistics extraction and schedule generation makes it slower than highly optimized C++ solvers (such as `neal`, which is up to 49.6x faster on $n=512$ dense MaxCut).

---

## Key Features

- **Automated Statistics Extraction**: Computes $(\sigma_c, H_d, f_r)$ directly from problem matrices in $O(|E|)$ time.
- **Adaptive Schedule Synthesis**: Generates step-by-step temperature and transverse field schedules tailored to problem structure.
- **Dual-Kernel Architecture**: High-speed JIT-accelerated kernel via Numba with a bit-identical pure-Python reference implementation.
- **Schedule Serialization**: Fast binary serialization and deserialization via MessagePack (`schedule_to_msgpack`, `schedule_from_msgpack`).
- **Deterministic Reproducibility**: Exact seed control ensuring bit-identical trajectory replication across runs.

---

## Installation

```bash
pip install pqblocks-anneal-schedule
```

---

## Quickstart

```python
import numpy as np
from pqblocks_anneal_schedule import (
    compute_instance_stats,
    generate_schedule,
    simulated_annealing,
)

# Define a 4-spin Ising ring
J = np.array([
    [0.0, -1.0,  0.0,  1.0],
    [-1.0,  0.0, -1.0,  0.0],
    [0.0, -1.0,  0.0, -1.0],
    [1.0,  0.0, -1.0,  0.0]
], dtype=float)
h = np.zeros(4, dtype=float)

# Extract graph statistics and synthesize adaptive schedule
stats = compute_instance_stats(J, h)
schedule = generate_schedule(stats, n=4, max_degree=2, budget_steps=64)

# Execute simulated annealing
result = simulated_annealing(J, h, schedule, seed=42, num_steps=64, num_runs=1)

print(f"Best Energy: {result.best_energy:.6f}")
print("Best Config:", result.best_config)
```

Execution output:

```text
Best Energy: -2.000000
Best Config: [-1 -1 -1 -1]
```

---

## Documentation

- [BENCHMARKS.md](BENCHMARKS.md): Empirical runtime measurements and comparative MaxCut evaluation against reference baselines.
- [TESTING.md](TESTING.md): Test suite architecture, coverage metrics, and validation procedures.
- [EVIDENCE.md](EVIDENCE.md): Execution logs and reproducibility artifacts.
- [VERIFICATION_REPORT.md](VERIFICATION_REPORT.md): Correctness, determinism, and serialization checks.
- [USE_CASES.md](USE_CASES.md): Practical applications and schedule caching patterns.
- [SECURITY_NOTES.md](SECURITY_NOTES.md): Non-cryptographic scope disclosure and numerical bounds.
- [AGENTS.md](AGENTS.md): Machine-readable interfaces and automated pipeline integration.
- [CHANGELOG.md](CHANGELOG.md): Version history and updates.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
