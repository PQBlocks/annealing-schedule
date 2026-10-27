# Changelog

All notable changes to `pqblocks-anneal-schedule` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] - 2026-10-26

### Added

- **Instance Statistics Computation**:
  - `compute_instance_stats`: Automated extraction of coupling dispersion $\sigma_c$, degree distribution entropy $H_d$, and frustration density $f_r$ from Ising coupling matrix $J$ and external field $h$.
  
- **Adaptive Schedule Generation**:
  - `generate_schedule`: Dynamic synthesis of step-by-step cooling trajectories $T(t) = T_0 \exp(-t^{\text{quench}})$ and transverse field parameters $\Gamma(t) = \Gamma_0 (1 - t)$ scaled to graph topology.

- **Dual-Kernel Simulated Annealing Engine**:
  - High-performance Numba-accelerated inner annealing loop.
  - Exact pure-Python reference implementation ensuring cross-platform fallback.
  - Bit-identical seed determinism and reproducible trajectory generation.

- **MessagePack Serialization**:
  - `schedule_to_msgpack` and `schedule_from_msgpack` for compact binary serialization and offline caching of generated schedules.

- **Verification & Testing**:
  - 85 unit and integration tests passing across Python 3.11, 3.12, 3.13, and 3.14.
  - Verification on planted golden instances against independent CP-SAT oracle ground truth.
  - Complete documentation suite including benchmarks, security notes, use cases, agent guide, and verification evidence.
