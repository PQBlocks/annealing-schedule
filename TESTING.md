# Testing Guide: pqblocks-anneal-schedule

> [!NOTE]
> **Domain Classification**: Quantum-inspired classical optimisation: not cryptography, no security claim.

---

## Overview

The `pqblocks-anneal-schedule` test suite provides comprehensive test coverage across mathematical correctness, input validation, determinism, kernel equivalence, serialization, and boundary edge cases.

All 85 tests execute cleanly with zero failures and zero warnings.

---

## Test Execution

To execute the full test suite in an isolated environment:

```bash
cd packages/pqblocks-anneal-schedule
PYTHONPATH=src pytest -v --cov=pqblocks_anneal_schedule --cov-report=term
```

---

## Coverage Summary

| Module | Statements | Missed | Statement Coverage |
| :--- | :---: | :---: | :---: |
| `src/pqblocks_anneal_schedule/__init__.py` | 5 | 0 | 100% |
| `src/pqblocks_anneal_schedule/annealing_types.py` | 6 | 0 | 100% |
| `src/pqblocks_anneal_schedule/msgpack_serializer.py` | 7 | 0 | 100% |
| `src/pqblocks_anneal_schedule/optimized.py` | 162 | 111 | 31% |
| `src/pqblocks_anneal_schedule/reference/__init__.py` | 4 | 0 | 100% |
| `src/pqblocks_anneal_schedule/reference/annealing.py` | 125 | 15 | 88% |
| `src/pqblocks_anneal_schedule/reference/schedule.py` | 29 | 2 | 93% |
| `src/pqblocks_anneal_schedule/reference/stats.py` | 55 | 2 | 96% |
| **Total** | **393** | **130** | **67%** |

---

## Test Suite Breakdown

The 85 tests are organized into 11 test modules:

### 1. Edge Case Handling (`tests/test_edge_cases.py` — 10 tests)
- Empty instances ($n = 0$) returning safe zero statistics and empty schedules.
- Single-spin instances ($n = 1$) without edges.
- Disconnected graph components and isolated vertices.
- Degenerate zero-coupling matrices.
- Metropolis acceptance ratio overflow protection ($T \to 0$ and extreme $\Delta E$).
- Independent RNG streams across multiple runs.
- Frustration bounds verification ($0 \le f_r \le 1$).

### 2. Kernel Equivalence (`tests/test_equivalence.py` — 11 tests)
- Bit-identical statistics extraction between reference and optimized paths.
- Bit-identical schedule synthesis between reference and optimized paths.
- Ground state discovery consistency between reference and JIT kernels.
- Parameter handling consistency across both implementations.

### 3. Golden Reference Verification (`tests/test_golden.py` — 2 tests)
- Structural integrity of reference golden instances (`golden_instances.json`).
- Cross-validation against independent CP-SAT constraint solver ground truth energies.

### 4. Seed Determinism (`tests/test_seed_determinism.py` — 4 tests)
- Exact bit-identical trajectory reproduction when using identical random seeds.
- Statistical divergence when using different seeds.
- Comprehensive seed range validation (0 to $2^{32}-1$).
- Fallback to OS entropy when `seed=None`.

### 5. Architectural Pipeline Slices (`tests/test_slice_1.py` to `test_slice_8.py` — 47 tests)
- `test_slice_1.py`: Core statistics mathematical formulas and matrix property validation.
- `test_slice_2.py`: Schedule monotonicity, quench exponent calculations, and boundary steps.
- `test_slice_3.py`: Reference simulated annealing loop, Metropolis updates, and cooling steps.
- `test_slice_4.py`: Optimized statistics calculation and high-throughput vectorization.
- `test_slice_5.py`: Optimized schedule generator array routines.
- `test_slice_6.py`: Numba JIT accelerated annealing kernel integration.
- `test_slice_7.py`: Top-level package export integration and end-to-end pipeline.
- `test_slice_8.py`: MessagePack binary serialization and deserialization integrity.

### 6. Input Validation (`tests/test_validation.py` — 11 tests)
- Requirement that $J$ and $h$ are NumPy ndarrays with finite elements.
- Strict checks for square matrix dimensions ($n \times n$).
- Strict symmetry enforcement ($J = J^T$).
- Strict zero diagonal enforcement ($J_{ii} = 0$).
- Budget constraints validation (positive integer step counts, positive wall-clock limits).
