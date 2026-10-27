# Verification Evidence: pqblocks-anneal-schedule

> [!NOTE]
> **Domain Classification**: Quantum-inspired classical optimisation: not cryptography, no security claim.

---

## 1. Clean-Room Test Execution

Command executed:

```bash
cd packages/pqblocks-anneal-schedule
PYTHONPATH=src pytest -v --cov=pqblocks_anneal_schedule --cov-report=term
```

Captured output:

```text
============================= test session starts ==============================
platform linux -- Python 3.14.8, pytest-9.1.1, pluggy-1.6.0 -- /usr/bin/python
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: packages/pqblocks-anneal-schedule
configfile: pyproject.toml
testpaths: tests
plugins: cov-7.1.0, hypothesis-6.168.0, platformdirs-4.13.0
collected 85 items

tests/test_edge_cases.py::test_empty_instance PASSED                     [  1%]
tests/test_edge_cases.py::test_single_spin PASSED                        [  2%]
tests/test_edge_cases.py::test_disconnected_components PASSED            [  3%]
tests/test_edge_cases.py::test_degenerate_couplings PASSED               [  4%]
tests/test_edge_cases.py::test_metropolis_overflow PASSED                [  5%]
tests/test_edge_cases.py::test_rng_stream_independence PASSED            [  7%]
tests/test_edge_cases.py::test_frustration_bounds PASSED                 [  8%]
tests/test_edge_cases.py::test_non_convergence_detection PASSED          [  9%]
tests/test_edge_cases.py::test_no_trotter_dimension PASSED               [ 10%]
tests/test_edge_cases.py::test_penalty_free_energy_bounded PASSED        [ 11%]
tests/test_equivalence.py::test_stats_bit_identity PASSED                [ 12%]
tests/test_equivalence.py::test_schedule_bit_identity PASSED             [ 14%]
tests/test_equivalence.py::test_both_kernels_produce_valid_results PASSED [ 15%]
tests/test_equivalence.py::test_4spin_ground_state_findable PASSED       [ 16%]
tests/test_equivalence.py::test_sparse_instance_both_kernels PASSED      [ 17%]
tests/test_equivalence.py::test_optimized_schedule_num_steps_param PASSED [ 18%]
tests/test_equivalence.py::test_optimized_sa_kernel_coverage PASSED      [ 20%]
tests/test_equivalence.py::test_optimized_edge_case_empty PASSED         [ 21%]
tests/test_equivalence.py::test_optimized_single_spin PASSED             [ 22%]
tests/test_equivalence.py::test_optimized_seed_none PASSED               [ 23%]
tests/test_equivalence.py::test_optimized_partial_success_tts PASSED     [ 24%]
tests/test_golden.py::test_golden_instances_format PASSED                [ 25%]
tests/test_golden.py::test_golden_instances_cpsat_verified PASSED        [ 27%]
tests/test_seed_determinism.py::test_same_seed_identical PASSED          [ 28%]
tests/test_seed_determinism.py::test_different_seeds_different PASSED    [ 29%]
tests/test_seed_determinism.py::test_seed_range_accepted PASSED          [ 30%]
tests/test_seed_determinism.py::test_seed_none_uses_os_entropy PASSED    [ 31%]
tests/test_slice_1.py::TestComputeInstanceStats::test_4spin_worked_example PASSED [ 32%]
tests/test_slice_1.py::TestComputeInstanceStats::test_empty_instance PASSED [ 34%]
tests/test_slice_1.py::TestComputeInstanceStats::test_degenerate_couplings PASSED [ 35%]
tests/test_slice_1.py::TestComputeInstanceStats::test_validation_J_not_square PASSED [ 36%]
tests/test_slice_1.py::TestComputeInstanceStats::test_validation_J_not_symmetric PASSED [ 37%]
tests/test_slice_1.py::TestComputeInstanceStats::test_validation_J_has_diagonal PASSED [ 38%]
tests/test_slice_1.py::TestGenerateSchedule::test_4spin_schedule_exact_match PASSED [ 40%]
tests/test_slice_1.py::TestGenerateSchedule::test_monotonicity PASSED    [ 41%]
tests/test_slice_1.py::TestGenerateSchedule::test_empty_instance_returns_empty_schedule PASSED [ 42%]
tests/test_slice_1.py::TestValidateInputs::test_J_must_be_ndarray PASSED [ 43%]
tests/test_slice_1.py::TestValidateInputs::test_h_must_be_ndarray PASSED [ 44%]
tests/test_slice_1.py::TestValidateInputs::test_J_finite_check PASSED    [ 45%]
tests/test_slice_1.py::TestValidateInputs::test_h_finite_check PASSED    [ 47%]
tests/test_slice_2.py::TestScheduleGenerator::test_4spin_schedule_exact_match PASSED [ 48%]
tests/test_slice_2.py::TestScheduleGenerator::test_monotonicity PASSED   [ 49%]
tests/test_slice_2.py::TestScheduleGenerator::test_empty_instance_returns_empty_schedule PASSED [ 50%]
tests/test_slice_2.py::TestScheduleGenerator::test_single_spin_instance PASSED [ 51%]
tests/test_slice_3.py::TestSimulatedAnnealingRef::test_4spin_worked_example PASSED [ 52%]
tests/test_slice_3.py::TestSimulatedAnnealingRef::test_empty_instance PASSED [ 54%]
tests/test_slice_3.py::TestSimulatedAnnealingRef::test_single_spin PASSED [ 55%]
tests/test_slice_3.py::TestSimulatedAnnealingRef::test_metropolis_overflow_protection PASSED [ 56%]
tests/test_slice_3.py::TestSimulatedAnnealingRef::test_rng_stream_independence PASSED [ 57%]
tests/test_slice_3.py::TestSimulatedAnnealingRef::test_same_seed_determinism PASSED [ 58%]
tests/test_slice_3.py::TestSimulatedAnnealingRef::test_validation_empty_schedule_n_gt_0 PASSED [ 60%]
tests/test_slice_3.py::TestSimulatedAnnealingRef::test_validation_negative_T PASSED [ 61%]
tests/test_slice_3.py::TestSimulatedAnnealingRef::test_validation_seed_range PASSED [ 62%]
tests/test_slice_4.py::TestOptimizedStats::test_4spin_bit_identity_with_ref PASSED [ 63%]
tests/test_slice_4.py::TestOptimizedStats::test_empty_instance PASSED    [ 64%]
tests/test_slice_4.py::TestOptimizedStats::test_degenerate_couplings PASSED [ 65%]
tests/test_slice_4.py::TestOptimizedStats::test_large_instance PASSED    [ 67%]
tests/test_slice_5.py::TestOptimizedSchedule::test_4spin_bit_identity_with_ref PASSED [ 68%]
tests/test_slice_5.py::TestOptimizedSchedule::test_empty_instance PASSED [ 69%]
tests/test_slice_5.py::TestOptimizedSchedule::test_monotonicity PASSED   [ 70%]
tests/test_slice_5.py::TestOptimizedSchedule::test_large_instance PASSED [ 71%]
tests/test_slice_6.py::TestOptimizedSA::test_4spin_equivalence PASSED    [ 72%]
tests/test_slice_6.py::TestOptimizedSA::test_same_seed_determinism_opt PASSED [ 74%]
tests/test_slice_6.py::TestOptimizedSA::test_empty_instance_opt PASSED   [ 75%]
tests/test_slice_6.py::TestOptimizedSA::test_single_spin_opt PASSED      [ 76%]
tests/test_slice_6.py::TestOptimizedSA::test_numba_available PASSED      [ 77%]
tests/test_slice_7.py::TestTopLevelIntegration::test_public_api_exists PASSED [ 78%]
tests/test_slice_7.py::TestTopLevelIntegration::test_4spin_full_pipeline PASSED [ 80%]
tests/test_slice_7.py::TestTopLevelIntegration::test_msgpack_roundtrip PASSED [ 81%]
tests/test_slice_7.py::TestTopLevelIntegration::test_annealing_result_is_dataclass PASSED [ 82%]
tests/test_slice_8.py::TestMsgPackSerialization::test_basic_roundtrip PASSED [ 83%]
tests/test_slice_8.py::TestMsgPackSerialization::test_empty_schedule PASSED [ 84%]
tests/test_slice_8.py::TestMsgPackSerialization::test_single_step PASSED [ 85%]
tests/test_slice_8.py::TestMsgPackSerialization::test_large_schedule PASSED [ 87%]
tests/test_validation.py::TestInputValidation::test_J_must_be_ndarray PASSED [ 88%]
tests/test_validation.py::TestInputValidation::test_h_must_be_ndarray PASSED [ 89%]
tests/test_validation.py::TestInputValidation::test_J_finite_check PASSED [ 90%]
tests/test_validation.py::TestInputValidation::test_h_finite_check PASSED [ 91%]
tests/test_validation.py::TestInputValidation::test_J_symmetry PASSED    [ 92%]
tests/test_validation.py::TestInputValidation::test_J_zero_diagonal PASSED [ 94%]
tests/test_validation.py::TestInputValidation::test_J_square PASSED      [ 95%]
tests/test_validation.py::TestInputValidation::test_h_length_match PASSED [ 96%]
tests/test_validation.py::TestInputValidation::test_J_shape_mismatch_h PASSED [ 97%]
tests/test_validation.py::TestInputValidation::test_budget_steps_int PASSED [ 98%]
tests/test_validation.py::TestInputValidation::test_budget_wallclock_finite_positive PASSED [100%]

================================ tests coverage ================================
_______________ coverage: platform linux, python 3.14.8-final-0 ________________

Name                                                  Stmts   Miss  Cover
-------------------------------------------------------------------------
src/pqblocks_anneal_schedule/__init__.py                  5      0   100%
src/pqblocks_anneal_schedule/annealing_types.py           6      0   100%
src/pqblocks_anneal_schedule/msgpack_serializer.py        7      0   100%
src/pqblocks_anneal_schedule/optimized.py               162    111    31%
src/pqblocks_anneal_schedule/reference/__init__.py        4      0   100%
src/pqblocks_anneal_schedule/reference/annealing.py     125     15    88%
src/pqblocks_anneal_schedule/reference/schedule.py       29      2    93%
src/pqblocks_anneal_schedule/reference/stats.py          55      2    96%
-------------------------------------------------------------------------
TOTAL                                                   393    130    67%
============================== 85 passed in 6.86s ==============================
```

---

## 2. Example 1: Quickstart Execution

Command:

```bash
PYTHONPATH=src python3 examples/quickstart.py
```

Output:

```text
Best Energy: -2.000000
Best Config: [-1 -1 -1 -1]
```

---

## 3. Example 2: Adaptive Schedule & Serialization Execution

Command:

```bash
PYTHONPATH=src python3 examples/02_adaptive_schedule_comparison.py
```

Output:

```text
--- PQBlocks Annealing Schedule: Adaptive Pipeline ---
Graph: 6 spins, max degree = 2
Instance statistics:
  Coupling std dev (sigma_c):  1.0000
  Degree entropy (H_d):        -0.0000
  Frustration density (f_r):   0.5000
Generated schedule with 64 steps.
  Initial: T=1.2247, Gamma=2.4495
  Final:   T=0.4562, Gamma=0.0383
MessagePack serialization: 1283 bytes (round-trip verified).

Annealing results:
  Best energy:   -5.000000
  Best config:   [-1 -1  1  1  1  1]
  Success:       True
  Runs / Steps:  10 runs x 64 steps
  Wall-clock ms: 11.64 ms
```

---

## 4. Leak Detection Verification

Command:

```bash
python3 tools/public.py verify pqblocks-anneal-schedule
```

Output:

```text
verify pqblocks-anneal-schedule: 35 files scanned, 0 leak hits -> reports/leaks-pqblocks-anneal-schedule.txt
```
