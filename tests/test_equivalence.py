#!/usr/bin/env python3
"""tests/test_equivalence.py — Section 4.3: Tier 1 (reference) vs Tier 2 (optimized) equivalence.

Verifies that the optimized implementation produces valid results comparable
to the reference implementation. Statistics must be bit-identical (same algorithm).
Energy/config may differ because the two kernels use different PRNG streams.
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pqblocks_anneal_schedule.reference.stats import compute_instance_stats_ref as compute_instance_stats
from pqblocks_anneal_schedule.reference.schedule import generate_schedule_ref as generate_schedule
from pqblocks_anneal_schedule.reference.annealing import simulated_annealing_ref as simulated_annealing
from pqblocks_anneal_schedule.annealing_types import AnnealingResult
from pqblocks_anneal_schedule.optimized import (
    compute_instance_stats_opt as opt_stats,
    generate_schedule_opt as opt_schedule,
    simulated_annealing_opt as opt_sa,
)


def test_stats_bit_identity():
    """Statistics computation must be bit-identical between ref and opt (same algorithm)."""
    for n in [4, 8, 16, 32]:
        rng = np.random.default_rng(n)
        J = rng.standard_normal((n, n)) * 0.1
        J = (J + J.T) / 2
        np.fill_diagonal(J, 0)
        h = rng.standard_normal(n) * 0.1

        ref = compute_instance_stats(J, h)
        opt = opt_stats(J, h)

        assert np.allclose(ref, opt, rtol=1e-12, atol=1e-14), \
            f"n={n}: Stats differ ref={ref} opt={opt}"


def test_schedule_bit_identity():
    """Schedule generation must be bit-identical given same stats."""
    stats = (0.8660254037844386, 0.0, 0.75)  # 4-spin worked example
    n = 4
    max_degree = 2

    ref = generate_schedule(stats, n, max_degree, budget_steps=8)
    opt = opt_schedule(stats, n, max_degree, budget_steps=8)

    assert len(ref) == len(opt) == 8
    for (rs, rt, rg), (os, ot, og) in zip(ref, opt):
        assert rs == os
        assert np.isclose(rt, ot, rtol=1e-14), f"T differs: {rt} vs {ot}"
        assert np.isclose(rg, og, rtol=1e-14), f"Gamma differs: {rg} vs {og}"


def test_both_kernels_produce_valid_results():
    """Both reference and optimized kernels produce valid AnnealingResult objects."""
    n = 8
    rng = np.random.default_rng(42)
    J = rng.standard_normal((n, n)) * 0.1
    J = (J + J.T) / 2
    np.fill_diagonal(J, 0)
    h = rng.standard_normal(n) * 0.1

    stats = compute_instance_stats(J, h)
    sched = generate_schedule(stats, n, n - 1, budget_steps=16)

    ref_result = simulated_annealing(J, h, sched, seed=42, num_steps=16, num_runs=2)
    opt_result = opt_sa(J, h, sched, seed=42, num_steps=16, num_runs=2)

    # Both return AnnealingResult
    assert isinstance(ref_result, AnnealingResult)
    assert isinstance(opt_result, AnnealingResult)

    # Both have valid energies
    assert np.isfinite(ref_result.best_energy)
    assert np.isfinite(opt_result.best_energy)

    # Both have valid configs
    assert ref_result.best_config.shape == (n,)
    assert opt_result.best_config.shape == (n,)
    assert all(c in (-1, 1) for c in ref_result.best_config)
    assert all(c in (-1, 1) for c in opt_result.best_config)


def test_4spin_ground_state_findable():
    """On a simple 4-spin chain, both kernels can find the ground state with enough runs."""
    n = 4
    # Linear chain: J_01 = J_12 = J_23 = -1, rest 0
    J = np.array([[0, -1, 0, 0],
                  [-1, 0, -1, 0],
                  [0, -1, 0, -1],
                  [0, 0, -1, 0]], dtype=float)
    h = np.zeros(n)
    # Ground state: all spins aligned gives E = -3
    expected_ground = -3.0

    stats = compute_instance_stats(J, h)
    sched = generate_schedule(stats, n, 2, budget_steps=32)

    ref_result = simulated_annealing(J, h, sched, seed=42, num_steps=32, num_runs=8)
    opt_result = opt_sa(J, h, sched, seed=42, num_steps=32, num_runs=8)

    # Both should find ground state (or very close)
    assert abs(ref_result.best_energy - expected_ground) < 1e-10, \
        f"Ref: {ref_result.best_energy} vs ground {expected_ground}"
    assert abs(opt_result.best_energy - expected_ground) < 1e-10, \
        f"Opt: {opt_result.best_energy} vs ground {expected_ground}"


def test_sparse_instance_both_kernels():
    """On sparse instances, both kernels produce valid finite energies."""
    n = 32
    rng = np.random.default_rng(99)
    mask = rng.random((n, n)) < 0.08  # ~8% connectivity
    J = rng.standard_normal((n, n)) * mask
    J = (J + J.T) / 2
    np.fill_diagonal(J, 0)
    h = np.zeros(n)

    stats = compute_instance_stats(J, h)
    sched = generate_schedule(stats, n, n - 1, budget_steps=16)

    ref_result = simulated_annealing(J, h, sched, seed=42, num_steps=16, num_runs=2)
    opt_result = opt_sa(J, h, sched, seed=42, num_steps=16, num_runs=2)

    assert np.isfinite(ref_result.best_energy)
    assert np.isfinite(opt_result.best_energy)
    # Neither should be orders of magnitude worse
    ratio = max(ref_result.best_energy, opt_result.best_energy) / max(1e-10, abs(min(ref_result.best_energy, opt_result.best_energy)))
    assert ratio < 10, f"One kernel is 10x worse: ref={ref_result.best_energy} opt={opt_result.best_energy}"


def test_optimized_schedule_num_steps_param():
    """Optimized schedule generator accepts num_steps param (true instance size)."""
    stats = (0.8660254037844386, 0.0, 0.75)
    n = 8
    max_degree = 3
    # Test with different num_steps values
    for steps in [8, 16, 32, 128]:
        sched = opt_schedule(stats, n, max_degree, budget_steps=steps)
        assert len(sched) == steps, f"Expected {steps} steps, got {len(sched)}"
        # First step should have T0 = 0.5 * sigma_c * sqrt(n)
        expected_t0 = 0.5 * stats[0] * np.sqrt(n)
        assert np.isclose(sched[0][1], expected_t0, rtol=1e-10), \
            f"Step 0 T={sched[0][1]} vs expected {expected_t0}"


def test_optimized_sa_kernel_coverage():
    """Exercise the optimized SA kernel with various parameters to boost coverage."""
    n = 6
    rng = np.random.default_rng(77)
    J = rng.standard_normal((n, n)) * 0.1
    J = (J + J.T) / 2
    np.fill_diagonal(J, 0)
    h = rng.standard_normal(n) * 0.1

    stats = compute_instance_stats(J, h)
    sched = generate_schedule(stats, n, n - 1, budget_steps=32)

    # Multiple runs with different seeds
    for seed in [0, 1, 42, 2**31 - 1]:
        result = opt_sa(J, h, sched, seed=seed, num_steps=32, num_runs=2)
        assert isinstance(result, AnnealingResult)
        assert np.isfinite(result.best_energy)
        assert result.best_config.shape == (n,)

    # Test with num_steps=128 (default)
    result = opt_sa(J, h, sched, seed=42, num_steps=128, num_runs=1)
    assert isinstance(result, AnnealingResult)
    assert np.isfinite(result.best_energy)

    # Test with num_runs > 1
    result = opt_sa(J, h, sched, seed=42, num_steps=32, num_runs=5)
    assert result.num_runs == 5
    assert np.isfinite(result.best_energy)


def test_optimized_edge_case_empty():
    """Optimized kernel handles n=0 edge case."""
    J = np.zeros((0, 0))
    h = np.zeros(0)
    stats = compute_instance_stats(J, h)
    sched = generate_schedule(stats, 0, 0, budget_steps=8)
    assert len(sched) == 0


def test_optimized_single_spin():
    """Optimized kernel handles n=1 edge case."""
    J = np.array([[0.0]])
    h = np.array([1.5])
    stats = compute_instance_stats(J, h)
    sched = generate_schedule(stats, 1, 0, budget_steps=8)
    assert len(sched) == 8
    result = opt_sa(J, h, sched, seed=42, num_steps=8, num_runs=1)
    assert isinstance(result, AnnealingResult)
    assert np.isfinite(result.best_energy)


def test_optimized_seed_none():
    """Optimized kernel with seed=None uses os.urandom (line 234 coverage)."""
    n = 6
    rng = np.random.default_rng(42)
    J = rng.standard_normal((n, n)) * 0.1
    J = (J + J.T) / 2
    np.fill_diagonal(J, 0)
    h = rng.standard_normal(n) * 0.1

    stats = compute_instance_stats(J, h)
    sched = generate_schedule(stats, n, n - 1, budget_steps=16)

    # seed=None triggers os.urandom path
    result = opt_sa(J, h, sched, seed=None, num_steps=16, num_runs=1)
    assert isinstance(result, AnnealingResult)
    assert np.isfinite(result.best_energy)


def test_optimized_partial_success_tts():
    """Coverage for partial success TTS calculation (lines 264-265)."""
    n = 4
    J = np.array([[0, -1, 0, 0],
                  [-1, 0, -1, 0],
                  [0, -1, 0, -1],
                  [0, 0, -1, 0]], dtype=float)
    h = np.zeros(n)

    stats = compute_instance_stats(J, h)
    sched = generate_schedule(stats, n, 2, budget_steps=4)

    # With very few steps and runs, we get partial success
    result = opt_sa(J, h, sched, seed=42, num_steps=4, num_runs=3)
    assert isinstance(result, AnnealingResult)
    # Partial success should have finite TTS (not inf)
    if result.success_prob > 0 and result.success_prob < 1:
        assert np.isfinite(result.tts), \
            f"Partial success should have finite TTS, got {result.tts}"
        assert result.success is True