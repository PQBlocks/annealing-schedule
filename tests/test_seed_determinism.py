#!/usr/bin/env python3
"""tests/test_seed_determinism.py — Seed determinism & SplitMix64.

Same seed must produce bit-identical results across multiple runs.
Different seeds must produce uncorrelated streams.
"""
import numpy as np
from pqblocks_anneal_schedule import compute_instance_stats, generate_schedule, simulated_annealing


def test_same_seed_identical():
    """Same seed + same inputs → bit-identical AnnealingResult."""
    n = 4
    J = np.array([[0, -1, 0, 1], [-1, 0, -1, 0], [0, -1, 0, -1], [1, 0, -1, 0]], dtype=float)
    h = np.zeros(n)
    stats = compute_instance_stats(J, h)
    sched = generate_schedule(stats, n, 2, budget_steps=8)
    
    r1 = simulated_annealing(J, h, sched, seed=42, num_steps=8, num_runs=1)
    r2 = simulated_annealing(J, h, sched, seed=42, num_steps=8, num_runs=1)
    
    assert r1.best_energy == r2.best_energy, f"Energy mismatch: {r1.best_energy} vs {r2.best_energy}"
    assert np.array_equal(r1.best_config, r2.best_config), "Config mismatch with same seed"
    # TTS/wallclock are timing-dependent; energy+config+success are deterministic
    assert r1.success == r2.success


def test_different_seeds_different():
    """Different seeds → (very likely) different results."""
    n = 4
    J = np.array([[0, -1, 0, 1], [-1, 0, -1, 0], [0, -1, 0, -1], [1, 0, -1, 0]], dtype=float)
    h = np.zeros(n)
    stats = compute_instance_stats(J, h)
    sched = generate_schedule(stats, n, 2, budget_steps=8)
    
    r1 = simulated_annealing(J, h, sched, seed=42, num_steps=8, num_runs=1)
    r2 = simulated_annealing(J, h, sched, seed=99, num_steps=8, num_runs=1)
    
    # With 8 steps on a 4-spin system, different seeds should differ
    # (not guaranteed but extremely likely)
    assert hash(r1.best_config.tobytes()) != hash(r2.best_config.tobytes()) or \
           r1.best_energy != r2.best_energy, \
        "Different seeds should produce different results"


def test_seed_range_accepted():
    """Seeds across the valid range [0, 2**64-1] are accepted."""
    n = 4
    J = np.array([[0, -1, 0, 0], [-1, 0, -1, 0], [0, -1, 0, -1], [0, 0, -1, 0]], dtype=float)
    h = np.zeros(n)
    stats = compute_instance_stats(J, h)
    sched = generate_schedule(stats, n, 2, budget_steps=4)
    
    for seed in [0, 1, 2**31 - 1, 2**31, 2**63 - 1, 2**64 - 1]:
        result = simulated_annealing(J, h, sched, seed=seed, num_steps=4, num_runs=1)
        assert result.best_energy is not None and np.isfinite(result.best_energy), \
            f"Seed {seed} produced non-finite energy"


def test_seed_none_uses_os_entropy():
    """seed=None should work (uses OS entropy)."""
    n = 4
    J = np.array([[0, -1, 0, 0], [-1, 0, -1, 0], [0, -1, 0, -1], [0, 0, -1, 0]], dtype=float)
    h = np.zeros(n)
    stats = compute_instance_stats(J, h)
    sched = generate_schedule(stats, n, 2, budget_steps=4)
    
    result = simulated_annealing(J, h, sched, seed=None, num_steps=4, num_runs=1)
    assert result.best_energy is not None and np.isfinite(result.best_energy)
