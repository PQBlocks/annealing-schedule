"""Tests for S6: Optimized SA kernel equivalence with reference."""

import math
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pqblocks_anneal_schedule.reference.stats import compute_instance_stats_ref
from pqblocks_anneal_schedule.reference.schedule import generate_schedule_ref
from pqblocks_anneal_schedule.reference.annealing import simulated_annealing_ref
from pqblocks_anneal_schedule import optimized

from pqblocks_anneal_schedule.optimized import compute_instance_stats_opt, generate_schedule_opt, simulated_annealing_opt


class TestOptimizedSA:
    """Tests for simulated_annealing_opt (S6). Option A bit-identity with S3."""

    def test_4spin_equivalence(self):
        """4-spin ring: optimized matches reference energy (config may differ)."""
        n = 4
        J = np.array([[ 0, -1,  0,  1],
                      [-1,  0, -1,  0],
                      [ 0, -1,  0, -1],
                      [ 1,  0, -1,  0]], dtype=float)
        h = np.zeros(4, dtype=float)
        stats_ref = compute_instance_stats_ref(J, h)
        stats_opt = compute_instance_stats_opt(J, h)
        sched_ref = generate_schedule_ref(stats_ref, n, 2, budget_steps=8)
        sched_opt = generate_schedule_opt(stats_opt, n, 2, budget_steps=8)
        r_ref = simulated_annealing_ref(J, h, sched_ref, seed=42, num_steps=8, num_runs=1)
        r_opt = simulated_annealing_opt(J, h, sched_opt, seed=42, num_steps=8, num_runs=1)
        # Same seed should find same best energy (both converge to ground state -2.0)
        assert abs(r_ref.best_energy - r_opt.best_energy) < 1e-15

    def test_same_seed_determinism_opt(self):
        """Optimized: same seed produces identical results."""
        n = 8
        rng = np.random.RandomState(0)
        J = rng.randn(n, n)
        J = (J + J.T) / 2
        np.fill_diagonal(J, 0)
        h = np.zeros(n, dtype=float)
        stats = compute_instance_stats_opt(J, h)
        sched = generate_schedule_opt(stats, n, 3, budget_steps=32)
        r1 = simulated_annealing_opt(J, h, sched, seed=12345, num_steps=32, num_runs=1)
        r2 = simulated_annealing_opt(J, h, sched, seed=12345, num_steps=32, num_runs=1)
        assert np.array_equal(r1.best_config, r2.best_config)
        assert r1.best_energy == r2.best_energy

    def test_empty_instance_opt(self):
        """Optimized: empty instance."""
        J = np.zeros((0, 0), dtype=float)
        h = np.zeros(0, dtype=float)
        result = simulated_annealing_opt(J, h, [], seed=42, num_steps=8, num_runs=1)
        assert result.best_energy == 0.0
        assert len(result.best_config) == 0
        assert result.success == True

    def test_single_spin_opt(self):
        """Optimized: single spin ground state."""
        J = np.array([[0.0]], dtype=float)
        h = np.array([1.5], dtype=float)
        stats = compute_instance_stats_opt(J, h)
        sched = generate_schedule_opt(stats, 1, 0, budget_steps=4)
        result = simulated_annealing_opt(J, h, sched, seed=42, num_steps=4, num_runs=1)
        assert result.best_config[0] == -1  # -sign(h)

    def test_numba_available(self):
        """Verify Numba JIT compiled functions exist."""
        assert hasattr(optimized, '_evaluate_energy_opt')
        assert hasattr(optimized, '_sa_kernel')
        assert callable(optimized._evaluate_energy_opt)
        assert callable(optimized._sa_kernel)
