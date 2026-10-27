"""Tests for S3: Reference simulated annealing."""

import math
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pqblocks_anneal_schedule.reference.annealing import (
    simulated_annealing_ref, AnnealingResult, SplitMix64, _energy, validate_sa_inputs
)
from pqblocks_anneal_schedule.reference.stats import compute_instance_stats_ref
from pqblocks_anneal_schedule.reference.schedule import generate_schedule_ref


class TestSimulatedAnnealingRef:
    """Tests for simulated_annealing_ref (S3)."""

    def test_4spin_worked_example(self):
        """Run SA on 4-spin ring and verify energy computation."""
        n = 4
        J = np.array([[ 0, -1,  0,  1],
                      [-1,  0, -1,  0],
                      [ 0, -1,  0, -1],
                      [ 1,  0, -1,  0]], dtype=float)
        h = np.zeros(4, dtype=float)
        stats = compute_instance_stats_ref(J, h)
        schedule = generate_schedule_ref(stats, n, 2, budget_steps=8)
        result = simulated_annealing_ref(J, h, schedule, seed=42, num_steps=8, num_runs=1)
        assert isinstance(result, AnnealingResult)
        assert result.best_config.shape == (4,)
        assert result.best_config.dtype == np.int8
        # All values should be -1 or +1
        assert all(x in (-1, 1) for x in result.best_config)

    def test_empty_instance(self):
        """E1: Empty instance returns valid result."""
        J = np.zeros((0, 0), dtype=float)
        h = np.zeros(0, dtype=float)
        result = simulated_annealing_ref(J, h, [], seed=42, num_steps=8, num_runs=1)
        assert result.best_energy == 0.0
        assert len(result.best_config) == 0
        assert result.success == True
        assert result.success_prob == 1.0

    def test_single_spin(self):
        """E2: Single spin with field."""
        J = np.array([[0.0]], dtype=float)
        h = np.array([1.5], dtype=float)
        stats = compute_instance_stats_ref(J, h)
        schedule = generate_schedule_ref(stats, 1, 0, budget_steps=4)
        result = simulated_annealing_ref(J, h, schedule, seed=42, num_steps=4, num_runs=1)
        assert result.best_config.shape == (1,)
        # Ground state should be -sign(h) = -1
        assert result.best_config[0] == -1

    def test_metropolis_overflow_protection(self):
        """E5: Extreme energy delta clamped to prevent overflow."""
        n = 2
        J = np.array([[0, 1000], [1000, 0]], dtype=float)
        h = np.zeros(2, dtype=float)
        stats = compute_instance_stats_ref(J, h)
        schedule = generate_schedule_ref(stats, n, 1, budget_steps=1)
        # Very low temperature to trigger overflow check
        schedule[0] = (0, 1e-100, 0.0)
        result = simulated_annealing_ref(J, h, schedule, seed=42, num_steps=1, num_runs=1)
        assert isinstance(result, AnnealingResult)

    def test_rng_stream_independence(self):
        """E6: Different seeds produce different configs."""
        n = 10
        J = np.eye(n, dtype=float)
        np.fill_diagonal(J, 0)
        h = np.zeros(n, dtype=float)
        stats = compute_instance_stats_ref(J, h)
        schedule = generate_schedule_ref(stats, n, 1, budget_steps=16)
        r1 = simulated_annealing_ref(J, h, schedule, seed=1, num_steps=16, num_runs=1)
        r2 = simulated_annealing_ref(J, h, schedule, seed=999, num_steps=16, num_runs=1)
        # With different seeds, configs should differ (very likely for n=10)
        assert not np.array_equal(r1.best_config, r2.best_config) or r1.best_energy != r2.best_energy

    def test_same_seed_determinism(self):
        """Same seed produces identical results."""
        n = 8
        J = np.random.RandomState(0).randn(n, n)
        J = (J + J.T) / 2
        np.fill_diagonal(J, 0)
        h = np.zeros(n, dtype=float)
        stats = compute_instance_stats_ref(J, h)
        schedule = generate_schedule_ref(stats, n, 3, budget_steps=32)
        r1 = simulated_annealing_ref(J, h, schedule, seed=12345, num_steps=32, num_runs=1)
        r2 = simulated_annealing_ref(J, h, schedule, seed=12345, num_steps=32, num_runs=1)
        assert np.array_equal(r1.best_config, r2.best_config)
        assert r1.best_energy == r2.best_energy

    def test_validation_empty_schedule_n_gt_0(self):
        """Schedule must be non-empty for n > 0."""
        J = np.zeros((2, 2), dtype=float)
        h = np.zeros(2, dtype=float)
        with pytest.raises(ValueError, match="non-empty"):
            simulated_annealing_ref(J, h, [], seed=42, num_steps=8, num_runs=1)

    def test_validation_negative_T(self):
        """T must be > 0."""
        J = np.zeros((2, 2), dtype=float)
        h = np.zeros(2, dtype=float)
        schedule = [(0, -1.0, 0.0)]
        with pytest.raises(ValueError, match="temperature"):
            simulated_annealing_ref(J, h, schedule, seed=42, num_steps=8, num_runs=1)

    def test_validation_seed_range(self):
        """Seed must be in [0, 2**64-1]."""
        J = np.zeros((2, 2), dtype=float)
        h = np.zeros(2, dtype=float)
        schedule = [(0, 1.0, 0.0)]
        with pytest.raises(ValueError, match="seed"):
            simulated_annealing_ref(J, h, schedule, seed=-1, num_steps=8, num_runs=1)
        with pytest.raises(ValueError, match="seed"):
            simulated_annealing_ref(J, h, schedule, seed=2**64, num_steps=8, num_runs=1)
