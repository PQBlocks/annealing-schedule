"""Tests for S2: Reference schedule generator."""
import math
import pytest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pqblocks_anneal_schedule.reference import generate_schedule_ref


class TestScheduleGenerator:
    """Tests for generate_schedule_ref (S2)."""

    def test_4spin_schedule_exact_match(self):
        """Test schedule computation for 4-spin example from spec."""
        # From spec section 1.5: 4-spin ring
        sigma_c = math.sqrt(0.75)  # ≈ 0.8660254038
        H_d = 0.0
        f_r = 0.75
        stats = (sigma_c, H_d, f_r)
        n = 4
        max_degree = 2
        schedule = generate_schedule_ref(stats, n, max_degree, budget_steps=8)

        expected = [
            (0, 0.8660, 1.7321),
            (1, 0.7166, 1.5155),
            (2, 0.6227, 1.2990),
            (3, 0.5487, 1.0825),
            (4, 0.4876, 0.8660),
            (5, 0.4359, 0.6495),
            (6, 0.3913, 0.4330),
            (7, 0.3526, 0.2165),
        ]

        assert len(schedule) == 8
        for i, (step, T, Gamma) in enumerate(schedule):
            assert step == expected[i][0]
            assert abs(T - expected[i][1]) < 0.0001, f"T mismatch at step {step}"
            assert abs(Gamma - expected[i][2]) < 0.0001, f"Gamma mismatch at step {step}"

    def test_monotonicity(self):
        """T and Gamma should be monotonically non-increasing."""
        stats = (1.0, 1.0, 0.5)
        schedule = generate_schedule_ref(stats, n=10, max_degree=5, budget_steps=128)

        T_prev = float('inf')
        Gamma_prev = float('inf')
        for step, T, Gamma in schedule:
            assert T <= T_prev, f"T not monotonic at step {step}"
            assert Gamma <= Gamma_prev, f"Gamma not monotonic at step {step}"
            T_prev = T
            Gamma_prev = Gamma

    def test_empty_instance_returns_empty_schedule(self):
        """E1: n=0 should return empty schedule."""
        schedule = generate_schedule_ref((0.0, 0.0, 0.0), n=0, max_degree=0)
        assert schedule == []

    def test_single_spin_instance(self):
        """E2: Single spin should work correctly."""
        stats = (1e-10, 0.0, 0.0)
        schedule = generate_schedule_ref(stats, n=1, max_degree=0, budget_steps=4)
        assert len(schedule) == 4
        # T and Gamma should decrease linearly/exp from T0/Gamma0
        assert schedule[0][1] > schedule[-1][1]  # T decreasing
        assert schedule[0][2] > schedule[-1][2]  # Gamma decreasing