"""
Reference implementation tests for the annealing schedule optimizer (S1-S3).
"""
import math
import pytest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pqblocks_anneal_schedule.reference.stats import compute_instance_stats_ref, validate_inputs
from pqblocks_anneal_schedule.reference.schedule import generate_schedule_ref


class TestComputeInstanceStats:
    """Tests for compute_instance_stats_ref (S1)."""

    def test_4spin_worked_example(self):
        """Test case from spec: 4-spin ring with antiferromagnetic defect."""
        n = 4
        J = np.array([[ 0, -1,  0,  1],
                      [-1,  0, -1,  0],
                      [ 0, -1,  0, -1],
                      [ 1,  0, -1,  0]], dtype=float)
        h = np.zeros(4, dtype=float)

        sigma_c, H_d, f_r = compute_instance_stats_ref(J, h)

        # sigma_c = sqrt(0.75) ≈ 0.8660254038
        assert abs(sigma_c - math.sqrt(0.75)) < 1e-6
        # H_d = 0 (all degrees equal)
        assert abs(H_d - 0.0) < 1e-6
        # f_r = 0.75 (3 frustrated edges out of 4)
        assert abs(f_r - 0.75) < 1e-6

    def test_empty_instance(self):
        """E1: Empty instance should return (0.0, 0.0, 0.0)."""
        J = np.zeros((0, 0), dtype=float)
        h = np.zeros(0, dtype=float)
        stats = compute_instance_stats_ref(J, h)
        assert stats == (0.0, 0.0, 0.0)

    def test_degenerate_couplings(self):
        """E4: Uniform couplings J_ij = c should yield sigma_c = epsilon."""
        J = np.ones((3, 3)) * 0.5
        np.fill_diagonal(J, 0)
        h = np.zeros(3)
        sigma_c, H_d, f_r = compute_instance_stats_ref(J, h)
        # Variance of [0.5, 0.5, 0.5] is 0, so sigma_c = epsilon
        assert abs(sigma_c - 1e-10) < 1e-15

    def test_validation_J_not_square(self):
        """Validation: J must be square."""
        J = np.array([[0, 1, 0], [0, 0, 1]])
        h = np.zeros(2)
        with pytest.raises(ValueError, match="square"):
            compute_instance_stats_ref(J, h)

    def test_validation_J_not_symmetric(self):
        """Validation: J must be symmetric."""
        J = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=float)
        h = np.zeros(3)
        with pytest.raises(ValueError, match="symmetric"):
            compute_instance_stats_ref(J, h)

    def test_validation_J_has_diagonal(self):
        """Validation: J must have zero diagonal."""
        J = np.array([[1, 0], [0, 0]], dtype=float)
        h = np.zeros(2)
        with pytest.raises(ValueError, match="zero diagonal"):
            compute_instance_stats_ref(J, h)


class TestGenerateSchedule:
    """Tests for generate_schedule (S2)."""

    def test_4spin_schedule_exact_match(self):
        """Test schedule computation for 4-spin example."""
        stats = (math.sqrt(0.75), 0.0, 0.75)  # from test_4spin_worked_example
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


class TestValidateInputs:
    """Tests for input validation."""

    def test_J_must_be_ndarray(self):
        """J must be numpy array."""
        with pytest.raises(ValueError, match="numpy.ndarray"):
            validate_inputs([[0, 1], [0, 0]], np.zeros(2))

    def test_h_must_be_ndarray(self):
        """h must be numpy array."""
        with pytest.raises(ValueError, match="numpy.ndarray"):
            validate_inputs(np.zeros((2, 2)), [0, 0])

    def test_J_finite_check(self):
        """J must have all finite values."""
        J = np.array([[0, 1], [1, 0]], dtype=float)
        J[0, 1] = np.inf
        h = np.zeros(2)
        with pytest.raises(ValueError, match="finite"):
            compute_instance_stats_ref(J, h)

    def test_h_finite_check(self):
        """h must have all finite values."""
        J = np.zeros((2, 2))
        h = np.array([0.0, np.nan])
        with pytest.raises(ValueError, match="finite"):
            compute_instance_stats_ref(J, h)