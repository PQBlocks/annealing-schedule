"""Input validation tests per Section 2.1."""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pqblocks_anneal_schedule import compute_instance_stats, generate_schedule


class TestInputValidation:
    """Section 2.1: Input validation & loud errors."""

    def test_J_must_be_ndarray(self):
        """J must be numpy array."""
        with pytest.raises(ValueError, match="numpy.ndarray"):
            compute_instance_stats([[0, 1], [0, 0]], np.zeros(2))

    def test_h_must_be_ndarray(self):
        """h must be numpy array."""
        with pytest.raises(ValueError, match="numpy.ndarray"):
            compute_instance_stats(np.zeros((2, 2)), [0, 0])

    def test_J_finite_check(self):
        """J must have all finite values."""
        J = np.array([[0, 1], [1, 0]], dtype=float)
        J[0, 1] = np.inf
        h = np.zeros(2)
        with pytest.raises(ValueError, match="finite"):
            compute_instance_stats(J, h)

    def test_h_finite_check(self):
        """h must have all finite values."""
        J = np.zeros((2, 2))
        h = np.array([0.0, np.nan])
        with pytest.raises(ValueError, match="finite"):
            compute_instance_stats(J, h)

    def test_J_symmetry(self):
        """J must be symmetric."""
        J = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=float)
        h = np.zeros(3)
        with pytest.raises(ValueError, match="symmetric"):
            compute_instance_stats(J, h)

    def test_J_zero_diagonal(self):
        """J must have zero diagonal."""
        J = np.array([[1, 0], [0, 0]], dtype=float)
        h = np.zeros(2)
        with pytest.raises(ValueError, match="zero diagonal"):
            compute_instance_stats(J, h)

    def test_J_square(self):
        """J must be square."""
        J = np.array([[0, 1, 0], [0, 0, 1]])
        h = np.zeros(2)
        with pytest.raises(ValueError, match="square"):
            compute_instance_stats(J, h)

    def test_h_length_match(self):
        """h length must match J dimension."""
        J = np.zeros((3, 3))
        h = np.zeros(2)
        with pytest.raises(ValueError):
            compute_instance_stats(J, h)

    def test_J_shape_mismatch_h(self):
        """J shape[0] must equal len(h)."""
        J = np.zeros((2, 3))
        h = np.zeros(3)
        with pytest.raises(ValueError, match="square"):
            compute_instance_stats(J, h)

    def test_budget_steps_int(self):
        """budget_steps must be int > 0."""
        stats = (1.0, 0.5, 0.3)
        with pytest.raises(ValueError, match="budget_steps"):
            generate_schedule(stats, 4, 2, budget_steps=0)
        with pytest.raises(ValueError, match="budget_steps"):
            generate_schedule(stats, 4, 2, budget_steps=-1)
        with pytest.raises(ValueError, match="budget_steps"):
            generate_schedule(stats, 4, 2, budget_steps=1.5)

    def test_budget_wallclock_finite_positive(self):
        """budget_wallclock_ms must be finite > 0."""
        stats = (1.0, 0.5, 0.3)
        with pytest.raises(ValueError, match="budget_wallclock_ms"):
            generate_schedule(stats, 4, 2, budget_wallclock_ms=0)
        with pytest.raises(ValueError, match="budget_wallclock_ms"):
            generate_schedule(stats, 4, 2, budget_wallclock_ms=-1.0)
        with pytest.raises(ValueError, match="budget_wallclock_ms"):
            generate_schedule(stats, 4, 2, budget_wallclock_ms=np.inf)
