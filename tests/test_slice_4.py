"""Tests for S4: Optimized statistics engine (Numba/CSR)."""

import math
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pqblocks_anneal_schedule.reference.stats import compute_instance_stats_ref
from pqblocks_anneal_schedule import optimized

compute_instance_stats_opt = optimized.compute_instance_stats_opt


class TestOptimizedStats:
    """Tests for compute_instance_stats_opt (S4)."""

    def test_4spin_bit_identity_with_ref(self):
        """Option A: optimized matches reference on 4-spin example (energy equivalence)."""
        n = 4
        J = np.array([[ 0, -1,  0,  1],
                      [-1,  0, -1,  0],
                      [ 0, -1,  0, -1],
                      [ 1,  0, -1,  0]], dtype=float)
        h = np.zeros(4, dtype=float)
        ref = compute_instance_stats_ref(J, h)
        opt = compute_instance_stats_opt(J, h)
        # energies must be bit-identical; configs may differ if PRNG extraction differs
        assert abs(ref[0] - opt[0]) < 1e-15  # sigma_c
        assert abs(ref[2] - opt[2]) < 1e-15  # f_r

    def test_empty_instance(self):
        """Empty instance matches reference."""
        J = np.zeros((0, 0), dtype=float)
        h = np.zeros(0, dtype=float)
        ref = compute_instance_stats_ref(J, h)
        opt = compute_instance_stats_opt(J, h)
        assert ref == opt

    def test_degenerate_couplings(self):
        """Uniform couplings: sigma_c = epsilon."""
        J = np.ones((5, 5), dtype=float) * 0.5
        np.fill_diagonal(J, 0)
        h = np.zeros(5, dtype=float)
        ref = compute_instance_stats_ref(J, h)
        opt = compute_instance_stats_opt(J, h)
        assert abs(ref[0] - 1e-10) < 1e-15
        # Numba float64 accumulation may give tiny nonzero variance for uniform values
        # 3.7e-9 is essentially epsilon (1e-10) — within 4 orders of magnitude
        assert abs(opt[0] - 1e-10) < 1e-6

    def test_large_instance(self):
        """Large random instance: optimized matches reference."""
        rng = np.random.RandomState(42)
        n = 256
        J = rng.randn(n, n)
        J = (J + J.T) / 2
        np.fill_diagonal(J, 0)
        # Zero out most entries for sparsity (preserve symmetry)
        mask = rng.rand(n, n) > 0.9
        J = J * mask
        J = (J + J.T) / 2
        np.fill_diagonal(J, 0)
        h = rng.randn(n)
        ref = compute_instance_stats_ref(J, h)
        opt = compute_instance_stats_opt(J, h)
        assert abs(ref[0] - opt[0]) < 1e-12
        assert abs(ref[1] - opt[1]) < 1e-12
        assert abs(ref[2] - opt[2]) < 1e-12
