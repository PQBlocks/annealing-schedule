"""Tests for S5: Optimized schedule generator."""

import math
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pqblocks_anneal_schedule.reference.schedule import generate_schedule_ref
from pqblocks_anneal_schedule import optimized
generate_schedule_opt = optimized.generate_schedule_opt


class TestOptimizedSchedule:
    """Tests for generate_schedule_opt (S5)."""

    def test_4spin_bit_identity_with_ref(self):
        """Optimized schedule matches reference on 4-spin example."""
        stats = (math.sqrt(0.75), 0.0, 0.75)
        n, max_degree = 4, 2
        ref = generate_schedule_ref(stats, n, max_degree, budget_steps=8)
        opt = generate_schedule_opt(stats, n, max_degree, budget_steps=8)
        assert len(ref) == len(opt) == 8
        for (rs, rT, rG), (os_,oT, oG) in zip(ref, opt):
            assert rs == os_
            assert abs(rT - oT) < 1e-15
            assert abs(rG - oG) < 1e-15

    def test_empty_instance(self):
        """Empty instance returns empty schedule."""
        ref = generate_schedule_ref((0.0, 0.0, 0.0), 0, 0, budget_steps=8)
        opt = generate_schedule_opt((0.0, 0.0, 0.0), 0, 0, budget_steps=8)
        assert ref == opt == []

    def test_monotonicity(self):
        """Optimized schedule is monotonic."""
        stats = (1.0, 1.0, 0.5)
        sched = generate_schedule_opt(stats, n=10, max_degree=5, budget_steps=128)
        T_prev = float('inf')
        G_prev = float('inf')
        for step, T, G in sched:
            assert T <= T_prev
            assert G <= G_prev
            T_prev = T
            G_prev = G

    def test_large_instance(self):
        """Large instance schedule matches reference."""
        stats = (2.5, 1.5, 0.3)
        ref = generate_schedule_ref(stats, n=512, max_degree=10, budget_steps=256)
        opt = generate_schedule_opt(stats, n=512, max_degree=10, budget_steps=256)
        assert len(ref) == len(opt) == 256
        for (rs, rT, rG), (os_,oT, oG) in zip(ref, opt):
            assert rs == os_
            assert abs(rT - oT) < 1e-12
            assert abs(rG - oG) < 1e-12
