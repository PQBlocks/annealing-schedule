"""Tests for S7: Top-level integration."""

import math
import pytest
import numpy as np

try:
    import pqblocks_anneal_schedule as src
    from pqblocks_anneal_schedule import (
        AnnealingResult,
        compute_instance_stats,
        generate_schedule,
        simulated_annealing,
        schedule_to_msgpack,
        schedule_from_msgpack,
    )
except ImportError:
    import pqblocks_anneal_schedule
    from pqblocks_anneal_schedule import (
        AnnealingResult,
        compute_instance_stats,
        generate_schedule,
        simulated_annealing,
        schedule_to_msgpack,
        schedule_from_msgpack,
    )


class TestTopLevelIntegration:
    """End-to-end integration tests for public API (S7)."""

    def test_public_api_exists(self):
        """All public symbols exported."""
        assert hasattr(src, 'AnnealingResult')
        assert hasattr(src, 'compute_instance_stats')
        assert hasattr(src, 'generate_schedule')
        assert hasattr(src, 'simulated_annealing')
        assert hasattr(src, 'schedule_to_msgpack')
        assert hasattr(src, 'schedule_from_msgpack')

    def test_4spin_full_pipeline(self):
        """Full pipeline: stats -> schedule -> SA on 4-spin ring."""
        n = 4
        J = np.array([[ 0, -1,  0,  1],
                      [-1,  0, -1,  0],
                      [ 0, -1,  0, -1],
                      [ 1,  0, -1,  0]], dtype=float)
        h = np.zeros(4, dtype=float)
        stats = compute_instance_stats(J, h)
        schedule = generate_schedule(stats, n, 2, budget_steps=8)
        result = simulated_annealing(J, h, schedule, seed=42, num_steps=8, num_runs=1)
        assert isinstance(result, AnnealingResult)
        assert result.best_config.shape == (4,)

    def test_msgpack_roundtrip(self):
        """MessagePack serialization round-trip."""
        stats = (math.sqrt(0.75), 0.0, 0.75)
        n, max_degree = 4, 2
        schedule = generate_schedule(stats, n, max_degree, budget_steps=8)
        packed = schedule_to_msgpack(schedule)
        assert isinstance(packed, bytes)
        assert len(packed) > 0
        reconstructed = schedule_from_msgpack(packed)
        assert len(reconstructed) == len(schedule)
        for (rs, rT, rG), (os_,oT, oG) in zip(schedule, reconstructed):
            assert rs == os_
            assert abs(rT - oT) < 1e-15
            assert abs(rG - oG) < 1e-15

    def test_annealing_result_is_dataclass(self):
        """AnnealingResult is a proper dataclass."""
        n = 4
        J = np.array([[ 0, -1,  0,  1],
                      [-1,  0, -1,  0],
                      [ 0, -1,  0, -1],
                      [ 1,  0, -1,  0]], dtype=float)
        h = np.zeros(4, dtype=float)
        stats = compute_instance_stats(J, h)
        schedule = generate_schedule(stats, n, 2, budget_steps=8)
        result = simulated_annealing(J, h, schedule, seed=42, num_steps=8, num_runs=1)
        assert hasattr(result, 'best_energy')
        assert hasattr(result, 'best_config')
        assert hasattr(result, 'tts')
        assert hasattr(result, 'success')
        assert hasattr(result, 'success_prob')
        assert hasattr(result, 'num_runs')
        assert hasattr(result, 'num_steps')
        assert hasattr(result, 'stats')
        assert hasattr(result, 'schedule')
        assert hasattr(result, 'wallclock_ms')
        assert hasattr(result, 'hardware')