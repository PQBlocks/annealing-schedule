"""Tests for S8: MessagePack serialization."""

import math
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pqblocks_anneal_schedule import generate_schedule, schedule_to_msgpack, schedule_from_msgpack


class TestMsgPackSerialization:
    """Tests for MessagePack wire protocol (S8)."""

    def test_basic_roundtrip(self):
        """Schedule round-trips through MessagePack."""
        stats = (1.0, 0.5, 0.3)
        schedule = generate_schedule(stats, n=10, max_degree=5, budget_steps=64)
        packed = schedule_to_msgpack(schedule)
        reconstructed = schedule_from_msgpack(packed)
        assert len(reconstructed) == len(schedule)
        for (rs, rT, rG), (os_,oT, oG) in zip(schedule, reconstructed):
            assert rs == os_
            assert abs(rT - oT) < 1e-15
            assert abs(rG - oG) < 1e-15

    def test_empty_schedule(self):
        """Empty schedule serializes correctly."""
        packed = schedule_to_msgpack([])
        reconstructed = schedule_from_msgpack(packed)
        assert reconstructed == []

    def test_single_step(self):
        """Single step schedule."""
        schedule = [(0, 1.0, 0.5)]
        packed = schedule_to_msgpack(schedule)
        reconstructed = schedule_from_msgpack(packed)
        assert reconstructed == schedule

    def test_large_schedule(self):
        """Large schedule (1024 steps) round-trips."""
        stats = (2.0, 1.0, 0.5)
        schedule = generate_schedule(stats, n=100, max_degree=10, budget_steps=1024)
        packed = schedule_to_msgpack(schedule)
        reconstructed = schedule_from_msgpack(packed)
        assert len(reconstructed) == 1024
        for (rs, rT, rG), (os_,oT, oG) in zip(schedule, reconstructed):
            assert rs == os_
            assert abs(rT - oT) < 1e-15
            assert abs(rG - oG) < 1e-15
