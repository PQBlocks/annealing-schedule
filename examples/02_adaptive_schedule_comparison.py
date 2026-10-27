#!/usr/bin/env python3
"""Example 02: Instance Statistics, Adaptive Schedule Generation, and MessagePack Serialization.

Demonstrates:
1. Defining an Ising model problem (J coupling matrix and h field vector).
2. Extracting graph statistics (coupling variance, degree entropy, frustration density).
3. Generating an instance-adaptive annealing schedule.
4. Serializing and deserializing the schedule via MessagePack.
5. Executing simulated annealing with seed determinism.
"""
import numpy as np
from pqblocks_anneal_schedule import (
    compute_instance_stats,
    generate_schedule,
    simulated_annealing,
    schedule_to_msgpack,
    schedule_from_msgpack,
)


def main():
    print("--- PQBlocks Annealing Schedule: Adaptive Pipeline ---")
    
    # 1. Define an Ising instance (6-spin frustrated ring)
    n = 6
    J = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = (i + 1) % n
        J[i, j] = -1.0 if i % 2 == 0 else 1.0  # Alternating ferromagnetic / antiferromagnetic
        J[j, i] = J[i, j]
    h = np.zeros(n, dtype=float)
    h[0] = 0.5
    h[3] = -0.5
    
    max_degree = int(np.max(np.sum(np.abs(J) > 0, axis=1)))
    print(f"Graph: {n} spins, max degree = {max_degree}")

    # 2. Extract instance statistics
    sigma_c, H_d, f_r = compute_instance_stats(J, h)
    print(f"Instance statistics:")
    print(f"  Coupling std dev (sigma_c):  {sigma_c:.4f}")
    print(f"  Degree entropy (H_d):        {H_d:.4f}")
    print(f"  Frustration density (f_r):   {f_r:.4f}")

    # 3. Generate adaptive schedule
    budget_steps = 64
    schedule = generate_schedule(
        stats=(sigma_c, H_d, f_r),
        n=n,
        max_degree=max_degree,
        budget_steps=budget_steps,
    )
    print(f"Generated schedule with {len(schedule)} steps.")
    print(f"  Initial: T={schedule[0][1]:.4f}, Gamma={schedule[0][2]:.4f}")
    print(f"  Final:   T={schedule[-1][1]:.4f}, Gamma={schedule[-1][2]:.4f}")

    # 4. MessagePack round-trip
    packed = schedule_to_msgpack(schedule)
    unpacked = schedule_from_msgpack(packed)
    assert len(unpacked) == len(schedule)
    print(f"MessagePack serialization: {len(packed)} bytes (round-trip verified).")

    # 5. Execute simulated annealing
    result = simulated_annealing(
        J=J,
        h=h,
        schedule=unpacked,
        seed=12345,
        num_steps=budget_steps,
        num_runs=10,
    )
    print("\nAnnealing results:")
    print(f"  Best energy:   {result.best_energy:.6f}")
    print(f"  Best config:   {result.best_config}")
    print(f"  Success:       {result.success}")
    print(f"  Runs / Steps:  {result.num_runs} runs x {result.num_steps} steps")
    print(f"  Wall-clock ms: {result.wallclock_ms:.2f} ms")


if __name__ == "__main__":
    main()
