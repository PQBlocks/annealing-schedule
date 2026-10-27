import numpy as np
from pqblocks_anneal_schedule import simulated_annealing, generate_schedule, compute_instance_stats

J = np.array([[0, -1, 0, 1], [-1, 0, -1, 0], [0, -1, 0, -1], [1, 0, -1, 0]], dtype=float)
h = np.zeros(4, dtype=float)
stats = compute_instance_stats(J, h)
sched = generate_schedule(stats, n=4, max_degree=2, budget_steps=64)
res = simulated_annealing(J, h, sched, seed=42, num_steps=64, num_runs=1)
print(f'Best Energy: {res.best_energy:.6f}')
print('Best Config:', res.best_config)
