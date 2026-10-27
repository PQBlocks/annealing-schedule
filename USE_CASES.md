# Use Cases: pqblocks-anneal-schedule

> [!NOTE]
> **Domain Classification**: Quantum-inspired classical optimisation: not cryptography, no security claim.

---

## Overview

`pqblocks-anneal-schedule` is engineered for combinatorial optimization problems that map naturally to Ising spin systems or Quadratic Unconstrained Binary Optimization (QUBO) formulations.

By tailoring the annealing temperature schedule and transverse field trajectory to the graph-theoretic properties of individual instances, the package delivers enhanced ground state hit rates on complex, structured problems.

---

## 1. MaxCut and Graph Partitioning

MaxCut seeks to partition the vertices of an undirected graph $G = (V, E)$ into two disjoint subsets $S$ and $V \setminus S$ such that the weight of edges crossing the cut is maximized.

In the Ising formulation:
- Each vertex $i \in V$ is assigned a spin variable $s_i \in \{-1, +1\}$.
- For each edge $(i, j)$ with weight $w_{ij}$, the coupling is $J_{ij} = w_{ij}$.
- The objective function is $H(s) = \sum_{(i, j)} J_{ij} s_i s_j$.

### Why Adaptive Scheduling Helps:
- **Degree Heterogeneity**: In graphs with irregular vertex degrees (e.g., scale-free networks, clustered partitions), high-degree nodes flip easily under high temperatures, causing premature trapping. The degree entropy metric ($H_d$) adjusts the quench parameter to allow deeper thermal exploration.
- **Frustrated Cycles**: In graphs with odd cycles or conflicting edge weights, the frustration density metric ($f_r$) scales the transverse field $\Gamma_0$ to assist in tunneling through localized energy barriers.

---

## 2. Precomputed Schedule Caching for High-Throughput Workloads

When solving ensembles of problem instances sharing the same structural characteristics (e.g., repeated runs on graphs generated from the same distribution), recomputing graph statistics and schedules on every run introduces unnecessary overhead.

### Workflow:
1. **Offline Phase**: Extract statistics $(\sigma_c, H_d, f_r)$ and generate the schedule once for the representative topology.
2. **Serialization**: Export the schedule to binary MessagePack format via `schedule_to_msgpack`.
3. **Execution Phase**: Distribute the precomputed byte sequence to worker nodes or GPU threads; deserialize with `schedule_from_msgpack` and execute parallel annealing runs with independent seeds.

```python
from pqblocks_anneal_schedule import schedule_to_msgpack, schedule_from_msgpack

# Precompute and serialize
packed_schedule = schedule_to_msgpack(schedule)

# Distributed worker execution
worker_schedule = schedule_from_msgpack(packed_schedule)
```

---

## 3. Quadratic Unconstrained Binary Optimization (QUBO)

QUBO problems with binary variables $x_i \in \{0, 1\}$ minimize:

$$f(x) = x^T Q x$$

These problems map bijectively to Ising spin systems $s_i \in \{-1, +1\}$ via the substitution $x_i = (s_i + 1) / 2$:

$$J_{ij} = \frac{1}{4} Q_{ij} \quad (i \ne j)$$
$$h_i = \frac{1}{2} Q_{ii} + \sum_{j \ne i} \frac{1}{4} (Q_{ij} + Q_{ji})$$

Once mapped, `pqblocks-anneal-schedule` extracts the coupling variance and frustration of the transformed Ising Hamiltonian to drive schedule optimization.
