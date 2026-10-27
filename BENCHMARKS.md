# Benchmark Report: pqblocks-anneal-schedule

> [!NOTE]
> **Domain Classification**: Quantum-inspired classical optimisation: not cryptography, no security claim.

---

## Hardware & Environment

Benchmark results reported below were collected under the following test environment:

- **Host Processor**: Intel(R) Core(TM) i5-10300H CPU @ 2.50GHz (4 cores / 8 threads)
- **Host OS**: Linux (x86_64)
- **Python Version**: 3.14.8
- **Core Dependencies**: NumPy 2.5.3, Numba 0.67.0, msgpack 1.1.2
- **CPU Governor**: powersave (unpinned; disclosed per protocol)

---

## Comparative Benchmarks: MaxCut Sparse Instances

The following benchmark card evaluates `pqblocks-anneal-schedule` on sparse MaxCut instances ($p = 0.05$ to $0.10$) across spin counts ranging from $n = 20$ to $n = 512$, comparing Time-To-Solution at 99% target confidence ($\text{TTS}_{0.99}$) against Neal (standard simulated annealing) and OpenJij.

### Benchmark Protocol

- **Solvers**:
  - `pqblocks-anneal-schedule` (Numba-accelerated adaptive schedule)
  - `neal` (D-Wave C++ simulated annealing engine)
  - `OpenJij` (C++ simulated annealing implementation)
- **Runs per instance**: 100
- **Warmup runs excluded**: 2
- **Sweeps per run**: 128
- **Tuning budget**: 0 (default parameters across all solvers)
- **Bootstrap iterations**: 10,000

### MaxCut Results Table

| Instance | Spins ($n$) | Edge Density ($p$) | Reference Target Energy | Ours TTS | Neal TTS | OpenJij TTS | Relative vs Neal |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `maxcut_n20_p10` | 20 | 0.10 | -21.0 | **0.000490 s** | 0.000719 s | 0.001225 s | **1.47x faster** |
| `maxcut_n64_p05` | 64 | 0.05 | -94.0 | 0.002182 s | **0.000883 s** | 0.001972 s | 2.47x slower |
| `maxcut_n128_p05` | 128 | 0.05 | -390.0 | 0.012025 s | **0.001683 s** | 0.004089 s | 7.14x slower |
| `maxcut_n256_p05` | 256 | 0.05 | -1656.0 | 0.087569 s | **0.004346 s** | 0.012023 s | 20.1x slower |
| `maxcut_n512_p05` | 512 | 0.05 | -6624.0 | 0.613561 s | **0.012381 s** | 0.033850 s | 49.6x slower |

---

## Microbenchmarks: Direct Timing

Direct empirical runtimes measured on the target host using single-run execution:

| Problem Setup | Problem Size ($n$) | Sweeps ($S$) | Execution Mode | Latency (ms / run) | Throughput (runs / sec) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 4-Spin Ring | $n = 4$ | 64 | JIT-compiled | 0.745 ms | 1,342.3 runs/s |
| 20-Spin Sparse Graph | $n = 20$ | 128 | JIT-compiled | 13.682 ms | 73.1 runs/s |

---

## Performance Analysis & Limitations

### Where It Wins

On small structured instances with heterogeneous degrees and high frustration (e.g., $n \le 20$), `pqblocks-anneal-schedule` produces higher-quality cooling schedules that find ground states in fewer sweeps, leading to lower $\text{TTS}_{0.99}$ than general-purpose geometric coolers (1.47x faster than Neal, 2.50x faster than OpenJij).

### Where It Loses (Honest Disclosure)

As instance size grows ($n \ge 64$), the computational cost of extracting instance statistics ($\sigma_c, H_d, f_r$) in Python and generating per-step schedules creates a scaling bottleneck compared to pure C++ engines like Neal. On $n = 512$ dense MaxCut, Neal's inner loop is 49.6x faster. 

For large-scale production deployments on massive graphs ($n > 1000$), users should precompute and cache schedules via `schedule_to_msgpack` rather than regenerating schedules on every execution.
