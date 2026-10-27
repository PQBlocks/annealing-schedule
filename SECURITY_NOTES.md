# Security & Numerical Safety Notes: pqblocks-anneal-schedule

> [!NOTE]
> **Domain Classification**: Quantum-inspired classical optimisation: not cryptography, no security claim.

---

## 1. Domain & Scope Classification

`pqblocks-anneal-schedule` is a numerical optimization library implementing quantum-inspired simulated annealing for combinatorial optimization. 

- **Not Cryptography**: This software does NOT implement cryptographic primitives, key encapsulation, encryption, digital signatures, or secure key exchange protocols.
- **No Cryptographic Guarantees**: No claims of post-quantum cryptographic security, side-channel resistance, constant-time execution, or zero-leakage memory hygiene are made.
- **RNG Purpose**: The pseudorandom number generators (PRNGs) employed in this package are designed for statistical quality and simulation determinism. They must **never** be used for cryptographic key generation, nonce creation, or security-sensitive randomness.

---

## 2. Input Validation & Defense-in-Depth

The package enforces strict precondition checking on all input arguments:

- **Matrix Symmetry**: The coupling matrix $J$ must be symmetric ($J_{ij} = J_{ji}$ within numerical tolerance). Asymmetric matrices are rejected with `ValueError`.
- **Zero Diagonal**: Self-couplings ($J_{ii}$) are prohibited in standard Ising formulations; any non-zero diagonal element triggers `ValueError`.
- **Finiteness**: Input matrices $J$ and vectors $h$ are checked for `NaN`, `+Inf`, and `-Inf`. Non-finite values raise `ValueError`.
- **Shape Consistency**: The dimensions of $J$ must be square ($n \times n$) and strictly match the length of the external field vector $h$ ($n$).
- **Positive Budgets**: Annealing step budgets must be positive integers ($S > 0$), and wall-clock budgets must be finite positive numbers ($W > 0.0$).

---

## 3. Numerical Stability & Overflow Protections

- **Metropolis Acceptance Ratio Guarding**:
  In Metropolis spin-flip updates, the acceptance probability is given by:
  
  $$P(\text{accept}) = \min(1, \exp(-\Delta E / T))$$
  
  When temperature $T \to 0$ or $\Delta E \gg 0$, naive exponentiation can cause underflow or overflow floating-point exceptions. The implementation clamps $-\Delta E / T$ to safe numerical ranges ($[-700.0, 0.0]$) before computing `math.exp`, preventing floating-point overflow warnings or crash conditions.
  
- **Variance Floor**:
  When computing coupling variance $\sigma_c^2 = \text{mean}(w^2) - \text{mean}(w)^2$, floating-point cancellation on degenerate graphs can produce tiny negative values (e.g., $-10^{-17}$). The variance calculation is clamped via `max(0.0, var_w)`, and $\sigma_c$ enforces a floor of $10^{-10}$ to prevent division by zero in schedule quench calculations.

---

## 4. Resource Consumption & DoS Prevention

Generating adaptive schedules requires $O(|E|)$ operations to scan graph edges. For extremely dense graphs with large $n$ ($n > 2000$), calculating statistics and generating 100,000 steps may consume several seconds of CPU time. The package warns if schedule generation exceeds 10% of the allocated wall-clock budget.
