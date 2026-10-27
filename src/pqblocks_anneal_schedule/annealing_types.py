from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray

@dataclass
class AnnealingResult:
    best_energy: float
    best_config: NDArray[np.int8]            # Shape (n,), values in {-1, +1}
    tts: float                               # Time-to-solution TTS(0.99) in seconds; float('inf') if success_prob == 0
    success: bool                            # True iff best_energy <= target_energy
    success_prob: float                      # Empirical success probability over runs (0.0 to 1.0)
    num_runs: int                            # Total runs attempted (preserved in denominator)
    num_steps: int                           # Steps per run
    stats: Tuple[float, float, float]        # (sigma_c, H_d, f_r) computed for instance
    schedule: List[Tuple[int, float, float]] # Schedule sequence consumed [(step, T, Gamma)]
    wallclock_ms: float                      # Measured wall-clock time in milliseconds
    hardware: str                            # Host CPU and environment fingerprint for benchmark audit
