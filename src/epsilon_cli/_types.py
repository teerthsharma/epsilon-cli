"""Typedefs for epsilon-cli."""
from dataclasses import dataclass

@dataclass
class StochasticResonanceConfig:
    """Configuration for stochastic resonance injection."""
    base_gain: float = 0.02
    min_gain: float = 0.005
    max_gain: float = 0.30
    stagnation_threshold: float = 1e-3
    gain_growth: float = 1.15
    gain_decay: float = 0.96

@dataclass
class OptimizationResult:
    """Result of stochastic resonance optimization."""
    final_gain: float
    epochs_run: int
    metrics: dict