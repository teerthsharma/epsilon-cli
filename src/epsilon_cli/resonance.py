"""Stochastic Resonance Injector — adaptive noise injection for context optimization.

Stochastic resonance injects controlled noise into arithmetic signal paths.
The key insight: adding the right amount of noise to a weak signal can
temporarily strengthen it, improving signal propagation through saturated layers.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class StochasticResonanceConfig:
    """Configuration for stochastic resonance injection."""
    base_gain: float = 0.02
    min_gain: float = 0.005
    max_gain: float = 0.30
    stagnation_threshold: float = 1e-3
    gain_growth: float = 1.15
    gain_decay: float = 0.96


class StochasticResonanceInjector:
    """Adaptive environmental-noise mixer for arithmetic signal paths.

    Tracks the loss/metric curve. When improvement stagnates, inject more
    noise (higher gain) to kick the optimizer out of local minima. When
    progress resumes, decay the gain back down.

    Usage:
        injector = StochasticResonanceInjector()
        for epoch in range(1000):
            metric = train_step()  # your training loop
            gain = injector.step(metric)
            # use gain to scale injected noise during training
    """

    def __init__(self, config: StochasticResonanceConfig | None = None):
        self.config = config or StochasticResonanceConfig()
        self._gain = float(self.config.base_gain)
        self._last_metric: float | None = None
        self._epochs_without_progress = 0

    @property
    def gain(self) -> float:
        """Current noise gain."""
        return self._gain

    def step(self, metric: float) -> float:
        """Update gain based on metric progress.

        Args:
            metric: Training metric (higher = better, e.g. accuracy).

        Returns:
            The new gain value to use for noise injection this step.
        """
        if self._last_metric is None:
            self._last_metric = metric
            return self._gain

        delta = metric - self._last_metric

        if delta < self.config.stagnation_threshold:
            # Stagnating: grow gain to inject more noise
            self._epochs_without_progress += 1
            if self._epochs_without_progress >= 2:
                self._gain = min(
                    self._gain * self.config.gain_growth,
                    self.config.max_gain
                )
        else:
            # Progressing: decay gain
            self._epochs_without_progress = 0
            self._gain = max(
                self._gain * self.config.gain_decay,
                self.config.min_gain
            )

        self._last_metric = metric
        return self._gain

    def inject_noise(self, values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Inject scaled noise into a tensor of activations.

        Args:
            values: Input array (any shape)
            rng: Random number generator

        Returns:
            values + noise_scaled_by_gain
        """
        noise = rng.standard_normal(values.shape)
        return values + self._gain * noise

    def summary(self) -> dict:
        """Return current state summary."""
        return {
            "gain": self._gain,
            "last_metric": self._last_metric,
            "epochs_stagnant": self._epochs_without_progress,
        }


def run_optimization_benchmark(
    dim: int = 128,
    steps: int = 100,
    seed: int = 42,
) -> dict:
    """Run a synthetic benchmark demonstrating stochastic resonance.

    Simulates a training loop on a simple quadratic loss surface.
    Shows that adding the right SR noise at stagnation speeds up convergence.

    Returns:
        dict with final metric, gain schedule, convergence info.
    """
    rng = np.random.default_rng(seed)
    config = StochasticResonanceConfig()
    injector = StochasticResonanceInjector(config)

    # Simulated "loss" curve — starts high, has a plateau, then improves
    # True optimum is near zero
    losses = []
    gains = []

    x = rng.standard_normal(dim)
    true_optimum = rng.standard_normal(dim) * 0.1

    for step in range(steps):
        # Quadratic loss: 0.5 * ||x - optimum||^2
        loss = 0.5 * np.mean((x - true_optimum) ** 2)
        losses.append(float(loss))

        gain = injector.step(-loss)  # step() expects higher=better
        gains.append(float(gain))

        # Gradient descent with injected SR noise
        grad = x - true_optimum
        noise = rng.standard_normal(dim)
        step_size = 0.05 * gain + 0.001  # Adaptive: bigger steps when stagnating
        x = x - step_size * grad - 0.01 * gain * noise

    return {
        "steps": steps,
        "final_loss": float(losses[-1]),
        "gain_schedule": gains,
        "loss_curve": losses,
    }