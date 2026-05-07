"""Tests for epsilon-cli resonance module."""
import numpy as np
from epsilon_cli.resonance import (
    StochasticResonanceInjector,
    StochasticResonanceConfig,
    run_optimization_benchmark,
)


def test_config_defaults():
    cfg = StochasticResonanceConfig()
    assert cfg.base_gain == 0.02
    assert cfg.min_gain == 0.005
    assert cfg.max_gain == 0.30


def test_injector_initial_state():
    inj = StochasticResonanceInjector()
    assert inj.gain == 0.02
    assert inj._last_metric is None


def test_injector_gain_grows_on_stagnation():
    inj = StochasticResonanceInjector()
    inj.step(0.5)  # Initialize
    initial_gain = inj.gain

    # Stagnate for 3 steps
    inj.step(0.5001)  # tiny improvement < threshold
    inj.step(0.5002)
    inj.step(0.5003)

    # Gain should have grown
    assert inj.gain > initial_gain


def test_injector_gain_decays_on_progress():
    inj = StochasticResonanceInjector()
    inj.step(0.5)
    inj._gain = 0.1  # Set high gain

    # Clear progress
    inj.step(0.7)  # big improvement > threshold

    assert inj.gain < 0.1


def test_inject_noise_shape_preserved():
    inj = StochasticResonanceInjector()
    rng = np.random.default_rng(42)
    arr = np.ones((10, 20))
    noisy = inj.inject_noise(arr, rng)
    assert noisy.shape == arr.shape


def test_benchmark_runs():
    result = run_optimization_benchmark(dim=64, steps=20, seed=42)
    assert result["steps"] == 20
    assert result["final_loss"] < 1.0  # Should converge somewhat
    assert len(result["gain_schedule"]) == 20