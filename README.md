# Epsilon CLI

**Context window optimization via stochastic resonance — adaptive noise injection for LLM training.**

---

## Background

Stochastic resonance (SR) is a phenomenon in which the performance of a nonlinear system improves when a moderate level of noise is added to a sub-threshold signal. Originally observed in paleoclimate dynamics (Benzi et al., 1981) and subsequently documented across neural systems, SR has found application in computational optimization where noise can help escaping shallow local minima.

This CLI implements **adaptive stochastic resonance injection** for training signal paths. The core observation is that during LLM training, certain context windows enter saturated states where gradient information becomes sub-threshold. Injecting calibrated noise into these signal paths can temporarily amplify weak signals, improving propagation through saturated layers and accelerating convergence.

---

## Method

### Algorithm

The `StochasticResonanceInjector` maintains a scalar gain parameter `g ∈ [g_min, g_max]` that scales injected noise. The gain is updated adaptively based on metric progress:

```
if metric_improvement < stagnation_threshold:
    g ← min(g × gain_growth, g_max)     # Stagnating: increase noise
else:
    g ← max(g × gain_decay, g_min)      # Progressing: reduce noise
```

**Parameters (defaults):**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_gain` | 0.02 | Initial noise gain |
| `min_gain` | 0.005 | Floor on noise gain |
| `max_gain` | 0.30 | Ceiling on noise gain |
| `stagnation_threshold` | 1e-3 | Delta threshold for stall detection |
| `gain_growth` | 1.15 | Multiplicative factor on stagnation |
| `gain_decay` | 0.96 | Multiplicative factor on progress |

### Noise Injection

For an activation tensor `x`, the injected noisy activation is:

```
x' = x + g · ε,   ε ~ N(0, I)
```

where `g` is the current gain and `ε` is standard Gaussian noise.

### Benchmark

The included benchmark simulates gradient descent on a quadratic loss surface with a known optimum. It demonstrates that SR-injected optimization reaches lower final loss than a no-noise baseline, particularly on loss surfaces with plateau regions.

---

## Installation

```bash
pip install epsilon-cli
```

---

## Usage

### Version Info

```bash
epsilon-info
```

### Run Optimization Benchmark

```bash
epsilon-stochastic --steps 200 --verbose
```

**Options:**

- `--steps INT` — Number of optimization steps (default: 100)
- `--dim INT` — Embedding dimension for synthetic problem (default: 128)
- `--seed INT` — Random seed for reproducibility (default: 42)
- `--verbose` — Print per-step gain and loss values

**Example output:**

```
Running SR benchmark: dim=128, steps=200, seed=42
Final loss: 0.001204
Initial gain: 0.020000
Final gain:   0.005000
```

**With `--verbose`:**

```
Gain schedule (every 10 steps):
  step    0: gain=0.020000  loss=2.841937
  step   10: gain=0.020000  loss=2.179442
  step   20: gain=0.020000  loss=1.652183
  ...
  step  190: gain=0.005000  loss=0.001347
  step  200: gain=0.005000  loss=0.001204
```

---

## API Integration

```python
import numpy as np
from epsilon_cli.resonance import StochasticResonanceInjector, StochasticResonanceConfig

# Customize parameters
config = StochasticResonanceConfig(
    base_gain=0.02,
    stagnation_threshold=1e-3,
    gain_growth=1.15,
    gain_decay=0.96,
)
injector = StochasticResonanceInjector(config)

# In your training loop
rng = np.random.default_rng(42)
for epoch in range(1000):
    metric = train_step()          # Your training logic
    gain = injector.step(metric)  # Adaptive gain update

    # Inject SR noise into activations
    noisy_activations = injector.inject_noise(activations, rng)
```

---

## References

- Benzi, R., Parisi, G., Sutera, A., & Vulpiani, A. (1981). Stochastic resonance in climatic change. *Tellus*, 34(1), 10–16.
- Gammaitoni, L., Hänggi, P., Jung, P., & Marchesoni, F. (1998). Stochastic resonance. *Reviews of Modern Physics*, 70(1), 223.
- McDonnell, M. D., & Ward, L. M. (2011). The benefits of noise in neural systems. *Nature Reviews Neuroscience*, 12(7), 415–426.