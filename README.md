# JaxSafeFilter

**Human-Centered Safety Filter (HCSF) for Autonomous Systems**

A JAX implementation of the safety filter algorithm from "Safety with Agency: Human-Centered Safety Filter with Application for AI-Assisted Motorsport."

## Overview

This implementation provides a safety mechanism that filters human inputs through a learned safety value function, intervening only when necessary while preserving human agency. The key principle is **minimal intervention**: the filter finds the action closest to the human's intent that still satisfies safety constraints.

### The Core Optimization Problem

```
u* = argmin ||u - u_human||²
     subject to: Q(x, u) ≥ γ · V(x)
```

Where:
- `u_human`: The action the human wants to take
- `V(x)`: State safety value (how safe is this state?)
- `Q(x, u)`: State-action safety value (how safe is taking action `u` in state `x`?)
- `γ`: Safety margin (typically 0.7)

## Project Structure

```
JaxSafeFilter/
├── src/
│   ├── model.py       # V and Q neural network architectures
│   ├── filter.py      # HCSF 2000-candidate line search optimization
│   └── utils.py       # Math helpers (ReLU, normalization, LERP)
├── scripts/
│   └── train_v.py     # Training logic for Safety Value Function
├── main.py            # End-to-end demo (State → Safe Action)
├── README.md          
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Quick Demo

```bash
python main.py
```

This runs the HCSF filter on mock driving data, showing:
1. A human's aggressive input (hard steering + full throttle)
2. A safe fallback action (straighten + brake)
3. The filtered output that minimally modifies human intent

### Training the V-Network

```bash
python scripts/train_v.py
```

Trains the safety value function on synthetic data. In production, replace with real trajectories from expert demonstrations.

## How It Works

### Line Search Algorithm

Instead of solving the constrained optimization analytically, we use a practical line search:

1. Generate 2000 candidate actions along `u_human → u_fallback`
2. Evaluate Q(x, u) for each candidate via batched inference
3. Find the first (smallest α) candidate where `Q(x, u) ≥ γ·V(x)`
4. Return that action as the safe output

```python
# The key interpolation
u_candidate = (1 - α) · u_human + α · u_fallback

# α = 0.0: Pure human input (no intervention)
# α = 1.0: Pure fallback (full override)
```

### Action Space (Motorsport Example)

```python
action = [steering, throttle, brake]
# steering: -1.0 (full left) to +1.0 (full right)
# throttle:  0.0 (none) to 1.0 (full)
# brake:     0.0 (none) to 1.0 (full)
```

### State Space

133-dimensional vector capturing:
- Vehicle telemetry (speed, acceleration, orientation)
- Track position and geometry
- Environmental conditions
- Historical trajectory data

## Key Concepts

### Minimal Intervention Principle

The filter doesn't simply override unsafe inputs—it finds the **closest** safe action to human intent. This preserves driver agency while ensuring safety.

### Safety Value Functions

- **V(x)**: Trained to predict long-term safety from state alone
- **Q(x, u)**: Trained to predict safety of taking action `u` in state `x`

The constraint `Q(x, u) ≥ γ·V(x)` ensures the action doesn't make things worse than the baseline state safety.

## Example Output

```
Human Input:     [0.5, 1.0, 0.0]   # Hard right, full throttle
Safe Output:     [0.3, 0.7, 0.1]   # Softer turn, less throttle, light brake
Alpha:           0.234             # 23.4% intervention
```

## References

- "Safety with Agency: Human-Centered Safety Filter with Application for AI-Assisted Motorsport"
- JAX Documentation: https://jax.readthedocs.io/

## License

MIT
