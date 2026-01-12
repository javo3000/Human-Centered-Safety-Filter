# Human-Centered Safety Filter (HCSF) — JAX/XLA Implementation

## Project Overview

This repository provides a **high-performance implementation of the Human-Centered Safety Filter (HCSF)**, inspired by research from the **Safe Robotics Lab at Princeton University**.

Traditional safety filters often act as *last-resort overrides*, abruptly replacing human input and degrading user experience. In contrast, this project implements a **smooth, optimization-based safety filter** that preserves **human agency** while enforcing safety constraints. The approach balances user intent with safety by evaluating the **state-value** ( V(x) ) against the **action-value** ( Q(x, u) ) of the human-proposed control.

The result is a safety mechanism that intervenes **only when necessary**, and by the **minimum amount required**.

---

## Key Features & Innovations

### JAX & XLA Acceleration

* Implemented entirely in **JAX**, leveraging the **XLA compiler** for hardware-level optimization.
* Uses `jax.vmap` to evaluate **~2,000 candidate action translations in parallel**, achieving **sub-millisecond latency** on modern accelerators.
* Designed to scale efficiently across GPU, TPU, and ASIC backends.

### State–Action Safety Mapping

* A lightweight **MLP architecture** maps **133-dimensional state vectors** to scalar safety values.
* Enables fast safety evaluation suitable for real-time control loops.

### Minimal-Intervention Optimization

* Implements a **line search** between the human command ( u_{\text{human}} ) and a safe fallback policy ( \pi^{\sharp} ).
* Guarantees the *least deviation from user intent* while satisfying the constraint:

[
Q(x, u) \geq \gamma V(x)
]

### Hardware-Aware Design

* Optimized for **resource-constrained systems** (e.g., SRAM-limited embedded platforms).
* Avoids expensive memory fetches by generating candidate actions **on-the-fly**, reducing bandwidth pressure and improving cache locality.

---

## Technical Stack

* **Core Logic:** JAX (functional programming paradigm)
* **Compiler:** XLA (GPU / TPU / ASIC optimization)
* **Simulation:** Compatible with high-fidelity racing environments
  (e.g., Assetto Corsa, Gym-based simulators)
* **Mathematical Foundation:**

  * Control Barrier Functions (CBFs)
  * Reinforcement Learning (Value-based safety constraints)

---

## Research Context

This implementation is inspired by:

> **Fisac et al.**
> *Safety with Agency: Human-Centered Safety Filter with Application to AI-Assisted Motorsports*

The project aims to translate the paper’s theoretical framework into a **practical, real-time, hardware-efficient system**.

---

## Installation & Usage

```bash
git clone https://github.com/yourusername/JaxSafeFilter.git
cd JaxSafeFilter
pip install jax jaxlib
python main.py --mode simulation
```

---

## Future Work

* Extend the safety filter to **multi-agent settings**.
* Benchmark against classical CBF-based safety filters under identical latency constraints.
* **Design a custom systolic-array ASIC** (via KiCad) tailored to this JAX/XLA workload, enabling ultra-low-latency safety filtering on embedded platforms.

This direction directly bridges **electrical engineering, hardware design, and AI safety**, aligning low-level system optimization with high-level control guarantees.

---

## Final Note

This project is part of a broader effort to explore **human-centered safety, scalable AI infrastructure, and hardware-aware learning systems**. Contributions, discussions, and critiques are welcome.
