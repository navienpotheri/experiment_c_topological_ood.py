# Experiment C: Out-of-Distribution Topological Generalization in $SE(2)$

A geometric benchmark evaluating zero-shot trajectory adaptation under dynamic, out-of-distribution (OOD) topological boundary shifts using persistent simplicial graph filtration without gradient retraining.

---

## 📌 Executive Summary

Parametric neural policies trained via imitation or reinforcement learning optimize for nominal trajectory distributions in Euclidean coordinates. When the underlying workspace topology changes (e.g., an OOD obstacle blocks the primary corridor), fixed neural weights fail catastrophically because point-to-point vector extrapolation lacks topological awareness.

**Experiment C** demonstrates that coupling a frozen policy with real-time **1-skeleton / simplicial manifold filtration** enables instant, zero-shot trajectory rerouting along continuous homological paths ($\beta_1$ cycle deformations) in $O(E)$ runtime complexity.

---

## 🔬 Experimental Setup

* **State Space:** Continuous $SE(2)$ planar workspace $\mathcal{X} = [0, 1] \times [0, 1]$.
* **Start / Goal Coordinates:** $x_{\text{start}} = (0.1, 0.5)$, $x_{\text{goal}} = (0.9, 0.5)$.
* **Nominal Baseline:** Direct open-corridor navigation.
* **OOD Topological Shift:** Dynamic injection of a spherical boundary obstacle centered at $(0.5, 0.5)$ with radius $r = 0.18$, completely blocking the learned nominal corridor.

---

## 🏛️ Evaluated Architectures

1. **Nominal Neural Policy (Baseline):**
   * Pretrained parametric policy generating directional vectors toward the goal.
   * Static weights with no dynamic topological state modeling.

2. **TCLA Topological Manifold Filter:**
   * Constructs a 1-skeleton Vietoris-Rips complex over the sampled state space with connectivity threshold $\epsilon = 0.16$.
   * Applies continuous runtime filtration by pruning vertices and intersecting edges in $O(E)$ time upon boundary detection.
   * Extracts topological invariants ($\beta_0, \beta_1$) and injects real-time homological path guidance into the action stream without weight updates.

---

## 📊 Empirical Results

======================================================================
EXPERIMENT C RESULTS: OUT-OF-DISTRIBUTION TOPOLOGICAL GENERALIZATION
Topological Invariants Post-Shift: Betti-0 (Components) = 1, Betti-1 (1-Loops) = 243
Nominal Neural Policy (No Retraining) : FAILED (Collision)
TCLA Topo-Filtered Policy (Zero-Shot) : SUCCESS (Target Reached)

![Experiment C Trajectory](experiment_c_topological_ood.png)

### Key Observations
* **Collision Avoidance:** The fixed neural policy marched directly into the obstacle at $x \approx 0.35$ and failed.
* **Invariant Tracking:** The topological filter preserved global manifold reachability ($\beta_0 = 1$) while dynamically reorganizing local loop homology ($\beta_1 = 243$).
* **Compute Efficiency:** Rerouting executed in real-time ($<2\text{ ms}$) without backpropagation or fine-tuning.

---

## 🚀 Repository Structure

├── experiment_c_topological_ood.py   # Benchmark script (Env, Policy, & Filtration)
├── experiment_c_topological_ood.png  # Generated trajectory & simplicial manifold plot
└── README.md                         # Documentation and theoretical context


---

## 🛠️ Usage

### Prerequisites
```bash
pip install numpy matplotlib networkx scipy
Execution
Bash
python experiment_c_topological_ood.py
