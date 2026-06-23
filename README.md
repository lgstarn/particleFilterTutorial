# Optimal Estimation and Particle Filtering Data Processing Pipeline

## Overview
A "filter" in this context describes a mathematical process that attempts to remove an unwanted component—specifically noise—from streaming data. This repository implements an L0 to L1 data processing pipeline to solve Bayesian state estimation problems. 

The software can be analyzed either from the perspective of stochastic differential equations (SDEs) or from a probability perspective. The pipeline compares the optimal, closed-form properties of the Kalman Filter against sequential Monte Carlo approximations (Particle Filters) for a synthetic 2D tracking target.

## System Dynamics
For the purpose of testing, we simulate a target moving in $\mathbb{R}^2$ under random acceleration. Given a sequence of noisy positional observations, the filters attempt to predict the "optimal" internal state alongside a dynamically calculated covariance representing the prediction's uncertainty.

We imagine a physical system modeled by the system of stochastic differential equations:
$dx/dt = u$
$dy/dt = v$
$du/dt = a_x$
$dv/dt = a_y$

Where $(a_x, a_y)$ represents a continuous-time, multivariate-Gaussian white noise process. This continuous-time system is approximated into a discrete-time backward Euler difference equation, mapping a hidden state vector $x_k$ to an imperfect observation $z_k$.

## Processing Levels
* **L0 (Raw Telemetry):** Uncalibrated, noisy sensor measurements ($Z$) generated via the injected `StateSpaceModel` system configurations.
* **L1 (Processed Output):** Derived outputs ($X_{hat}$, $P_{cov}$) estimated by the filtering framework, complete with 1, 2, and $3\sigma$ dynamic confidence boundaries.

## Implemented Filter Variants

1. **Kalman Filter**: The exact analytic solution for optimal state estimation. It assumes the state and observation models are strictly linear, and that all associated probability density functions are perfectly Gaussian. 
2. **Sampling Importance Resampling (SIR) Particle Filter**: Estimates the state by running the same system equations many times across different stochastic paths (particles). Particle survival is based on likelihood weights. Degeneracy is mitigated via systematic resampling.
3. **Auxiliary Sampling Importance Resampling (ASIR)**: Improves upon standard SIR by pulling the *current* observation into the resampling phase *before* propagating the particles, maximizing computational efficiency when sensor variance is low.
4. **Continuous/Implicit Resampling (CIR)**: Maps particles to the high-probability region of the target posterior using an explicit minimization step (optimization over the log-likelihood) prior to sampling, preventing sample depletion.
5. **Brownian Bridge Solver**: An independent SDE path sampler utilizing chainless implicit methods to generate random walks constrained rigidly by both initial and terminal boundary values.

## Usage
The repository is fully managed via a robust CLI implementation.

```bash
# Run all filters in sequence (Default)
python tracking_pipeline.py 

# Run a specific filter with custom particle count and animation playback speed
python tracking_pipeline.py --filter asir --particles 250 --frame-skip 4

# Run the implicit chainless Brownian Bridge SDE sampler
python tracking_pipeline.py -
-run-bridge
```

---

## The State of the Art (Post-2009)

The foundational work establishing SIR, ASIR, and Implicit Resampling (such as Chorin's 2009 Implicit Particle Filter) represents an era of data assimilation focused heavily on overcoming weight degeneracy in relatively restricted state spaces. In recent years, the state of the art in Bayesian filtering has undergone two massive shifts: surviving the "curse of dimensionality" and integrating deep learning.

### 1. The High-Dimensional Collapse Problem
As researchers attempted to apply particle filters to massive physical systems (e.g., global weather forecasting models containing millions of state variables), they discovered a mathematical roadblock: standard particle filters collapse exponentially as the dimension scales. Snyder et al. (2008) formally proved that unless the ensemble size is exponentially large, a single particle will absorb a weight of 1.0 while all others vanish. The state of the art has since evolved to utilize **Localized Particle Filters (LPF)**, which compute localized, geographic sub-weights to completely bypass global degeneracy.

### 2. The Ensemble Kalman Particle Filter (EnKPF)
Rather than choosing between the Kalman filter (which scales perfectly to high dimensions but fails on non-Gaussian data) and the Particle Filter (which handles non-Gaussian data but collapses in high dimensions), researchers have bridged the two. The modern **EnKPF** utilizes an Ensemble Kalman step as an incredibly highly-tuned proposal distribution to shift the particles, followed by a localized PF resampling step to correct for non-Gaussian deformations (Frei & Künsch, 2013).

### 3. Differentiable Particle Filters (DPFs)
The current bleeding edge of the field replaces explicitly coded, Taylor-expanded mathematical physics models with neural networks. By rewriting the particle filter's recursive structure in differentiable frameworks like PyTorch or JAX, the entire algorithm can be backpropagated. This allows an AI to explicitly *learn* the optimal system dynamics $F$, the observation model $H$, and the sensor noise profiles entirely from historical L0 data, utilizing the particle filter purely as an algorithmic prior (Jonschkowski et al., 2018).

### References
1. Snyder, C., Bengtsson, T., Bickel, P., & Anderson, J. (2008). *Obstacles to High-Dimensional Particle Filtering*. Monthly Weather Review, 136(12), 4629-4640.
2. Frei, M., & Künsch, H. R. (2013). *Bridging the ensemble Kalman and particle filters*. Biometrika, 100(4), 781-800.
3. Jonschkowski, R., Rastogi, D., & Brock, O. (2018). *Differentiable Particle Filters: End-to-End Learning with Algorithmic Priors*. Robotics: Science and Systems XIV.
