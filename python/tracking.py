"""
Optimal Estimation and Particle Filtering Data Processing Pipeline

This module implements a complete L0 to L1 data processing pipeline for a 
synthetic 2D tracking problem, alongside a demonstration of Brownian Bridge 
path sampling.

Processing Levels:
    L0 (Raw Data): Generation of synthetic ground truth state vectors and the 
                   associated noisy sensor measurements. 
    L1 (Processed Telemetry): Execution of Bayesian tracking filters (Kalman, SIR, 
                              ASIR, CIR) to estimate the hidden state and 
                              compute covariance bounds.

System Assumptions & Mathematics:
    We approximate a continuous-time stochastic process using a discrete-time sequence. 
    In this generalized state-space framework, the state transition evaluates how the 
    system evolves over time, denoted as X_k = F * X_{k-1} + v_{k-1}, where v_{k-1} 
    represents multivariate Gaussian process noise. The imperfect observation model is 
    defined as Z_k = H * X_k + n_k, where n_k represents the sensor observation noise.

References:
    1. Kalman, R. (1960). "A new approach to linear filtering and prediction problems." 
    2. Gordon, N., Salmond, D., & Smith, A. (1993). "Novel approach to nonlinear/non-Gaussian 
       Bayesian state estimation."
    3. Chorin, A. J., & Tu, X. (2009). "Implicit sampling for particle filters."
    4. Weare, J. (2009). "Particle filtering with path sampling and an application to a bimodal 
       ocean current model."
"""

import logging
import argparse
from abc import ABC, abstractmethod
from typing import Any, Type, Tuple

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
from pydantic import BaseModel, ConfigDict, Field

# Define a type alias for standard float arrays
FloatArray = npt.NDArray[np.float64]

# Configure professional logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# CONFIGURATION & PHYSICS MODEL
# ---------------------------------------------------------
class StateSpaceModel(BaseModel):
    """
    Pydantic data model representing the physical state-space system and sensor configuration.
    
    Mathematical Formulation:
        State Equation: X_k = F * X_{k-1} + v_{k-1} 
        Observation Equation: Z_k = H * X_k + n_k
        
    Attributes:
        state_transition (FloatArray): The matrix (F) defining how the state evolves.
        observation_model (FloatArray): The matrix (H) mapping the true state to observations.
        process_covariance (FloatArray): The matrix (Q) defining the process noise covariance.
        sensor_covariance (FloatArray): The matrix (R) defining the sensor noise covariance.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    state_transition: FloatArray = Field(..., description="System state transition matrix (F)")
    observation_model: FloatArray = Field(..., description="Sensor observation matrix (H)")
    process_covariance: FloatArray = Field(..., description="Process noise covariance matrix (Q)")
    sensor_covariance: FloatArray = Field(..., description="Sensor noise covariance matrix (R)")

    @property
    def state_dim(self) -> int:
        return self.state_transition.shape[0]

    @property
    def obs_dim(self) -> int:
        return self.observation_model.shape[0]


# ---------------------------------------------------------
# L0 DATA GENERATION
# ---------------------------------------------------------
class TargetSimulation:
    """Generates synthetic ground truth and telemetry using an injected StateSpaceModel."""
    
    def __init__(self, model: StateSpaceModel, n_steps: int = 100) -> None:
        self.model = model
        self.n_steps = n_steps

    def generate_telemetry(self, initial_state: FloatArray) -> Tuple[FloatArray, FloatArray]:
        """
        Generates the continuous true states and the corresponding sensor measurements.
        
        Algorithm:
            1. Initialize X_0 = initial_state
            2. For each step k from 1 to n_steps:
               a. Sample v_k ~ MultivariateNormal(0, Q)
               b. Compute X_k = F * X_{k-1} + v_k
               c. Sample n_k ~ MultivariateNormal(0, R)
               d. Compute Z_k = H * X_k + n_k
        """
        true_states = np.zeros((self.model.state_dim, self.n_steps))
        measurements = np.zeros((self.model.obs_dim, self.n_steps))
        
        true_states[:, 0] = initial_state
        
        for step in range(self.n_steps):
            if step > 0:
                process_noise = np.random.multivariate_normal(
                    np.zeros(self.model.state_dim), self.model.process_covariance
                )
                true_states[:, step] = self.model.state_transition @ true_states[:, step-1] + process_noise
            
            sensor_noise = np.random.multivariate_normal(
                np.zeros(self.model.obs_dim), self.model.sensor_covariance
            )
            measurements[:, step] = self.model.observation_model @ true_states[:, step] + sensor_noise
            
        return true_states, measurements


# ---------------------------------------------------------
# L1 PROCESSING ARCHITECTURE (TRACKING FILTERS)
# ---------------------------------------------------------
class BaseTrackingFilter(ABC):
    """Abstract base class enforcing the execution contract for all L1 tracking filters."""
    
    def __init__(self, model: StateSpaceModel, **kwargs: Any) -> None:
        self.model = model

    @abstractmethod
    def run_filter(self, measurements: FloatArray, initial_state: FloatArray) -> Tuple[FloatArray, FloatArray]:
        pass


class KalmanFilter(BaseTrackingFilter):
    """
    Standard linear Kalman Filter optimal estimator.
    
    For systems characterized by strictly linear transitions and Gaussian noise profiles, 
    the Kalman filter provides an optimal, closed-form recursive solution to estimate 
    the internal state sequence.
    """
    
    def run_filter(self, measurements: FloatArray, initial_state: FloatArray) -> Tuple[FloatArray, FloatArray]:
        """
        Runs the standard Predict-Update Kalman loop over the measurement series.
        
        Algorithm:
            1. Prediction Step (Time Update):
               Project the state ahead:     X_{k|k-1} = F * X_{k-1|k-1}
               Project the error covar:     P_{k|k-1} = F * P_{k-1|k-1} * F^T + Q
               
            2. Update Step (Measurement Update):
               Compute measurement residual: y_k = Z_k - H * X_{k|k-1}
               Compute residual covar:       S_k = H * P_{k|k-1} * H^T + R
               Compute Kalman Gain:          K_k = P_{k|k-1} * H^T * S_k^{-1}
               
               Update state estimate:        X_{k|k} = X_{k|k-1} + K_k * y_k
               Update error covar:           P_{k|k} = (I - K_k * H) * P_{k|k-1}
        """
        n_steps = measurements.shape[1]
        estimated_states = np.zeros((self.model.state_dim, n_steps))
        estimate_covariances = np.zeros((self.model.state_dim, self.model.state_dim, n_steps))
        
        estimated_states[:, 0] = initial_state
        estimate_covariances[:, :, 0] = self.model.process_covariance 
        identity = np.eye(self.model.state_dim)
        
        for step in range(1, n_steps):
            # Predict
            predicted_state = self.model.state_transition @ estimated_states[:, step-1]
            predicted_cov = (self.model.state_transition @ estimate_covariances[:, :, step-1] 
                             @ self.model.state_transition.T + self.model.process_covariance)
            
            # Update
            residual = measurements[:, step] - self.model.observation_model @ predicted_state 
            residual_cov = (self.model.observation_model @ predicted_cov 
                            @ self.model.observation_model.T + self.model.sensor_covariance) 
            
            # Use np.linalg.solve for numerical stability instead of matrix inversion
            # K = P * H^T * S^-1  ==> K^T = S^-1 * H * P ==> S * K^T = H * P
            kalman_gain = np.linalg.solve(residual_cov, self.model.observation_model @ predicted_cov).T
            
            estimated_states[:, step] = predicted_state + kalman_gain @ residual
            estimate_covariances[:, :, step] = (identity - kalman_gain @ self.model.observation_model) @ predicted_cov
            estimate_covariances[:, :, step] = (estimate_covariances[:, :, step] + estimate_covariances[:, :, step].T) / 2.0
            
        return estimated_states, estimate_covariances


class BaseParticleFilter(BaseTrackingFilter):
    """
    Intermediate class holding shared Monte Carlo logic such as systematic resampling.
    
    Particle filtering is a versatile recursive estimation method. These filters 
    approximate the hidden states of a dynamic system by simulating numerous independent 
    trajectories, or particles. Every simulated trajectory is assigned a likelihood 
    weight reflecting its probability given the observed data. To prevent weight 
    degeneracy—where a single trajectory dominates the ensemble—a systematic resampling 
    step is applied periodically.
    """
    
    def __init__(self, model: StateSpaceModel, n_particles: int = 100, **kwargs: Any) -> None:
        super().__init__(model)
        self.n_particles = n_particles

    def _resample_indices(self, weights: FloatArray) -> npt.NDArray[np.int_]:
        """
        Calculates systematic resampling indices to resolve particle weight degeneracy.
        
        Algorithm (Systematic Resampling):
            To resample, we draw a uniform random variable and step through the cumulative 
            density function of the weights, replicating particles proportional to their 
            likelihood mass.
        """
        indices = np.zeros(self.n_particles, dtype=int)
        cumulative_weights = np.cumsum(weights)
        uniform_samples = np.zeros(self.n_particles)
        uniform_samples[0] = np.random.uniform(0, 1 / self.n_particles)
        
        index = 0
        for step in range(self.n_particles):
            uniform_samples[step] = uniform_samples[0] + step / self.n_particles
            while index < self.n_particles - 1 and uniform_samples[step] > cumulative_weights[index]:
                index += 1
            indices[step] = index
            
        return indices


class SIRParticleFilter(BaseParticleFilter):
    """Sampling Importance Resampling (SIR) particle filter implementation."""
    
    def run_filter(self, measurements: FloatArray, initial_state: FloatArray) -> Tuple[FloatArray, FloatArray]:
        """
        Algorithm:
            1. Prior Sample (Predict): 
               X_k^(i) = F * X_{k-1}^(i) + v_k, where v_k ~ N(0, Q)
            2. Weight Update (Likelihood):
               w_k^(i) = N(Z_k ; H * X_k^(i), R)
            3. Normalize:
               w_k^(i) = w_k^(i) / SUM(w_k)
            4. Resample
        """
        n_steps = measurements.shape[1]
        particles = np.zeros((self.model.state_dim, self.n_particles, n_steps))
        weights = np.zeros((self.n_particles, n_steps))
        
        particles[:, :, 0] = np.tile(initial_state, (self.n_particles, 1)).T
        weights[:, 0] = 1.0 / self.n_particles
        
        for step in range(1, n_steps):
            for i in range(self.n_particles):
                process_noise = np.random.multivariate_normal(
                    np.zeros(self.model.state_dim), self.model.process_covariance
                )
                particles[:, i, step] = self.model.state_transition @ particles[:, i, step-1] + process_noise
                
                expected_measurement = self.model.observation_model @ particles[:, i, step]
                weights[i, step] = multivariate_normal.pdf(
                    measurements[:, step], mean=expected_measurement, cov=self.model.sensor_covariance
                )
            
            weights[:, step] /= np.sum(weights[:, step]) 
            resampled_idx = self._resample_indices(weights[:, step])
            particles[:, :, step] = particles[:, resampled_idx, step]
            weights[:, step] = 1.0 / self.n_particles
            
        estimated_states = np.mean(particles, axis=1)
        estimate_covariances = np.zeros((self.model.state_dim, self.model.state_dim, n_steps))
        for step in range(n_steps):
            estimate_covariances[:, :, step] = np.cov(particles[:, :, step])
            
        return estimated_states, estimate_covariances


class ASIRParticleFilter(BaseParticleFilter):
    """
    Auxiliary Sampling Importance Resampling (ASIR) particle filter implementation.
    
    The Auxiliary SIR algorithm enhances the standard SIR approach by leveraging the 
    upcoming measurement during the resampling phase, before the particles are fully 
    propagated forward in time. This pre-selection drastically improves efficiency.
    """

    def run_filter(self, measurements: FloatArray, initial_state: FloatArray) -> Tuple[FloatArray, FloatArray]:
        """
        Algorithm:
            Stage 1: Look-ahead Prediction & First-Stage Weights
                mu_k^(i) = E[X_k | X_{k-1}^(i)] = F * X_{k-1}^(i)
                lambda_k^(i) proportional to w_{k-1}^(i) * p(Z_k | mu_k^(i))
            Stage 2: Auxiliary Resampling
                Resample indices based on the auxiliary weights lambda_k.
            Stage 3: Particle Propagation
                X_k^(j) = F * X_{k-1}^(parent_j) + v_k
            Stage 4: Final Weight Correction
                w_k^(j) proportional to p(Z_k | X_k^(j)) / p(Z_k | mu_k^(parent_j))
        """
        n_steps = measurements.shape[1]
        particles = np.zeros((self.model.state_dim, self.n_particles, n_steps))
        weights = np.zeros((self.n_particles, n_steps))

        particles[:, :, 0] = np.tile(initial_state, (self.n_particles, 1)).T
        weights[:, 0] = 1.0 / self.n_particles

        for step in range(1, n_steps):
            mu_k = np.zeros((self.model.state_dim, self.n_particles))
            aux_weights = np.zeros(self.n_particles)

            for i in range(self.n_particles):
                mu_k[:, i] = self.model.state_transition @ particles[:, i, step-1]
                expected_meas = self.model.observation_model @ mu_k[:, i]
                likelihood = multivariate_normal.pdf(
                    measurements[:, step], mean=expected_meas, cov=self.model.sensor_covariance
                )
                aux_weights[i] = likelihood * weights[i, step-1]

            aux_weights /= np.sum(aux_weights)
            resampled_indices = self._resample_indices(aux_weights)

            for j, parent_idx in enumerate(resampled_indices):
                process_noise = np.random.multivariate_normal(
                    np.zeros(self.model.state_dim), self.model.process_covariance
                )
                particles[:, j, step] = self.model.state_transition @ particles[:, parent_idx, step-1] + process_noise

                actual_expected = self.model.observation_model @ particles[:, j, step]
                actual_likelihood = multivariate_normal.pdf(
                    measurements[:, step], mean=actual_expected, cov=self.model.sensor_covariance
                )

                mu_expected = self.model.observation_model @ mu_k[:, parent_idx]
                mu_likelihood = multivariate_normal.pdf(
                    measurements[:, step], mean=mu_expected, cov=self.model.sensor_covariance
                )

                weights[j, step] = actual_likelihood / mu_likelihood

            weights[:, step] /= np.sum(weights[:, step])

        estimated_states = np.zeros((self.model.state_dim, n_steps))
        estimate_covariances = np.zeros((self.model.state_dim, self.model.state_dim, n_steps))
        
        for step in range(n_steps):
            estimated_states[:, step] = np.average(particles[:, :, step], axis=1, weights=weights[:, step])
            diff = particles[:, :, step] - estimated_states[:, step].reshape(-1, 1)
            estimate_covariances[:, :, step] = (weights[:, step] * diff) @ diff.T

        return estimated_states, estimate_covariances


class CIRParticleFilter(BaseParticleFilter):
    """Continuous/Implicit Resampling (CIR) Particle Filter implementation."""

    def run_filter(self, measurements: FloatArray, initial_state: FloatArray) -> Tuple[FloatArray, FloatArray]:
        n_steps = measurements.shape[1]
        particles = np.zeros((self.model.state_dim, self.n_particles, n_steps))
        weights = np.zeros((self.n_particles, n_steps))
        
        particles[:, :, 0] = np.tile(initial_state, (self.n_particles, 1)).T
        weights[:, 0] = 1.0 / self.n_particles

        # Precompute constants for numerical stability
        q_inv = np.linalg.pinv(self.model.process_covariance)
        r_inv = np.linalg.inv(self.model.sensor_covariance)
        hessian = q_inv + self.model.observation_model.T @ r_inv @ self.model.observation_model
        posterior_cov = np.linalg.pinv(hessian)

        for step in range(1, n_steps):
            current_measurement = measurements[:, step]

            for i in range(self.n_particles):
                prev_state = particles[:, i, step-1]

                def objective_func(x: FloatArray) -> Tuple[float, FloatArray]:
                    state_diff = x - self.model.state_transition @ prev_state
                    obs_diff = current_measurement - self.model.observation_model @ x
                    val = 0.5 * (state_diff.T @ q_inv @ state_diff) + 0.5 * (obs_diff.T @ r_inv @ obs_diff)
                    grad = q_inv @ state_diff - self.model.observation_model.T @ r_inv @ obs_diff
                    return val, grad

                opt_result = minimize(objective_func, x0=prev_state, jac=True, method='BFGS')
                mode_state = opt_result.x

                particles[:, i, step] = np.random.multivariate_normal(mode_state, posterior_cov)
                
                expected_z = self.model.observation_model @ self.model.state_transition @ prev_state
                innov_cov = self.model.observation_model @ self.model.process_covariance @ self.model.observation_model.T + self.model.sensor_covariance
                weights[i, step] = weights[i, step-1] * multivariate_normal.pdf(
                    current_measurement, mean=expected_z, cov=innov_cov
                )
            
            weights[:, step] /= np.sum(weights[:, step]) 
            resampled_idx = self._resample_indices(weights[:, step])
            particles[:, :, step] = particles[:, resampled_idx, step]
            weights[:, step] = 1.0 / self.n_particles
            
        estimated_states = np.mean(particles, axis=1)
        estimate_covariances = np.zeros((self.model.state_dim, self.model.state_dim, n_steps))
        for step in range(n_steps):
            estimate_covariances[:, :, step] = np.cov(particles[:, :, step])
            
        return estimated_states, estimate_covariances

class FourDVarSmoother(BaseTrackingFilter):
    """
    Sliding-Window 4D-Variational (4DVar) Smoother.

    Assimilates batches of observations over a finite time window. This allows
    the smoother to find optimal local trajectories while still adapting to
    unmodeled process noise between windows.
    """

    def run_filter(self, measurements: FloatArray, initial_state: FloatArray) -> Tuple[FloatArray, FloatArray]:
        n_steps = measurements.shape[1]
        window_size = 15  # Assimilation window size (tune this to see the smoothing effect)
        
        # 1. Relax the background prior drastically. 
        # If we use the single-step Q matrix, the optimizer gets tethered to the prior.
        # We use a broad diagonal matrix so we trust the observations more than the prior.
        b_inv = np.eye(self.model.state_dim) * 1e-4
        r_inv = np.linalg.inv(self.model.sensor_covariance)
        
        estimated_states = np.zeros((self.model.state_dim, n_steps))
        estimate_covariances = np.zeros((self.model.state_dim, self.model.state_dim, n_steps))
        
        # The background guess for the start of the first window
        current_x_b = initial_state
        
        for start_idx in range(0, n_steps, window_size):
            end_idx = min(start_idx + window_size, n_steps)
            current_w_size = end_idx - start_idx
            
            def cost_function(x0: FloatArray) -> Tuple[float, FloatArray]:
                """4DVar Cost function for the current sliding window."""
                diff_b = x0 - current_x_b
                j_cost = 0.5 * (diff_b.T @ b_inv @ diff_b)
                grad = b_inv @ diff_b
                
                x_k = x0
                for k in range(current_w_size):
                    if k > 0:
                        x_k = self.model.state_transition @ x_k
                    
                    f_k = np.linalg.matrix_power(self.model.state_transition, k)
                    obs_diff = measurements[:, start_idx + k] - self.model.observation_model @ x_k
                    
                    j_cost += 0.5 * (obs_diff.T @ r_inv @ obs_diff)
                    grad -= (self.model.observation_model @ f_k).T @ r_inv @ obs_diff
                    
                return float(j_cost), grad

            # Optimize the initial state for THIS specific window
            opt_result = minimize(cost_function, x0=current_x_b, jac=True, method='BFGS')
            optimized_x0 = opt_result.x
            
            # Compute Analysis Covariance for this window
            hessian = b_inv.copy()
            for k in range(current_w_size):
                f_k = np.linalg.matrix_power(self.model.state_transition, k)
                h_f_k = self.model.observation_model @ f_k
                hessian += h_f_k.T @ r_inv @ h_f_k
            
            window_cov = np.linalg.pinv(hessian)
            
            # Forward propagate the optimal trajectory through the window to save the states
            x_k = optimized_x0
            cov_k = window_cov
            for k in range(current_w_size):
                if k > 0:
                    x_k = self.model.state_transition @ x_k
                    # Strong constraint inside the window (no Q added)
                    cov_k = self.model.state_transition @ cov_k @ self.model.state_transition.T
                
                estimated_states[:, start_idx + k] = x_k
                estimate_covariances[:, :, start_idx + k] = cov_k
            
            # Set the background guess for the NEXT window to the final state of this window
            if end_idx < n_steps:
                current_x_b = self.model.state_transition @ x_k
                
        return estimated_states, estimate_covariances

class WeakConstraint4DVarSmoother(BaseTrackingFilter):
    """
    Weak-Constraint 4DVar Smoother (Forcing Estimation).

    Instead of assuming a perfect physics model, this algorithm reconstructs 
    BOTH the optimal initial state and the sequence of unknown process noise 
    (random accelerations) over the assimilation window using the Adjoint method.
    """

    def run_filter(self, measurements: FloatArray, initial_state: FloatArray) -> Tuple[FloatArray, FloatArray]:
        n_steps = measurements.shape[1]
        window_size = 15  
        
        # Inverses for the cost function penalties
        b_inv = np.eye(self.model.state_dim) * 1e-4
        r_inv = np.linalg.inv(self.model.sensor_covariance)

        # Regularize Q to make it full-rank. This closes the null-space loophole
        # and heavily penalizes non-physical "teleportation" of the position states.
        regularized_q = self.model.process_covariance + np.eye(self.model.state_dim) * 1e-6
        q_inv = np.linalg.inv(regularized_q)
        
        estimated_states = np.zeros((self.model.state_dim, n_steps))
        estimate_covariances = np.zeros((self.model.state_dim, self.model.state_dim, n_steps))
        
        current_x_b = initial_state
        
        for start_idx in range(0, n_steps, window_size):
            end_idx = min(start_idx + window_size, n_steps)
            w_steps = end_idx - start_idx
            
            # The control vector 'u' contains X_0 AND all the process noise vectors (w_k)
            ctrl_dim = self.model.state_dim + (w_steps - 1) * self.model.state_dim
            u0 = np.zeros(ctrl_dim)
            u0[0:self.model.state_dim] = current_x_b
            
            def cost_and_grad(u: FloatArray) -> Tuple[float, FloatArray]:
                x0 = u[0:self.model.state_dim]
                
                # --- 1. Forward Pass ---
                X_traj = np.zeros((self.model.state_dim, w_steps))
                X_traj[:, 0] = x0
                
                for k in range(w_steps - 1):
                    w_k = u[self.model.state_dim * (k + 1) : self.model.state_dim * (k + 2)]
                    X_traj[:, k+1] = self.model.state_transition @ X_traj[:, k] + w_k
                    
                # --- 2. Compute Cost J ---
                diff_b = x0 - current_x_b
                j_cost = 0.5 * (diff_b.T @ b_inv @ diff_b)
                
                for k in range(w_steps - 1):
                    w_k = u[self.model.state_dim * (k + 1) : self.model.state_dim * (k + 2)]
                    j_cost += 0.5 * (w_k.T @ q_inv @ w_k)
                    
                for k in range(w_steps):
                    obs_diff = measurements[:, start_idx + k] - self.model.observation_model @ X_traj[:, k]
                    j_cost += 0.5 * (obs_diff.T @ r_inv @ obs_diff)
                    
                # --- 3. Backward Adjoint Pass (Exact Gradients) ---
                grad_u = np.zeros_like(u)
                adj_lambda = np.zeros(self.model.state_dim)
                
                # Sweep backwards to calculate the sensitivities
                for k in range(w_steps - 1, -1, -1):
                    obs_residual = self.model.observation_model @ X_traj[:, k] - measurements[:, start_idx + k]
                    
                    # Update the adjoint variable (lambda)
                    adj_lambda = self.model.observation_model.T @ r_inv @ obs_residual + self.model.state_transition.T @ adj_lambda
                    
                    if k > 0:
                        # Gradient w.r.t the process noise at step k-1
                        w_k_minus_1 = u[self.model.state_dim * k : self.model.state_dim * (k + 1)]
                        grad_u[self.model.state_dim * k : self.model.state_dim * (k + 1)] = q_inv @ w_k_minus_1 + adj_lambda
                    else:
                        # Gradient w.r.t the initial state X_0
                        grad_u[0:self.model.state_dim] = b_inv @ diff_b + adj_lambda
                        
                return float(j_cost), grad_u

            # Optimize the entire sequence simultaneously
            opt_result = minimize(cost_and_grad, x0=u0, jac=True, method='BFGS')
            opt_u = opt_result.x
            
            # Extract the final optimized trajectory
            opt_x_traj = np.zeros((self.model.state_dim, w_steps))
            opt_x_traj[:, 0] = opt_u[0:self.model.state_dim]
            for k in range(w_steps - 1):
                w_k = opt_u[self.model.state_dim * (k + 1) : self.model.state_dim * (k + 2)]
                opt_x_traj[:, k+1] = self.model.state_transition @ opt_x_traj[:, k] + w_k
            
            estimated_states[:, start_idx:end_idx] = opt_x_traj
            
            # Note: In Weak-Constraint 4DVar, calculating the exact posterior covariance requires 
            # inverting the massive Hessian of the expanded control space. For visual tracking, 
            # we inject the baseline process covariance to keep the visualization loop happy.
            for k in range(w_steps):
                estimate_covariances[:, :, start_idx + k] = self.model.process_covariance
            
            if end_idx < n_steps:
                current_x_b = opt_x_traj[:, -1]
                
        return estimated_states, estimate_covariances

class RTSSmoother(BaseTrackingFilter):
    """
    Rauch-Tung-Striebel (RTS) Smoother.
    
    A two-pass (forward-backward) optimal smoother. It acts as a physics-informed 
    spline, providing a perfectly continuous trajectory that accounts for process noise
    at every single timestep, eliminating the rigid segments of Strong-Constraint 4DVar.
    """
    def run_filter(self, measurements: FloatArray, initial_state: FloatArray) -> Tuple[FloatArray, FloatArray]:
        n_steps = measurements.shape[1]
        
        # Pass 1: Forward Kalman Filter
        kf = KalmanFilter(self.model)
        kf_states, kf_covs = kf.run_filter(measurements, initial_state)
        
        # Pass 2: Backward Smoothing Pass
        smoothed_states = np.zeros_like(kf_states)
        smoothed_covs = np.zeros_like(kf_covs)
        
        # The ultimate state is exactly the forward filter's estimate
        smoothed_states[:, -1] = kf_states[:, -1]
        smoothed_covs[:, :, -1] = kf_covs[:, :, -1]
        
        for k in range(n_steps - 2, -1, -1):
            # Predict forward from step k
            pred_state = self.model.state_transition @ kf_states[:, k]
            pred_cov = (self.model.state_transition @ kf_covs[:, :, k] 
                        @ self.model.state_transition.T + self.model.process_covariance)
            
            # Calculate Smoother Gain (C_k)
            # Use np.linalg.solve for stability instead of direct inversion
            smoother_gain = np.linalg.solve(pred_cov, self.model.state_transition @ kf_covs[:, :, k]).T
            
            # Smooth the state and covariance
            smoothed_states[:, k] = kf_states[:, k] + smoother_gain @ (smoothed_states[:, k+1] - pred_state)
            
            # Force symmetry on the smoothed covariance update to prevent plotting crashes
            cov_update = kf_covs[:, :, k] + smoother_gain @ (smoothed_covs[:, :, k+1] - pred_cov) @ smoother_gain.T
            smoothed_covs[:, :, k] = (cov_update + cov_update.T) / 2.0
            
        return smoothed_states, smoothed_covs

class AlphaBetaGammaFilter(BaseTrackingFilter):
    """
    Alpha-Beta-Gamma Filter.
    
    A steady-state, fixed-gain filter. While computationally cheaper and theoretically
    less optimal than a dynamic Kalman filter, it explicitly tracks acceleration 
    (the gamma term), allowing it to follow parabolic maneuvers that a 
    Constant-Velocity model cannot.
    """
    def run_filter(self, measurements: FloatArray, initial_state: FloatArray) -> Tuple[FloatArray, FloatArray]:
        n_steps = measurements.shape[1]
        dt = 0.05  # Standardized time step from your simulation

        # ABG requires tracking acceleration, so we internally use a 6D state:
        # [x, y, vx, vy, ax, ay]
        internal_states = np.zeros((6, n_steps))
        internal_states[0:4, 0] = initial_state
        
        # Heuristic tuning parameters (These dictate the "stiffness" of the filter)
        alpha = 0.1     # Position trust
        beta = 0.005    # Velocity trust
        gamma = 0.0001  # Acceleration trust
        
        for step in range(1, n_steps):
            # 1. Predict Forward (Kinematic Equations)
            pred_x = internal_states[0, step-1] + internal_states[2, step-1]*dt + 0.5*internal_states[4, step-1]*(dt**2)
            pred_y = internal_states[1, step-1] + internal_states[3, step-1]*dt + 0.5*internal_states[5, step-1]*(dt**2)
            
            pred_vx = internal_states[2, step-1] + internal_states[4, step-1]*dt
            pred_vy = internal_states[3, step-1] + internal_states[5, step-1]*dt
            
            pred_ax = internal_states[4, step-1]
            pred_ay = internal_states[5, step-1]
            
            # 2. Calculate Residual (Measurement - Prediction)
            res_x = measurements[0, step] - pred_x
            res_y = measurements[1, step] - pred_y
            
            # 3. Update States using fixed gains
            internal_states[0, step] = pred_x + alpha * res_x
            internal_states[1, step] = pred_y + alpha * res_y
            
            internal_states[2, step] = pred_vx + (beta / dt) * res_x
            internal_states[3, step] = pred_vy + (beta / dt) * res_y
            
            # Different texts scale gamma differently; 2*gamma/dt^2 is standard for stability
            internal_states[4, step] = pred_ax + (2 * gamma / (dt**2)) * res_x
            internal_states[5, step] = pred_ay + (2 * gamma / (dt**2)) * res_y

        # Slice the output to match the 4D [x, y, vx, vy] expectation of your visualizer
        estimated_states = internal_states[0:4, :]
        
        # ABG does not calculate uncertainty covariance. We generate static 
        # placeholder matrices so the plotting function's ellipse math doesn't crash.
        estimate_covariances = np.zeros((self.model.state_dim, self.model.state_dim, n_steps))
        for i in range(n_steps):
            estimate_covariances[:, :, i] = self.model.process_covariance * 2.0

        return estimated_states, estimate_covariances

class FilterFactory:
    """Registry-based Factory pattern for dynamically instantiating specific tracking filters."""
    
    _registry: dict[str, Type[BaseTrackingFilter]] = {
        'kalman': KalmanFilter,
        'sir': SIRParticleFilter,
        'asir': ASIRParticleFilter,
        'cir': CIRParticleFilter,
        '4dvar': FourDVarSmoother,
        'wc4dvar': WeakConstraint4DVarSmoother,
        'rts': RTSSmoother,
        'abg': AlphaBetaGammaFilter
    }

    @classmethod
    def create(cls, filter_type: str, model: StateSpaceModel, **kwargs: Any) -> BaseTrackingFilter:
        f_type = filter_type.lower()
        if f_type not in cls._registry:
            raise ValueError(f"Filter '{f_type}' not found. Options: {list(cls._registry.keys())}")
        return cls._registry[f_type](model, **kwargs)


# ---------------------------------------------------------
# STANDALONE SDE SOLVERS
# ---------------------------------------------------------
class BrownianBridgeSampler:
    """Object-oriented implementation of Chorin's Implicit Sampling for a Brownian Bridge."""
    
    def __init__(self, n_steps: int, n_particles: int, variance: float) -> None:
        self.n_steps = n_steps
        self.n_particles = n_particles
        self.variance = variance
        self.dt = 1.0 / n_steps

    def _path_objective(self, inner_x: FloatArray, target_x: float) -> Tuple[float, FloatArray]:
        full_path = np.concatenate(([0.0], inner_x, [target_x]))
        diffs = np.diff(full_path)
        energy = np.sum((diffs**2) / (2 * self.variance * self.dt))
        
        grad = np.zeros_like(inner_x)
        for i in range(len(inner_x)):
            grad[i] = (2 * inner_x[i] - full_path[i] - full_path[i+2]) / (self.variance * self.dt)
            
        return float(energy), grad

    def sample_ensemble(self, target_x: float) -> Tuple[FloatArray, FloatArray]:
        paths = np.zeros((self.n_particles, self.n_steps))
        weights = np.ones(self.n_particles) / self.n_particles
        paths[:, -1] = target_x 

        for i in range(self.n_particles):
            initial_guess = np.linspace(0, target_x, self.n_steps)[1:-1]
            opt_result = minimize(self._path_objective, x0=initial_guess, args=(target_x,), jac=True, method='BFGS')
            paths[i, 1:-1] = opt_result.x
            
        return paths, weights


# ---------------------------------------------------------
# VISUALIZATION
# ---------------------------------------------------------
class FilterVisualizer:
    
    def __init__(self, figsize: Tuple[int, int] = (8, 8), pause_time: float = 0.01, frame_skip: int = 2) -> None:
        self.figsize = figsize
        self.pause_time = pause_time
        self.frame_skip = max(1, frame_skip)
        self.fig: plt.Figure | None = None
        self.ax: plt.Axes | None = None
        self.ellipses: list[Ellipse] = []

    def _setup_plot(self, true_states: FloatArray, title: str) -> None:
        plt.ion() 
        # Explicitly create a new, distinct window using the title
        self.fig = plt.figure(title, figsize=self.figsize)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim(np.min(true_states[0, :]) - 0.5, np.max(true_states[0, :]) + 0.5)
        self.ax.set_ylim(np.min(true_states[1, :]) - 0.5, np.max(true_states[1, :]) + 0.5)
        self.ax.set_title(title)
        self.ax.grid(True)

    def animate_comet_covar(self, true_states: FloatArray, measurements: FloatArray, 
                            estimated_states: FloatArray, estimate_covariances: FloatArray, title: str) -> None:
        self._setup_plot(true_states, title)
        if self.ax is None:
            return
            
        truth_line, = self.ax.plot([], [], 'g-', label='True State', linewidth=2)
        obs_scatter = self.ax.scatter([], [], c='gray', marker='x', alpha=0.5, label='Observations')
        est_line, = self.ax.plot([], [], 'b--', label='Filter Estimate', linewidth=2)
        self.ax.legend()
        
        for step in range(1, true_states.shape[1], self.frame_skip): 
            truth_line.set_data(true_states[0, :step], true_states[1, :step])
            obs_scatter.set_offsets(measurements[:, :step].T)
            est_line.set_data(estimated_states[0, :step], estimated_states[1, :step])
            
            for ell in self.ellipses: 
                ell.remove()
            self.ellipses.clear()
            
            step_covariance = estimate_covariances[0:2, 0:2, step]
            eigenvalues, eigenvectors = np.linalg.eigh(step_covariance)
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            
            for n_std in [1, 2, 3]:
                width = 2 * n_std * np.sqrt(eigenvalues[0])
                height = 2 * n_std * np.sqrt(eigenvalues[1])
                ell_patch = Ellipse(
                    xy=(estimated_states[0, step], estimated_states[1, step]), 
                    width=width, height=height, angle=angle, 
                    edgecolor='blue', facecolor='none', alpha=0.4 / n_std
                )
                self.ax.add_patch(ell_patch)
                self.ellipses.append(ell_patch)
                
            plt.pause(self.pause_time)
            
        plt.ioff()
        plt.show(block=False)
        # plt.pause(1.5)
        # plt.close()


# ---------------------------------------------------------
# EXECUTION PIPELINE / CLI
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Optimal Estimation and Particle Filtering Data Processing Pipeline")
    parser.add_argument("--filter", type=str, default="all", choices=['all', 'kalman', 'sir', 'asir', 'cir', 'abg', 
                                                                      '4dvar', 'wc4dvar', 'rts'], 
                        help="Specify which filter algorithm to execute.")
    parser.add_argument("--particles", type=int, default=50, 
                        help="Number of particles for Monte Carlo ensemble filters.")
    parser.add_argument("--steps", type=int, default=400, 
                        help="Number of simulation timesteps.")
    parser.add_argument("--frame-skip", type=int, default=2, 
                        help="Animation frame skip interval for rendering speed.")
    parser.add_argument("--pause-factor", type=float, default=3.0,
                        help="Multiplier for the animation pause time (e.g., 5 = 0.05s pause).")
    parser.add_argument("--run-bridge", action="store_true", 
                        help="Execute the standalone Brownian Bridge SDE solver.")
    args = parser.parse_args()

    logger.info("Defining System Physics and Sensor Configuration Data Model...")
    time_step = 0.05
    obs_variance = 0.05
    dt2 = 0.5 * time_step**2

    model_config = StateSpaceModel(
        state_transition=np.array([
            [1.0, 0.0, time_step, 0.0],
            [0.0, 1.0, 0.0,       time_step],
            [0.0, 0.0, 1.0,       0.0],
            [0.0, 0.0, 0.0,       1.0]
        ]),
        observation_model=np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ]),
        process_covariance=(
            np.array([[dt2, 0.0], [0.0, dt2], [time_step, 0.0], [0.0, time_step]]) @ 
            np.array([[0.5, 0.01], [0.01, 0.5]]) @ 
            np.array([[dt2, 0.0], [0.0, dt2], [time_step, 0.0], [0.0, time_step]]).T
        ),
        sensor_covariance=np.array([
            [obs_variance, 0.0], 
            [0.0,          obs_variance]
        ])
    )
    
    logger.info("Generating L0 Target Telemetry Data...")
    sim = TargetSimulation(model=model_config, n_steps=args.steps)
    initial_state = np.zeros(model_config.state_dim)
    true_states, measurements = sim.generate_telemetry(initial_state)
    
    calc_pause = args.pause_factor * 0.01 
    vis = FilterVisualizer(frame_skip=args.frame_skip, pause_time=calc_pause)

    # Determine which filters to run
    # filters_to_run = ['kalman', 'sir', 'asir', '4dvar', 'rts'] if args.filter == 'all' else [args.filter]
    filters_to_run = ['abg'] if args.filter == 'all' else [args.filter]

    for f_type in filters_to_run:
        logger.info(f"Instantiating and executing the {f_type.upper()} Filter...")
        tracker = FilterFactory.create(f_type, model=model_config, n_particles=args.particles)
        estimated_states, estimate_covariances = tracker.run_filter(measurements, initial_state)
        
        vis.animate_comet_covar(
            true_states, measurements, estimated_states, estimate_covariances, 
            f"{f_type.upper()} Tracking with Covariance"
        )
        
    if args.run_bridge:
        logger.info("Executing Standalone Brownian Bridge Simulation...")
        bridge = BrownianBridgeSampler(n_steps=args.steps, n_particles=args.particles, variance=1.0)
        paths, weights = bridge.sample_ensemble(target_x=5.0)
        logger.info(f"Bridge Ensemble Computed. Target state reached: {paths[0, -1]:.2f}")

    logger.info("All filter animations complete. Close the windows to exit.")
    # This blocks the script from exiting until you manually close the windows
    plt.show()


if __name__ == "__main__":
    main()
