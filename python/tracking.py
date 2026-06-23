"""
Optimal Estimation and Particle Filtering Data Processing Pipeline
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
    """Standard linear Kalman Filter optimal estimator."""
    
    def run_filter(self, measurements: FloatArray, initial_state: FloatArray) -> Tuple[FloatArray, FloatArray]:
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
            
        return estimated_states, estimate_covariances


class BaseParticleFilter(BaseTrackingFilter):
    """Intermediate class holding shared Monte Carlo systematic resampling logic."""
    
    def __init__(self, model: StateSpaceModel, n_particles: int = 100, **kwargs: Any) -> None:
        super().__init__(model)
        self.n_particles = n_particles

    def _resample_indices(self, weights: FloatArray) -> npt.NDArray[np.int_]:
        """Calculates systematic resampling indices to resolve particle weight degeneracy."""
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
    """Auxiliary Sampling Importance Resampling (ASIR) particle filter implementation."""

    def run_filter(self, measurements: FloatArray, initial_state: FloatArray) -> Tuple[FloatArray, FloatArray]:
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
        q_inv = np.linalg.inv(self.model.process_covariance)
        r_inv = np.linalg.inv(self.model.sensor_covariance)
        hessian = q_inv + self.model.observation_model.T @ r_inv @ self.model.observation_model
        posterior_cov = np.linalg.inv(hessian)

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


class FilterFactory:
    """Registry-based Factory pattern for dynamically instantiating specific tracking filters."""
    
    _registry: dict[str, Type[BaseTrackingFilter]] = {
        'kalman': KalmanFilter,
        'sir': SIRParticleFilter,
        'asir': ASIRParticleFilter,
        'cir': CIRParticleFilter
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
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
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
            eigenvalues, eigenvectors = np.linalg.eig(step_covariance)
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
        plt.pause(1.5)
        plt.close()


# ---------------------------------------------------------
# EXECUTION PIPELINE / CLI
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Optimal Estimation and Particle Filtering Data Processing Pipeline")
    parser.add_argument("--filter", type=str, default="all", choices=['all', 'kalman', 'sir', 'asir', 'cir'], 
                        help="Specify which filter algorithm to execute.")
    parser.add_argument("--particles", type=int, default=50, 
                        help="Number of particles for Monte Carlo ensemble filters.")
    parser.add_argument("--steps", type=int, default=100, 
                        help="Number of simulation timesteps.")
    parser.add_argument("--frame-skip", type=int, default=2, 
                        help="Animation frame skip interval for rendering speed.")
    parser.add_argument("--run-bridge", action="store_true", 
                        help="Execute the standalone Brownian Bridge SDE solver.")
    args = parser.parse_args()

    logger.info("Defining System Physics and Sensor Configuration Data Model...")
    time_step = 0.05
    obs_variance = 0.1
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
    
    vis = FilterVisualizer(frame_skip=args.frame_skip)

    # Determine which filters to run
    filters_to_run = ['kalman', 'sir', 'asir', 'cir'] if args.filter == 'all' else [args.filter]

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


if __name__ == "__main__":
    main()
