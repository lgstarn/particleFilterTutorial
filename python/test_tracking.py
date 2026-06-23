import pytest
import numpy as np
from pydantic import ValidationError

# Assuming the main file is named tracking_pipeline.py
from tracking_pipeline import (
    StateSpaceModel, 
    TargetSimulation, 
    FilterFactory, 
    KalmanFilter, 
    SIRParticleFilter
)

@pytest.fixture
def kinematic_model() -> StateSpaceModel:
    """Fixture providing a standard 2D kinematic configuration for testing."""
    time_step = 0.05
    dt2 = 0.5 * time_step**2
    
    F = np.array([[1.0, 0.0, time_step, 0.0],
                  [0.0, 1.0, 0.0, time_step],
                  [0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0]])
    
    H = np.array([[1.0, 0.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0]])
                  
    G = np.array([[dt2, 0.0], [0.0, dt2], [time_step, 0.0], [0.0, time_step]])
    accel_cov = np.array([[0.5, 0.01], [0.01, 0.5]])
    Q = G @ accel_cov @ G.T
    
    R = np.array([[0.1, 0.0], [0.0, 0.1]])

    return StateSpaceModel(
        state_transition=F,
        observation_model=H,
        process_covariance=Q,
        sensor_covariance=R
    )

def test_model_validation():
    """Ensures Pydantic strictly enforces the configuration schema."""
    with pytest.raises(ValidationError):
        # Missing required matrices should fail immediately
        StateSpaceModel(state_transition=np.eye(4))

def test_telemetry_generation(kinematic_model):
    """Tests the L0 data pipeline dimensions and output types."""
    n_steps = 50
    sim = TargetSimulation(model=kinematic_model, n_steps=n_steps)
    initial_state = np.zeros(4)
    
    true_states, measurements = sim.generate_telemetry(initial_state)
    
    assert true_states.shape == (4, n_steps), "True state dimensions are incorrect."
    assert measurements.shape == (2, n_steps), "Measurement dimensions are incorrect."
    # Check that the system actually moved
    assert not np.allclose(true_states[:, -1], initial_state), "Target failed to evolve."

def test_factory_instantiation(kinematic_model):
    """Verifies the factory pattern successfully routes to the correct objects."""
    kalman = FilterFactory.create('kalman', model=kinematic_model)
    sir = FilterFactory.create('sir', model=kinematic_model, n_particles=50)
    
    assert isinstance(kalman, KalmanFilter)
    assert isinstance(sir, SIRParticleFilter)
    assert sir.n_particles == 50
    
    with pytest.raises(ValueError):
        FilterFactory.create('invalid_filter', model=kinematic_model)

def test_kalman_filter_execution(kinematic_model):
    """Tests L1 processing bounds and matrix properties for the Kalman Filter."""
    n_steps = 10
    sim = TargetSimulation(model=kinematic_model, n_steps=n_steps)
    initial_state = np.zeros(4)
    _, measurements = sim.generate_telemetry(initial_state)
    
    kf = FilterFactory.create('kalman', model=kinematic_model)
    x_hat, p_cov = kf.run_filter(measurements, initial_state)
    
    assert x_hat.shape == (4, n_steps)
    assert p_cov.shape == (4, 4, n_steps)
    
    # Covariance matrices must be symmetric
    final_cov = p_cov[:, :, -1]
    assert np.allclose(final_cov, final_cov.T), "Covariance matrix is not symmetric."
    
    # Covariance matrices must be positive semi-definite (eigenvalues >= 0)
    eigenvalues = np.linalg.eigvals(final_cov)
    assert np.all(eigenvalues >= -1e-8), "Covariance matrix is not positive semi-definite."

def test_particle_filter_execution(kinematic_model):
    """Tests L1 processing dimensional outputs for sequential Monte Carlo."""
    n_steps = 10
    sim = TargetSimulation(model=kinematic_model, n_steps=n_steps)
    initial_state = np.zeros(4)
    _, measurements = sim.generate_telemetry(initial_state)
    
    sir = FilterFactory.create('sir', model=kinematic_model, n_particles=20)
    x_hat, p_cov = sir.run_filter(measurements, initial_state)
    
    assert x_hat.shape == (4, n_steps)
    assert p_cov.shape == (4, 4, n_steps)
