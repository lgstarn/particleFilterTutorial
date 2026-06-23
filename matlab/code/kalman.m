% Run the extended Kalman filter on a given state model, observation model
% with known covariances.  Both the full and linearized models are
% necessary.
%
% Input:
%
% Z are the noisy observations of the true state at each timestep (M x nsteps)
% N is the dimension of the state space (scalar)
% M is the dimension of the observation space with M <= N (scalar)
% X0 is the initial state of the model (N x 1)
% nsteps are the number of time steps to run the model for (scalar)
% f_k_func is a function pointer to the full model.  The format expected is
%
%      X_k = f_k_func(k,X_{k-1})
%
% where k is a scalar representing the current time step and X_k and 
% X_{k-1} are (N x 1) state vectors
%
% F_k_func is a function pointer of the linearized state model.  The format 
% expected is
%
%      F_k = F_k_func(k,X_{k-1})
% 
% where k is a scalar representing the current time step and X_{k-1} is a 
% (N x 1) predicted state vector.  F_k should be an (N x N) 
% matrix
%
% Q_k_func is a function pointer to the state covariance.  The format
% expected is 
%
%      Q_k = Q_k_func(k)
%
% where k is a scalar representing the current time step.  Q_k should be an
% (N x N) matrix
%
% h_k_func is a function pointer to the full observation model.  The format
% expected is
%
%      Z_k = h_k_func(k,X_k)
%
% where k is a scalar representing the current time step and X_k is the
% (N x 1) predicted state at this timestep.  Z_k is an (M x 1) observation
% corresponding to this state
%
% H_k_func is a function pointer of the linearized observation model.  The 
% format expected is
%
%      H_k = H_k_func(k,X_{k-1})
% 
% where k is a scalar representing the current time step and 
% X_{k-1} is a (N x 1) state vector.  H_k should be an (M x N) matrix
% 
% R_k_func is a function pointer to the observation covariance.  The format
% expected is 
%
%      R_k = R_k_func(k)
%
% where k is a scalar representing the current time step.  R_k should be an
% (M x M) matrix
%
% Output:
%
% Xhat is the prediction of X at each timestep based on Z (N x nsteps)
% P is the covariance estimate at each timestep (N x N x nsteps)
function [ Xhat, P ] = kalman( Z, N, M, X_0, nsteps, f_k_func, F_k_func, ...
                                     Q_k_func, h_k_func, H_k_func, R_k_func)    
    % Kalman filter estimate
    Xhat = zeros(N, nsteps);
    
    % Assume initial state is known exactly - can change this later
    Xhat(:,1) = X_0;
    
    % State prediction covariance, leave at 0 for initial value since 
    % state is known perfectly initially
    P = zeros(N,N,nsteps);
    
    % Now do ensemble Kalman filter based on observations Z
    for k = 2:nsteps   
        % Get the linearized state operator from F_k_func
        F_k = F_k_func(k-1, Xhat(:,k-1));
        
        % Get prediction state from the full model
        m_k_km1 = f_k_func(k-1, Xhat(:,k-1));
        
        % Get the state covariance from Q_k_func
        Q_k = Q_k_func(k-1);
        
        % Get prediction covariance
        P_k_km1 = Q_k + F_k*P(:,:,k-1)*F_k';
        
        % Get the linearized observation operator from H_k_func
        H_k = H_k_func(k-1, m_k_km1);
        
        % Get the observation covariance from R_k_func
        R_k = R_k_func(k-1);

        % Get innovation covariance
        S = H_k*P_k_km1*H_k' + R_k; 
        
        % Get optimal Kalman gain
        K = (P_k_km1*H_k')/S;
        
        % Updated state estimate using the full observation model
        Xhat(:,k) = m_k_km1 + K*(Z(:,k) - h_k_func(k, m_k_km1));
        
        % Updated estimate covariance 
        P(:,:,k) = P_k_km1 - K*H_k*P_k_km1;
    end
end