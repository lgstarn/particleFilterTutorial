function [ X ] = trueState( N, nsteps, X_0, f_k_func, Q_k_func )
    % True state
    X = zeros(N, nsteps);

    % state noise is white, thus mean 0
    v_mu = zeros(N,1);
    
    % Set the initial state
    X(:,1) = X_0;
        
    % initialize true state
    for k = 2:nsteps
        % Get the state covariance from the Q_k_func
        Q_k = Q_k_func(k-1);
        
        % sample multivariate white noise with mean v_mu and cov Q_k
        v_km1 = mvnrnd(v_mu, Q_k);
        
        % update true state
        X(:,k) = f_k_func(k-1, X(:,k-1)) + v_km1';
    end
end

