function [ Z ] = observations(M, X, nsteps, R_k_func, h_k_func)
    % set the initial measurement to be the true position
    Z(:,1) = h_k_func(0, X(:,1));
    
    % observation noise is white, thus mean 0
    n_mu = zeros(M,1);

    % make imperfect observations of state
    for k = 2:nsteps
        % Get the observation covariance from R_k_func
        R_k = R_k_func(k-1);
        
        % sample multivariate white noise with mean n_mu and cov R
        n_k = mvnrnd(n_mu, R_k);
        
        % make imperfect observation of true state X
        Z(:,k) = h_k_func(k-1, X(:,k)) + n_k';
    end
end