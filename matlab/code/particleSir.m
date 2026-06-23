function [ Xhat, W ] = particleSir( Z, N, M, X_0, nsteps, Ns, ...
    priorSample_func, likelihoodPdf_func)

    % Particle filter estimate with Ns number of ensembles
    Xhat = zeros(N, Ns, nsteps);
    
    % Weights for the particles
    W = zeros(Ns, nsteps);
    
    % Initialize all particles to the known initial state
    for i=1:Ns
        Xhat(:,i,1) = X_0;
        W(i,1) = 1/Ns;
    end
        
    % For each time step
    for k=2:nsteps
        % Sum of the particle weights
        t = 0;
        
        % For each ensemble
        for i=1:Ns
            % Sample from the prior distribution
            Xhat(:,i,k) = priorSample_func(k-1, Xhat(:,i,k-1));
            % Get the likelihood of the sample and use it as the weight 
            W(i,k) = likelihoodPdf_func(k-1, Z(:,k), Xhat(:,i,k));
            % Add to the sum
            t = t + W(i,k);
        end
        
        % renormalize weights
        W(:,k) = W(:,k)/t;
        
        % Resample the particles
        [Xstar, wstar] = resample(Xhat(:,:,k), W(:,k), Ns);
        
        % Use the resampled particles and weights
        Xhat(:,:,k) = Xstar;
        W(:,k) = wstar;
    end
end