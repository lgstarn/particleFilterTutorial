function [ Xhat, W ] = particleAsir( Z, N, M, X_0, nsteps, Ns, ...
    priorSample_func, likelihoodPdf_func)

    Xhat = zeros(N, Ns, nsteps);
    W = zeros(Ns, nsteps);
    
    % Initialize all particles to the known initial state
    for i=1:Ns
        Xhat(:,i,1) = X_0;
        W(i,1) = 1/Ns;
    end
    
    for k=2:nsteps
        t = 0;
        
        mu_k = zeros(N,Ns);
        
        for i=1:Ns
            mu_k(:,i) = priorSample_func(k-1, Xhat(:,i,k-1));
        
            W(i,k) = likelihoodPdf_func(k-1, Z(:,k), mu_k(:,i))*W(i,k-1);
            
            t = t + W(i,k);
        end
        
        % renormalize weights
        W(:,k) = W(:,k)/t;
        
        [Xstar, wstar, pindex] = resample(mu_k, W(:,k), Ns);
        
        t = 0;
        
        for j=1:Ns
            Xhat(:,j,k) = priorSample_func(k-1, Xhat(:,pindex(j),k-1));
            
            W(j,k) = likelihoodPdf_func(k-1, Z(:,k), Xhat(:,j,k))/...
                likelihoodPdf_func(k-1, Z(:,k), mu_k(:,pindex(j)));
            
            t = t + W(j,k);
        end
        
        W(:,k) = W(:,k)/t;
    end
end