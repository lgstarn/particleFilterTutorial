function [Xstar, wstar, pindex] = resample(X, W, Ns)
    % The resampled particles
    Xstar = zeros(size(X));
    
    % The resampled weights
    wstar = ones(size(W))/Ns;
    
    % The indexes of the particles
    pindex = zeros(Ns,1);

    % The cumulative weight of the particles
    c = zeros(Ns,1);
    
    % For each additional particle
    for i=2:Ns
        % Add to the cumulative weight
        c(i) = c(i-1) + W(i);
    end
    
    % The index of the particle to resample
    i = 1;
    
    % The cumlative weight 
    u = zeros(Ns,1);
    
    % Sample a uniform random between (0,1)
    u(1) = rand()/Ns;
    
    % For each particle
    for j=1:Ns
        % Compute the cumlative weight of the resamping
        u(j) = u(1) + (j-1)/Ns;
        
        % Keep moving along the CDF to find the right particle
        while i < Ns && u(j) > c(i) 
            i = i + 1;
        end
        
        % Now set the particle and parent index
        Xstar(:,j) = X(:,i);
        pindex(j) = i;
    end
end