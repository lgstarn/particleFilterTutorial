function [ Xhat, W ] = particleCir( Z, N, M, X_0, nsteps, Ns, ...
    f_k_func, F_k_func, Q_k_func, h_k_func, H_k_func, R_k_func)

    dt = 1/Ns;

    Xhat = zeros(N, Ns, nsteps);
    W = zeros(Ns, nsteps);
    
    % Initialize all particles to the known initial state
    for i=1:Ns
        Xhat(:,i,1) = X_0;
        W(i,1) = 1/Ns;
    end

    options = optimset('Display','off','GradObj','on');
    
    ksi_mu = zeros(N,1);
    
    ksi_sigma = eye(N);
    
    for k=2:nsteps
        
        for i=1:Ns
            ksi = mvnrnd(ksi_mu, ksi_sigma)';

            x = fminunc(@(x)equation_4(k, x, Xhat(:,i,k-1), ksi, Z(:,k), ...
                f_k_func, F_k_func, Q_k_func, h_k_func, H_k_func, ...
                R_k_func, N, M, dt), ...
                Xhat(:,i,k-1), options);
            
            Xhat(:,i,k) = x;

%            F = exp(-((X - sum(f_func(Xhat(i,:), t))^2)/(2*beta)));
%            W(i) = F*norm(jacobian);
        end
            
%        [Xstar, wstar] = resample(Xhat(:,:,k), W(:,k), Ns);
        
%        Xhat(:,:,k) = Xstar;
%        W(:,k) = wstar;
    end
end

function [f g] = equation_4(k, x_k, x_km1, ksi, z_k, ...
    f_k_func, F_k_func, Q_k_func, h_k_func, H_k_func, R_k_func, N, M, dt)


%    ksipart = -log(2*pi)*N/2 -0.5*(ksi'*ksi);
    ksipart = -0.5*(ksi'*ksi);

    a = f_k_func(k,x_km1)*dt;
    
    Q_k = Q_k_func(k);
        
    qkinvxk = Q_k\(x_k-a);
    
%    xpart = -log(2*pi)*N/2 -log(det(Q_k))/2 -0.5*(x_k-a)'*qkinvxk;
    xpart = -0.5*(x_k-a)'*qkinvxk;
    
    hx = h_k_func(k, x_k);
    H_k = H_k_func(k, x_k);
    
    R_k = R_k_func(k);
    
    rkinvzk  = R_k\(hx - z_k);
    
%    hpart  = -log(2*pi)*M/2 -log(det(R_k))/2 - 0.5*(hx-z_k)'*rkinvzk;
    hpart  = - 0.5*(hx-z_k)'*rkinvzk;
    
    fs = (ksipart - xpart - hpart);
    f = fs^2;
    g = -2*fs*(-qkinvxk - (rkinvzk'*H_k)');
end