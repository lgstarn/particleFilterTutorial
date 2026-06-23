% Brownian bridge.
%
% Solves the SDE 
%
% x_{n+1} = f(x_n, t_n)*deltat + v_n, x_1 = 0, x_N = X
%
% for x_1, x_2, \ldots, x_N, where f(x_n, t_n) is a possibily non-linear 
% function, t_n = n*deltat, deltat = 1/N, and N are the number of time 
% steps.
function [ Xhat, W ] = brownianBridge( Ns, nsteps, X, beta )

    dt = 1/nsteps;

    Xhat = zeros(Ns,nsteps);

    W = zeros(Ns,1);
    W(:) = 1/Ns;
    
    t = linspace(0,1,nsteps);
    
    f_func = @(x,t)meanShiftBrownianMotion(x,t,nsteps);

    for i=1:Ns
        ksi = normrnd(0,1,nsteps-2,1);

        options = optimset('Display','off','GradObj','on');
        
        [x,fval,exitflag,output,jacobian] = ...
            fminunc(@(x)equation_4(x,t,ksi,0,X,beta,nsteps,dt,f_func),...
            normrnd(0,beta,1,nsteps-2),options);

        Xhat(i,2:nsteps-1) = x;
        Xhat(i,1) = 0;
        Xhat(i,nsteps) = X;

        F = exp(-((X - sum(f_func(Xhat(i,:), t))^2)/(2*beta)));
        W(i) = F*norm(jacobian);
    end
    
    close all
    plot(t,Xhat')
    title(sprintf('Brownian bridge with variance of %g between 0 and %g for Ns = %d',beta,X,Ns))
end

function [f g] = equation_4(xt, t, ksi, x_0, x_n, beta, nsteps, dt, f_func)
    x = [x_0 xt x_n];

    ksipart = -0.5*dot(ksi,ksi);

    a = f_func(x,t)*dt;
    
    abetapart = -((x_n - sum(a))^2)/(2*beta);
    
    expsum = 0;
    gradientpart = zeros(1,nsteps);
    
    bns = beta/nsteps;
    
    for i=2:nsteps
        temp = (x(i) - x(i-1) - a(i-1));
        gradientpart(i-1) = gradientpart(i-1) + temp/bns;
        gradientpart(i) = gradientpart(i) - temp/bns;
        expsum = expsum - temp^2/(2*bns);
    end
    
    fs = (ksipart + abetapart - expsum);
    f = fs^2;
    
    % gradient of f with respect to x
    g = -2*fs*gradientpart(2:nsteps-1);
end

function f = meanShiftBrownianMotion(x,t,nsteps)
    f = zeros(size(x))/nsteps;
end