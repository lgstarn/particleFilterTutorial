% Solves the 2-D tracking problem using the SIR particle filter.
%
% See kalman tracking filter for more information.
function sirTrackingTest( )

    % number of tracking steps
    nsteps = 100;
    
    % final time
    tf = 1;
    
    % time step
    dt = tf/nsteps;
    
    % number of state variables (x,y,u,v)
    N = 4;
    
    % number of observation variables (x,y)
    M = 2;
    
    % initial state is zero for all variables
    X_0 = zeros(4,1);
    
    % Covariance between x and y random acceleration
    accel = [0.5 0.01; 0.01 0.5];
    
    % Set the number of particles
    Ns = 100;
    
    % State model functions: f_k is full state model, F_k is linearized 
    % state model, Q_k is state error covariance
    f_k_func = @(k,x)trackingStateFullModel(x,dt);
    Q_k_func = @(k)trackingStateCovariance(accel,dt);
    
    disp('Now simulating the true state...')
    
    % Get the true state we will use for our observations
    X = trueState(N, nsteps, X_0, f_k_func, Q_k_func);

    disp('Done.')
    
    priorSample_func = @(k, Xkm1)trackingPriorSample(k, Xkm1, Q_k_func, ...
        N, dt);
    
    % Solve the tracking problem using the extended Kalman filter with
    % increasing levels of observation errors
    for i = 1:3
        obserr = 10^(-8+i);
        
        fprintf('Now creating observations with covariance of %e.\n', obserr)
        
        % Observation model functions: h_k is full obs model, R_k is obs 
        % error covariance, likelihoodPdf_func is the likelihood pdf.
        h_k_func = @(k,x)trackingObservationFullModel(x);
        R_k_func = @(k)trackingObservationCovariance(obserr);
        likelihoodPdf_func = @(k,Z,X_k)trackingLikelihoodPdf(k, Z, X_k, ...
            h_k_func, R_k_func);

        % Get the imperfect observations of the true state
        Z = observations(M, X, nsteps, R_k_func, h_k_func);
        disp('Done.')

        fprintf('Now running the particle filter with %d particles...\n', Ns)

        [Xhat, W] = particleSir(Z, N, M, X_0, nsteps, Ns, ...
            priorSample_func, likelihoodPdf_func);
        
        disp('Done.')

        Xhatm = zeros(N,nsteps);
        
        for j=1:N
            Xhatm(j,:) = sum(squeeze(Xhat(j,:,:)).*W);
        end

        P = zeros(N,N,nsteps);

        for j=1:nsteps
            P(:,:,j) = cov(Xhat(:,:,j)');            
        end
        
        disp('Now plotting the tracking results...')
        
        plot2DTracking( nsteps, X, Xhatm, P, Z, ' - particle filter', ...
            sprintf(' - particle filter (Ns = %d)', Ns), false);

        disp('Done.')

% Uncomment to plot the weights - but they are all constant, so it is
% boring
%        disp('Now plotting the particle weights...')
%        close all
%        
%        ymax = max(W(:));
%        
%        for j=1:nsteps
%            bar(W(:,j),'b')
%            xlim([1,length(W(:,j))])
%            ylim([0,ymax])
%            title(sprintf('Weights at step %d',j));
%            drawnow
%            pause(0.05)
%        end
%        disp('Done.')
    end
end

% In this problem, the full model and the linearized model are the same
function fx = trackingStateFullModel(x, dt)
    fx = trackingStateLinearizedModel(dt)*x;
end

% The state operator / model which is linear
function F = trackingStateLinearizedModel(dt)
    % x_k = x_{k-1} + v_{k-1}*dt
    % y_k = y_{k-1} + u_{k-1}*dt
    % u_k = u_{k-1}
    % v_k = v_{k-1}
    F = [1 0 dt 0; 
         0 1 0 dt;
         0 0 1  0;
         0 0 0  1];
end

% Get the covariance of the state
function Q = trackingStateCovariance(accel, dt)
    
    % Complete model is:
    % x_k = x_{k-1} + v_{k-1}*dt + 0.5a_{k-1}^x dt^2
    % y_k = y_{k-1} + u_{k-1}*dt + 0.5a_{k-1}^y dt^2
    % u_k = u_{k-1} + a_{k-1}^x*dt
    % v_k = v_{k-1} + a_{k-1}^y*dt
    
    % x and y acceleration magnitude
    dt2 = 0.5*dt^2;
    
    % The scalar component of the variance matrix
    G = [dt2 0;
         0   dt2;
         dt  0;
         0   dt;];
     
    Q = G*accel*G';    
end

function fx = trackingPriorSample(k, Xkm1, Q_k_func, N, dt)
    % state noise is white, thus mean 0
    v_mu = zeros(N,1);

    % Get the covariance for this 
    Q_k = Q_k_func(k);

    % sample multivariate white noise with mean v_mu and cov Q_k
    v_km1 = mvnrnd(v_mu, Q_k);
   
    fx = trackingStateFullModel(Xkm1, dt) + v_km1';
end

% Get the covariance of the observations
function R = trackingObservationCovariance(obserr)
    R = [obserr 0; 
         0 obserr];
end

% Get the full observation model - in this case, the full model and the 
% linearized model are the same
function h = trackingObservationFullModel(x)
    h = trackingObservationLinearizedModel()*x;
end

% Get the observation linear model as a matrix
function H = trackingObservationLinearizedModel()
    % linear observation operator - throw away u and v from (x,y,u,v)
    H = [1 0 0 0;
         0 1 0 0];
end

function p = trackingLikelihoodPdf(k, Z, Xk, h_k_func, R_k_func)
    % Make a measurement of Xk
    h_xk = h_k_func(k,Xk);

    % Get the covariance for the observation operator
    R_k = R_k_func(k);

    % sample multivariate white noise with mean v_mu and cov Q_k
    p = mvnpdf(Z, h_xk, R_k);
end