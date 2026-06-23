% Solves the 2-D tracking problem using the Kalman filter.
%
% For the purpose of testing, we generate an object (such as a car) that 
% moves around in R^2 with random accelaration.  Given a set of noisy 
% observations, we use the Kalman filter to come up with an "optimal" 
% prediction of state as well as the covariance of our prediction at each 
% timestep.
%
% Optimal prediction here is possible because both the state and 
% observation models are linear and the noise, and thus all associated 
% probability density functions, are Gaussian.
%
% The position and velocity of the object (x,y,u,v) are predicted
% although measurements are only available for the position (x,y).  
%
% We can imagine a physical system modelled by the system of stochastic 
% differential equations
%
% dx/dt = u
% dy/dt = v
% du/dt = \eta_x
% dv/dt = \eta_y
%
% with initial conditions (x,y,u,v) = 0.
%
% Here, \mathbf{\eta} = (\eta_x, \eta_y) is a multi-Gaussian white noise 
% stochastic process.  It is assumed this process is continuous time, 
% continuous valued.
% 
% We approximate this system of with a backward Euler difference equation, 
% giving a discrete time process
%
% x_k = x_{k-1} + u_{k-1} \Delta t + a_x \frac{\Delta t^2}{2} 
% y_k = y_{k-1} + v_{k-1} \Delta t + a_y \frac{\Delta t^2}{2} 
% u_k = u_{k-1} + a_{k-1}^x \Delta t
% v_k = v_{k-1} + a_{k-1}^y \Delta t
%
% Where \mathbb{x} = \{ x_k, y_k, u_k, v_k \}_{k=1}^{nsteps} and
% \mathbb{a} = \{ a_x^k, a_y^k \}_{k=1}^{nsteps} are discrete 
% time, continuous valued stochastic processes.  Here \mathbb{a} 
% approximates \mathbb{\eta} as \Delta t \rightarrow 0.  We now
% have a linear stochastic differential equation which we will solve using
% the Kalman filter.
%
% We plot our solution using a modified comet plot.  Since we know the 
% variance of our prediction, we can compute confidence ellipses.  The
% 1-stdev, 2-stdev, and 3-stdev confidence ellipses are plotted.
function kalmanTrackingTest( )

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
    X0 = zeros(4,1);
    
    % Covariance between x and y random acceleration
    accel = [0.5 0.01; 0.01 0.5];
    
    % State model functions: f_k is full state model, F_k is linearized 
    % state model, Q_k is state error covariance
    f_k_func = @(k,x)trackingStateFullModel(x,dt);
    F_k_func = @(k,x)trackingStateLinearizedModel(dt);
    Q_k_func = @(k)trackingStateCovariance(accel,dt);
    
    % Get the true state we will use for our observations
    X = trueState(N,nsteps,X0, f_k_func, Q_k_func);
    
    % Solve the tracking problem using the extended Kalman filter with
    % increasing levels of observation errors
    for i = 2:3
        obserr = 10^(-8+i);
        
        % Observation model functions: h_k is full obs model, H_k is linearized
        % obs model, R_k is obs error covariance
        h_k_func = @(k,x)trackingObservationFullModel(x);
        H_k_func = @(k,x)trackingObservationLinearizedModel();
        R_k_func = @(k)trackingObservationCovariance(obserr);

        % Get the imperfect observations of the true state
        Z = observations(M, X, nsteps, R_k_func, h_k_func);

        % Run the extended Kalman filter to get predictions and covariance
        [Xhat, P] = kalman(Z, N, M, X0, nsteps, f_k_func, F_k_func, ...
            Q_k_func, h_k_func, H_k_func, R_k_func);
        
        % Plot the solution to the 2D tracking problem
        plot2DTracking(nsteps, X, Xhat, P, Z, ' - Kalman filter', ...
            '- Kalman filter', false)
    end
end

% In this problem, the full model and the linearized model are the same
function f = trackingStateFullModel(x, dt)
    f = trackingStateLinearizedModel(dt)*x;
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