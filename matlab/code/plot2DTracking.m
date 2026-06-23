% Plot the solution of the 2D tracking problem.  First plot a comet plot
% with the state X and noisy measurements Z, then plot the tracking
% solution with the true state X and the prediction Xhat with confidence
% ellipses from the covariance, P
function plot2DTracking( nsteps, X, Xhat, P, Z, stateStr, trackStr, smooth )
    % close all open graphs
    close all
    
    % 1/4 the screen
    scrsz = get(0,'ScreenSize');
    figure('Position',[1 scrsz(4)/2 scrsz(3)/2 scrsz(4)/2])
    
    % Initialize matrix to hold prediction confidence ellipses
    Ellipse = zeros(3,nsteps);
    
    if (smooth)
        Xhats = zeros(size(Xhat));
        smoother = -2:2;
        smoothSize = length(smoother);
    end
    
    for i=1:nsteps
        % Get the x and y covariances for the ith time step
        Pt = P(1:2,1:2,i);
        
        % Get the eigenvalues/vectors for this timestep
        [v lambda] = eig(Pt);
        
        % Set the axis length of the ellipses to be the eigenvalues of Pt
        Ellipse(1,i) = sqrt(lambda(1,1));
        Ellipse(2,i) = sqrt(lambda(2,2));
        
        % if the first eigenvalue is bigger, use it as semimajor axis
        if (Ellipse(1,i) > Ellipse(2,i))
            Ellipse(3,i) = atan2(v(2,1),v(1,1));
        % else, the second eigenvalue is bigger, use it as semimajor axis
        else
            Ellipse(3,i) = atan2(v(2,2),v(1,2));
        end
    
        if (smooth)
            if (i < smoother(smoothSize)+1 || i >= nsteps + smoother(1) - 1)
                Xhats(:,i) = Xhat(:,i);
            else
                Xhats(:,i) = mean(Xhat(:,i+smoother),2);
            end
        end
    end
    
    % plot a graph of the true state and measurements
    myCometCovar([Z(1,:); X(1,:)], [Z(2,:); X(2,:)],[2;1],...
        sprintf('State evolution and measurements %s',stateStr))
    pause(2);
    
    if (smooth)
        % plot a graph of the true state and smoothed tracking estimate 
        % with a confidence ellipse for 1-stdev, 2-stdev, and 3-stdev
        myCometCovar([X(1,:); Xhats(1,:); Xhats(1,:); Xhats(1,:)], ...
            [X(2,:); Xhats(2,:); Xhats(2,:); Xhats(2,:)], [1;3;3;3],...
            sprintf('Smooth tracking with 1/2/3-stdev confidence ellipses %s', ...
            trackStr), Ellipse,[0;1;2;3])
    else
        % plot a graph of the true state and non-smoothed tracking estimate 
        % with a confidence ellipse for 1-stdev, 2-stdev, and 3-stdev
        myCometCovar([X(1,:); Xhat(1,:); Xhat(1,:); Xhat(1,:)], ...
            [X(2,:); Xhat(2,:); Xhat(2,:); Xhat(2,:)], [1;3;3;3],...
            sprintf('Tracking with 1/2/3-stdev confidence ellipses %s', ...
            trackStr), Ellipse,[0;1;2;3])
    end
    pause(2);
end