function myCometCovar(varargin) %x1,y1,x2,y2,covar,titleStr)
%COMET  Comet-like trajectory.
%   COMET(Y) displays an animated comet plot of the vector Y.
%   COMET(X,Y) displays an animated comet plot of vector Y vs. X.
%   COMET(X,Y,p) uses a comet of length p*length(Y).  Default is p = 0.10.
%
%   COMET(AX,...) plots into AX instead of GCA.
%
%   Example:
%       t = -pi:pi/200:pi;
%       comet(t,tan(sin(t))-sin(tan(t)))
%
%   See also COMET3.

%   Charles R. Denham, MathWorks, 1989.
%   Revised 2-9-92, LS and DTP; 8-18-92, 11-30-92 CBM.
%   Copyright 1984-2007 The MathWorks, Inc. 
%   $Revision: 5.12.4.4 $  $Date: 2008/01/21 14:59:45 $

% Parse possible Axes input
[ax,args,nargs] = axescheck(varargin{:});

error(nargchk(4,6,nargs,'struct'));

% Parse the rest of the inputs
if nargs == 4, [x,y,type,titleStr] = deal(args{:}); hasEllipse = false; end
if nargs == 6, [x,y,type,titleStr,P,PStretch] = deal(args{:}); hasEllipse = true; end
    
if (~hasEllipse && max(type) > 2)
    error('Specified an elipse type but did not pass in the ellipse matrix.')
end

% Parse the rest of the inputs
p = 0.10;

ax = newplot(ax);
if ~ishold(ax)
  [minx,maxx] = minmax(1.05*x(:));
  [miny,maxy] = minmax(1.05*y(:));
  axis(ax,[minx maxx miny maxy])
end

title(titleStr)

NPlot = size(x,1);
m = size(x,2);

k = round(p*m);

headStyles = zeros(NPlot,1);
bodyStyles = zeros(NPlot,1);
tailStyles = zeros(NPlot,1);

A = colormap(lines);

for i=1:NPlot
    cindex = (i-1)*3;

    if (type(i) == 1)
        headStyles(i) = line('parent',ax,'color',A(cindex+1+1,:),'marker','+','erase','xor', ...
                  'xdata',x(i,1),'ydata',y(i,1),'MarkerSize',10);
    elseif (type(i) == 2)
        headStyles(i) = line('parent',ax,'color',A(cindex+1+1,:),'marker','x','erase','none', ...
                  'xdata',x(i,1),'ydata',y(i,1),'MarkerSize',12);
    else
        headStyles(i) = line('parent',ax,'color',A(cindex+1+1,:),'linestyle','-','erase','xor', ...
                  'xdata',[],'ydata',[]);
    end
    
    if (type(i) == 1 || type(i) == 3)
        bodyStyles(i) = line('parent',ax,'color',A(cindex+1+2,:),'linestyle','-','erase','none', ...
                      'xdata',[],'ydata',[]);
        tailStyles(i) = line('parent',ax,'color',A(cindex+1+3,:),'linestyle','-','erase','none', ...
                      'xdata',[],'ydata',[]);
    end
end

% This try/catch block allows the user to close the figure gracefully
% during the comet animation.
try
    % Grow the body
    for i = 2:k+1
        j = i-1:i;
        
        for l = 1:NPlot
            if (type(l) == 1 || type(l) == 2)
                plotData(headStyles(l),x(l,i),y(l,i));
            elseif (type(l) == 3)
                plotEllipse(headStyles(l),x(l,i),y(l,i),P(:,i),PStretch(l));
            end
            
            if (bodyStyles(l) ~= 0)
                plotData(bodyStyles(l),x(l,j), y(l,j));
            end
        end
        drawnow
    end

    % Primary loop
    for i = k+2:m
        j = i-1:i;

        for l = 1:NPlot
            if (type(l) == 1 || type(l) == 2)
                plotData(headStyles(l),x(l,i),y(l,i));
            elseif (type(l) == 3)
                plotEllipse(headStyles(l),x(l,i),y(l,i),P(:,i),PStretch(l));
            end
            
            if (bodyStyles(l) ~= 0)
                plotData(bodyStyles(l),x(l,j), y(l,j));
            end
            
            if (tailStyles(l) ~= 0)
                plotData(tailStyles(l),x(l,j-k),y(l,j-k))
            end
        end
        drawnow
    end

    % Clean up the tail
    for i = m+1:m+k
        j = i-1:i;

        for l = 1:NPlot
            if (tailStyles(l) ~= 0)
                plotData(tailStyles(l),x(l,j-k),y(l,j-k))
            end
        end
        drawnow
    end
catch E
    if ~strcmp(E.identifier, 'MATLAB:class:InvalidHandle')
        rethrow(E);
    end
end

function plotData(lineStyle, x, y)
    for n=1:size(x,1)
        set(lineStyle(n),'xdata',x(n,:),'ydata',y(n,:));
    end
end

function plotEllipse(lineStyle,x,y,P,PStretch)
    Nb = 50;
    xpos=x;
    ypos=y;
    radm=PStretch*P(1);
    radn=PStretch*P(2);
    an = P(3);
    co=cos(an);
    si=sin(an);
    the=linspace(0,2*pi,Nb(rem(k-1,size(Nb,1))+1,:)+1);
    
    Xe = radm*cos(the)*co-si*radn*sin(the)+xpos;
    Ye = radm*cos(the)*si+co*radn*sin(the)+ypos;

    set(lineStyle,'XData',Xe,'YData',Ye);

    drawnow
end

function [minx,maxx] = minmax(x)
    minx = min(x(isfinite(x)));
    maxx = max(x(isfinite(x)));
    if minx == maxx
      minx = maxx-1;
      maxx = maxx+1;
    end
end

end