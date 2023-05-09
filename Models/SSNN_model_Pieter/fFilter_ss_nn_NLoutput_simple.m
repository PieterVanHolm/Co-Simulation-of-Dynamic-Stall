function [y,x] = fFilter_ss_nn_NLoutput_simple(model,u,x1)
% model is a ss_nn model

% extract model parameters
A   = model.LW{2,2};
B   = model.IW{2};
Wx  = model.LW{2,1};
Wfx = model.LW{1,2};
Wfu = model.IW{1};
bf  = model.b{1};
bx  = model.b{2};
C   = model.LW{4,2};
D   = model.IW{4};
Wy  = model.LW{4,3};
Wgx = model.LW{3,2};
Wgu = model.IW{3};
bg  = model.b{3};
by  = model.b{4};


y = zeros(length(u),1);
x = zeros(size(A,1),length(u));
xTemp = x1;
for t=1:length(u)
    x(:,t) = xTemp;
    % output equation
    y(t) = [C D]*[xTemp;u(t)]+Wy*tansig([Wgx Wgu]*[xTemp;u(t)]+bg)+by;
    
    % state equation
    xTemp = [A B]*[xTemp;u(t)]+Wx*tansig([Wfx Wfu]*[xTemp;u(t)]+bf)+bx;
end


end