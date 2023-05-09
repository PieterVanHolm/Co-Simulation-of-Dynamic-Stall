function [y,x] = SSNN(u,x_prev,A,B,Wx,Wfx,Wfu,bf,bx,C,D,Wy,Wgx,Wgu,bg,by)

y = [C D]*[x_prev;u]+Wy*tansig([Wgx Wgu]*[x_prev;u]+bg)+by;

x = [A B]*[x_prev;u]+Wx*tansig([Wfx Wfu]*[x_prev;u]+bf)+bx;

% % extract model parameters
% A   = model.LW{2,2};
% B   = model.IW{2};
% Wx  = model.LW{2,1};
% Wfx = model.LW{1,2};
% Wfu = model.IW{1};
% bf  = model.b{1};
% bx  = model.b{2};
% C   = model.LW{4,2};
% D   = model.IW{4};
% Wy  = model.LW{4,3};
% Wgx = model.LW{3,2};
% Wgu = model.IW{3};
% bg  = model.b{3};
% by  = model.b{4};

% 
% y = zeros(length(u),1);
% x = zeros(size(A,1),length(u));
% xTemp = x1;

% for t=1:length(u)
%     % output equation
%     y(t) = [C D]*[xTemp;u(t)]+Wy*tansig([Wgx Wgu]*[xTemp;u(t)]+bg)+by;
%     
%     % state equation
%     xTemp = [A B]*[xTemp;u(t)]+Wx*tansig([Wfx Wfu]*[xTemp;u(t)]+bf)+bx;
%     x(:,t) = xTemp;
% end

end
