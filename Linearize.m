function [out] = Linearize(u,x,Case,A,B,Wx,Wfx,Wfu,bf,bx,C,D,Wy,Wgx,Wgu,bg,by)

% %% Load model parameters:
% A   = model.LW{2,2}; % 3x3
% Wx  = model.LW{2,1}; % 3x30
% Wfx = model.LW{1,2}; % 30x3
% Wfu = model.IW{1}; % 30x1
% bf  = model.b{1}; % 30x1
% C   = model.LW{4,2}; % 1x3
% Wy  = model.LW{4,3}; % 1x30
% Wgx = model.LW{3,2}; % 30x3
% Wgu = model.IW{3}; % 30x1 
% bg  = model.b{3}; % 30x1

%% Linearization:
if Case == 'A'
    % Al:
    Al = zeros(3,3);
    Al_1 = zeros(30,1);
    Al_2 = zeros(30,1);
    Al_3 = zeros(30,1);
    for i=1:30
        Al_1(i,1) = (sech(Wfx(i,1)*x(1,1) + Wfx(i,2)*x(2,1) + Wfx(i,3)*x(3,1) + Wfu(i)*u + bf(i)))^2*Wfx(i,1);
        Al_2(i,1) = (sech(Wfx(i,1)*x(1,1) + Wfx(i,2)*x(2,1) + Wfx(i,3)*x(3,1) + Wfu(i)*u + bf(i)))^2*Wfx(i,2);
        Al_3(i,1) = (sech(Wfx(i,1)*x(1,1) + Wfx(i,2)*x(2,1) + Wfx(i,3)*x(3,1) + Wfu(i)*u + bf(i)))^2*Wfx(i,3);
    end
    Al(:,1) = A(:,1) + Wx*Al_1;
    Al(:,2) = A(:,2) + Wx*Al_2;
    Al(:,3) = A(:,3) + Wx*Al_3;
    
    out = Al;
end


if Case == 'C'
    
    Cl = zeros(1,3);
    Cl_1 = zeros(30,1);
    Cl_2 = zeros(30,1);
    Cl_3 = zeros(30,1);
    for i=1:30
        Cl_1(i,1) = (sech(Wgx(i,1)*x(1,1) + Wgx(i,2)*x(2,1) + Wgx(i,3)*x(3,1) + Wgu(i)*u + bg(i)))^2*Wgx(i,1);
        Cl_2(i,1) = (sech(Wgx(i,1)*x(1,1) + Wgx(i,2)*x(2,1) + Wgx(i,3)*x(3,1) + Wgu(i)*u + bg(i)))^2*Wgx(i,2);
        Cl_3(i,1) = (sech(Wgx(i,1)*x(1,1) + Wgx(i,2)*x(2,1) + Wgx(i,3)*x(3,1) + Wgu(i)*u + bg(i)))^2*Wgx(i,3);
    end
    Cl(1,1) = C(1,1) + Wy*Cl_1;
    Cl(1,2) = C(1,2) + Wy*Cl_2;
    Cl(1,3) = C(1,3) + Wy*Cl_3;

    out = Cl;
end

end