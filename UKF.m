%function [Dataset_measurements,Dataset_model,Dataset_filter] = UKF(measurement_case)

%% =============================Initialisation============================
set(0,'DefaultTextInterpreter','latex');
fs=20; % Font size
lw=2; % Line size
ms=10; % Marker size

%===========Measurements=============

File = 'offset20_amp8_1Hz.mat';

FolderName = '/Users/pieter/Desktop/VUB/2022-2023/MA1 PROJECT/Ma1project_PieterVanHolm/DataEXP_right';
FullPath = fullfile(FolderName, File);
load(FullPath);

cl_avg = [cl_avg;cl_avg;cl_avg];
cl_std= [cl_std;cl_std;cl_std];
ym = cl_avg;
ym_plus_error = ym + cl_std;
ym_min_error = ym - cl_std;
R = (cl_std.^2);   % add uncertainty of pressure taps!!
R(1,:) = R(1,:) + 0; % adding uncertainty from the taps
aoa_avg = [aoa_avg;aoa_avg;aoa_avg];
aoa_std = [aoa_std;aoa_std;aoa_std];

%==arrays initialisation==
nstates = 3;
noutputs = 1;
npoints = length(aoa_avg(:,1)); % Number of time instances
npoints
% Initialise state response array
x_array = zeros(nstates,npoints);
% Initialise state velocity array
y_array = zeros(noutputs,npoints);
% Assign initial conditions
x_array(:,1) = zeros(nstates,1);

i=0;      % index of time instance

% Initialisation Kalman filter parameters 
P_prev = [1 0 0;0 1 0;0 0 1]*10^(-6);
Q = [1 0 0;0 1 0;0 0 1]*10^(-5);

I = [1 0 0;0 1 0;0 0 1];

% Load model
model = load('ss_nn_results_8sweeps_1000iter_3nx_30nn_4000N_relerrAv0p0138_abserr0p0031_NLoutput.mat');
% extract model parameters
A   = model.model_nn.LW{2,2};
B   = model.model_nn.IW{2};
Wx  = model.model_nn.LW{2,1};
Wfx = model.model_nn.LW{1,2};
Wfu = model.model_nn.IW{1};
bf  = model.model_nn.b{1};
bx  = model.model_nn.b{2};
C   = model.model_nn.LW{4,2};
D   = model.model_nn.IW{4};
Wy  = model.model_nn.LW{4,3};
Wgx = model.model_nn.LW{3,2};
Wgu = model.model_nn.IW{3};
bg  = model.model_nn.b{3};
by  = model.model_nn.b{4};

%% =====================UKF========================
alpha = 0.001;
betha = 2;
kappa = nstates*((1/alpha^2)-1);
lambda = (alpha^2)*(nstates+kappa) - nstates;

while i < length(aoa_avg(:,1))
    % Increment time index
    i=i+1;
    %========PREDICTION=========

    % Load previous update:
    u = aoa_avg(i,1);
    x_prev = x_array(:,i);
    
    % Eigenvalue decomposition
    P_prev = P_prev.*(nstates + lambda);
    [V,Diag] = eig(P_prev);
    P_sqrt = V*sqrt(Diag)*transpose(V);
    
    % Compute Sigma points and weights:
    Wi_m_prev = zeros(1,2*nstates+1);
    Wi_m_prev(1) = lambda/(nstates+lambda);
    Wi_c_prev = zeros(1,2*nstates+1);
    Wi_c_prev(1) = lambda/(nstates+lambda) + (1-alpha^2+betha);
    
    Xi_prev = zeros(nstates,2*nstates+1);
    Xi_prev(:,1) = x_prev;
    
    for j = 1:nstates
        range = P_sqrt(:,j);
        Xi_prev(:,j) = x_prev + range;
        Xi_prev(:,nstates+j) = x_prev - range;
        Wi_m_prev(:,j) = 1/(2*(nstates+lambda));
        Wi_m_prev(:,nstates+j) = 1/(2*(nstates+lambda));
        Wi_c_prev(:,j) = 1/(2*(nstates+lambda));
        Wi_c_prev(:,nstates+j) = 1/(2*(nstates+lambda));
    end

    % Compute prediction:
    x_pred = zeros(3,1);
    P_pred = zeros(nstates,nstates);
    f_Xi_pred_array = zeros(nstates,2*nstates+1);
    h_Xi_pred_array = zeros(1,2*nstates+1);
    for j = 1:2*nstates+1
        [h_Xi_pred,f_Xi_pred] = SSNN(u,Xi_prev(:,j),A,B,Wx,Wfx,Wfu,bf,bx,C,D,Wy,Wgx,Wgu,bg,by); % Prediction of state x(k+1/k)
        x_pred = x_pred + Wi_m_prev(j)*f_Xi_pred;
        f_Xi_pred_array(:,j) = f_Xi_pred;
        h_Xi_pred_array(:,j) = h_Xi_pred;
    end
    for j = 1:2*nstates+1
        covar_x = f_Xi_pred_array(:,j) - x_pred; 
        P_pred = P_pred + Wi_c_prev(j)*(covar_x*covar_x');
    end
    P_pred = P_pred + Q;

    %========UPDATE=========

    % Compute update:
    y_pred = 0;
    for j = 1:2*nstates+1
        y_pred = y_pred + Wi_m_prev(j)*h_Xi_pred_array(:,j);
    end
    P_y = 0;
    P_xy = zeros(nstates,1);
    for j = 1:2*nstates+1
        covar_x = f_Xi_pred_array(:,j) - x_pred;
        covar_y = h_Xi_pred_array(:,j) - y_pred;
        P_y = P_y + Wi_c_prev(:,j)*((covar_y)*(covar_y)');
        P_xy = P_xy + Wi_c_prev(:,j)*((covar_x)*(covar_y)');
    end 
    P_y = P_y + R(i);
    K = P_xy/P_y;
    e = ym(i) - y_pred; 
    x_upd = x_pred + K*e;
    P_upd = P_pred - K*P_y*K'; 
    
    y_upd = SSNN(u,x_upd,A,B,Wx,Wfx,Wfu,bf,bx,C,D,Wy,Wgx,Wgu,bg,by);
    %====save data=====
    x_array(:,i+1) = x_upd;
    y_array(:,i+1) = y_upd;
    P_prev = P_upd;
end

x_array(:,1) = [];
y_array(:,1) = [];

%% ==================MODEL PREDICTION==================:

% initialize arrays:
x_model_array = zeros(nstates,npoints);
y_model_array = zeros(noutputs,npoints);
% set initial value x(k=0):
x_model_array(:,1) = zeros(nstates,1);
i = 1;
% loop model over all measurements:
while i < npoints-1
    [y_model_array(:,i),x_model_array(:,i+1)] = SSNN(aoa_avg(i,1),x_model_array(:,i),A,B,Wx,Wfx,Wfu,bf,bx,C,D,Wy,Wgx,Wgu,bg,by);
    i = i+1;
end

%% ==================Removal of first cycle==================:

l = length(aoa_avg(:,1))/3;
aoa_avg = aoa_avg(l+1:end,1);
ym = ym(l+1:end,1);
ym_plus_error = ym_plus_error(l+1:end,1);
ym_min_error = ym_min_error(l+1:end,1);
x_array = x_array(:,l+1:end);
y_array = y_array(:,l+1:end);
x_model_array = x_model_array(:,l+1:end);
y_model_array = y_model_array(:,l+1:end);

%% ==================Plot==================== :

figure(1)
hold on
title(File);
plot(aoa_avg(:,1),y_array(1,:),'b',aoa_avg(:,1),y_model_array(1,:),'r.',aoa_avg(:,1),ym(:,1),'g');
plot(aoa_avg(:,1),ym(:,1),'g');
p1 = plot(aoa_avg(:,1),ym_plus_error(:,1),'g:');
p1.Color(4) = 0.4;
p2 = plot(aoa_avg(:,1),ym_min_error(:,1),'g:');
p2.Color(4) = 0.4;
xlabel('angle of attack [deg]');
ylabel('$C_L$');
legend('UKF','model','measurements','Location','southeast');
hold off