%function [Dataset_measurements,Dataset_model,Dataset_filter] = EKF(measurement_case)

%% =============================Initialisation============================
set(0,'DefaultTextInterpreter','latex');
fs=20; % Font size
lw=2; % Line size
ms=10; % Marker size
addpath('SSNN_model_pieter','DataEXP_right');
%===========Measurements=============

%File = append(measurement_case,'.mat');
File = 'offset20_amp4_0.4Hz.mat';

FolderName = '/Users/pieter/Desktop/VUB/2022-2023/MA1 PROJECT/Ma1project_PieterVanHolm/DataEXP_right';
FullPath = fullfile(FolderName, File);
DATA = load(FullPath);%,'cl_avg','cl_std','aoa_avg','aoa_std');

ym = [DATA.cl_avg;DATA.cl_avg;DATA.cl_avg];
cl_std = DATA.cl_std;
R = [cl_std.^2;cl_std.^2;cl_std.^2]; % add uncertainty of pressure taps!!
ym_plus_error = ym + R;
ym_min_error = ym - R;
aoa_average = [DATA.aoa_avg;DATA.aoa_avg;DATA.aoa_avg];
aoa_standard = [DATA.aoa_std;DATA.aoa_std;DATA.aoa_std];

%==arrays initialisation==
nstates = 3;
noutputs = 1;
npoints = length(aoa_average(:,1)); % Number of time instances
% Initialise state response array
x_array = zeros(nstates,npoints);
% Initialise state velocity array
y_array = zeros(noutputs,npoints);
y_test_array = zeros(noutputs,npoints);
% Assign initial conditions
x_array(:,1) = zeros(nstates,1);

i=0;      % index of time instance

% Initialisation Kalman filter parameters 
P_prev = [1 0 0;0 1 0;0 0 1]; % 'prev' stands for 'previous'
Q = [0.00001 0 0;0 0.00001 0;0 0 0.00001];
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

%% =====================EKF========================

while i < npoints
    % Increment time index
    i=i+1;
    %disp(i);
    %========PREDICTION=========
    x_prev = x_array(:,i);
    u = aoa_average(i,1);
    Al = Linearize(u,x_prev,'A',A,B,Wx,Wfx,Wfu,bf,bx,C,D,Wy,Wgx,Wgu,bg,by); % Linearization around x(k/k)

    [y_pred,x_pred] = SSNN(u,x_prev,A,B,Wx,Wfx,Wfu,bf,bx,C,D,Wy,Wgx,Wgu,bg,by); % Prediction of state x(k+1/k)
    P_pred = Al*P_prev*Al' + Q; % Prediction of P(k+1/k)

    %==========UPDATE===========
    Cl = Linearize(u,x_pred,'C',A,B,Wx,Wfx,Wfu,bf,bx,C,D,Wy,Wgx,Wgu,bg,by); % Linearization around x(k+1/k)

    S = Cl*P_pred*Cl' + R(i);
    K = P_pred*Cl'/S; % Update of K(k+1)
    e = ym(i) - y_pred; % Error = measurement - prediction
    x_upd = x_pred + K*e; % Update of state vector to x(k+1/k+1)
    P_upd = (I - K*Cl)*P_pred; % Update of state uncertainty to P(k+1/k+1)
    
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
    [y_model_array(:,i),x_model_array(:,i+1)] = SSNN(aoa_average(i,1),x_model_array(:,i),A,B,Wx,Wfx,Wfu,bf,bx,C,D,Wy,Wgx,Wgu,bg,by);
    i = i+1;
end

%% ==================Removal of first cycle==================:

l = length(aoa_average(:,1))/3;
aoa_average = aoa_average(l+1:end,1);
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
plot(aoa_average(:,1),y_array(1,:),'b',aoa_average(:,1),y_model_array(1,:),'r.',aoa_average(:,1),ym(:,1),'g');
p1 = plot(aoa_average(:,1),ym_plus_error(:,1),'g:');
p1.Color(4) = 0.4;
p2 = plot(aoa_average(:,1),ym_min_error(:,1),'g:');
p2.Color(4) = 0.4;
xlabel('angle of attack [deg]');
ylabel("$C_L$");
legend('EKF','model','measurements','Location','southeast');
hold off

%% ==================Output==================== :
Dataset_angles = aoa_average;
Dataset_measurements = ym;
Dataset_model = y_model_array';
Dataset_filter = y_array';

%end






