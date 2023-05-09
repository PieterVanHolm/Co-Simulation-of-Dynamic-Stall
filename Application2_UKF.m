clear
clc

%% =============================Initialisation============================
set(0,'DefaultTextInterpreter','latex');
fs=20; % Font size
lw=2; % Line size
ms=10; % Marker size

%===========Measurements=============

addpath('./CycleToCycleVariations','./CycleToCycleVariations/ModelParameters_Application2','./CycleToCycleVariations/ModelParameters_Application2/ss_nn_CycleToCycleVariation_EXP_500iter_3nx_30nn_07NcycTrain_39NcycTot_relerrAv0.3602_abserr0.0567_NLoutput_Application2.mat');
Path = './CycleToCycleVariations/ModelParameters_Application2/ss_nn_CycleToCycleVariation_EXP_500iter_3nx_30nn_07NcycTrain_39NcycTot_relerrAv0.3602_abserr0.0567_NLoutput_Application2.mat';
DATA = load(Path,'aoa16_6_1Hz','cl16_6_1Hz','t16_6_1Hz','model_nn','N');

cl_data = DATA.cl16_6_1Hz;
aoa_data  = DATA.aoa16_6_1Hz;
time_data = DATA.t16_6_1Hz;

% Create sinusoidal aoa data:

alpha_0 = 16;
amplitude = 6;
num_cycles = 39;
num_points = length(aoa_data);
frequency = num_cycles / num_points;
t = linspace(0, num_cycles/frequency, num_points);
aoa_sinusoidal = alpha_0 + amplitude*sin(2*pi*frequency*t);
aoa_sinusoidal = aoa_sinusoidal';

%=========== Model =============

%model = load('ss_nn_CycleToCycleVariation_EXP_500iter_3nx_30nn_07NcycTrain_39NcycTot_relerrAv0.2521_abserr0.0399_NLoutput_Application1.mat');
model = load('ss_nn_CycleToCycleVariation_EXP_500iter_3nx_30nn_07NcycTrain_39NcycTot_relerrAv0.3602_abserr0.0567_NLoutput_Application2.mat');
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

% % ===================== LOOP ========================
% n_measurements = length(cl_data);
% MAE_array = [];
% n_array = [];
% 
% for n = 200:-1:1
%     n
%     [MAE] = loop(n,cl_data,aoa_data,time_data,A,B,Wx,Wfx,Wfu,bf,bx,C,D,Wy,Wgx,Wgu,bg,by);
%     MAE_array = [MAE_array MAE];
%     n_array = [n_array n];
% end
% 
% figure(1)
% hold on
% title('Mean Absolute Error i.f.o. number of measurement points');
% plot(n_array,MAE_array);
% xlabel('Number of measurement points');
% ylabel('Mean Absolute Error');
% hold off
% 
% function [MAE] = loop(n,cl_data,aoa_data,time_data,A,B,Wx,Wfx,Wfu,bf,bx,C,D,Wy,Wgx,Wgu,bg,by)

% =========== Available measurements =============

% Array with # of measurement points per cycle:
available_measurements_cl = zeros(length(cl_data),1);
available_measurements_time = zeros(length(time_data),1);

n_per_cycle = 200; % <---------------------------------------------------------------------------------------------------
stap = 200/n_per_cycle;

start = 1;
available_measurements_cl(start:stap:end) = cl_data(start:stap:end);
available_measurements_time(start:stap:end) = time_data(start:stap:end);
n_measurements = nnz(available_measurements_cl);

% % Array with random measurement points:
% available_measurements_cl = zeros(length(cl_data),1);
% available_measurements_time = zeros(length(cl_data),1);
% 
% rng(3);
% n_measurements = 39*3;
% rand = randi(7800,n_measurements,1);
% for i=1:length(rand)
%     index = rand(i);
%     available_measurements_cl(index) = cl_data(index);
%     available_measurements_time(index) = time_data(index);
% end

% =========== Arrays & parameters initialisations =============

%==arrays initialisation==
nstates = 3;
noutputs = 1;
npoints = length(cl_data(:,1)); % Number of time instances
% Initialise state response array
x_array = zeros(nstates,npoints);
% Initialise state velocity array
y_array = zeros(noutputs,npoints);
% Assign initial conditions
x_array(:,1) = zeros(nstates,1);

% Initialisation Kalman filter matrices:
P_prev = [1 0 0;0 1 0;0 0 1]; % 'prev' stands for 'previous'
Q = [0 0 0;0 0 0;0 0 0];
I = [1 0 0;0 1 0;0 0 1];
R = 0.025;

% Initialize model arrays:
x_model_array = zeros(nstates,length(aoa_data));
y_model_array = zeros(length(aoa_data),1);
% set initial value x(k=0):
x_model_array(:,1) = zeros(nstates,1);

% Error array:
Error_array = zeros(npoints,1);

%% ===================== LOOP ========================
i = 0;
while i < npoints
    % Increment time index
    i=i+1;
    [y_model_array(i),x_model_array(:,i+1)] = MODEL(aoa_sinusoidal(i),x_model_array(:,i),A,B,Wx,Wfx,Wfu,bf,bx,C,D,Wy,Wgx,Wgu,bg,by);
    [x_array(:,i+1),y_array(:,i+1),P_prev,Error_array(i),Q] = UKF(nstates,available_measurements_cl(i),y_model_array(i),aoa_sinusoidal(i),cl_data(i),x_array(:,i),Q,R,P_prev,A,B,Wx,Wfx,Wfu,bf,bx,C,D,Wy,Wgx,Wgu,bg,by);
end

x_array(:,1) = [];
y_array(:,1) = [];
x_model_array(:,1) = [];

%% ==================Removal of first cycle==================:

aoa_data = aoa_data(201:end,1);
time_data = time_data(201:end,1);
cl_data = cl_data(201:end,1);
Error_array = Error_array(201:end,1);
x_array = x_array(:,201:end);
y_array = y_array(:,201:end);
x_model_array = x_model_array(:,201:end);
y_model_array = y_model_array(201:end,:);
available_measurements_cl = available_measurements_cl(201:end);
available_measurements_time = available_measurements_time(201:end);
available_measurements_time = nonzeros(available_measurements_time);

%% ================== MEAN ABSOLUTE ERROR ==================== :
MAE = mean(abs(Error_array))

%end
%% ================== PLOT ==================== :
 
figure('units','normalized','outerposition',[0 0 1 1])
meas_text = num2str(n_per_cycle);
title_text = append('Application2 (UKF): ',meas_text,' measurements per cycle');
sbpl1 = subplot(3,1,1:2);
title(title_text);
hold on
p1 = plot(time_data,cl_data,'g');
p1.Color(4) = 0.7;
p2 = plot(time_data,y_model_array,'r','MarkerSize',3);
p2.Color(4) = 0.7;
p3 = plot(time_data,y_array,'b');
if n_per_cycle < 5
    for i=1:length(available_measurements_time)
        xline(available_measurements_time(i),'k:');
    end
end
ps = [p1,p2,p3];
legend(ps,'measurements','model prediction','UKF prediction','Location','southeast');
ylabel('$C_L$');
xlabel('Time [s]');
hold off
sbpl2 = subplot(3,1,3);
hold on
p1 = plot(time_data,Error_array,'k');
p2 = yline(MAE,'r-');

mean_text = num2str(MAE);
text = append('Mean Absolute Error = ',mean_text);
legend('Error',text,'Location','southeast');
ylabel('Error in $C_L$');
xlabel('Time [s]');
linkaxes([sbpl1,sbpl2], 'x');
hold off

% end

%% ================== FUNCTIONS ==================== :
function [x,y,P,error,Q] = UKF(nstates,available_measurements_cl_i,y_model_array_i,aoa_data_i,cl_data_i,x_array_i,Q,R,P_prev,A,B,Wx,Wfx,Wfu,bf,bx,C,D,Wy,Wgx,Wgu,bg,by)

    % Initialization of parameters:
    alpha = 1;
    betha = 2;
    kappa = nstates*((1/alpha^2)-1);
    lambda = (alpha^2)*(nstates+kappa) - nstates;
    
    %========ADAPTIVE Q MATRIX DEPENDING ON ERROR BETWEEN MEASUREMENT AND MODEL=========
    measurement = available_measurements_cl_i;
    if measurement ~= 0
        mistake = abs(measurement - y_model_array_i);
        Power = 3 - 5*mistake;
        %Q = [1 0 0;0 1 0;0 0 1]*10.^(-Power);
        Q = [1 0 0;0 1 0;0 0 0]*10.^(-4);
    end
    
    %========PREDICTION=========

    % Load previous update:
    u = aoa_data_i;
    x_prev = x_array_i;
    
    % Eigenvalue decomposition
    P_prev = P_prev.*(nstates + lambda);
    [V,Diag] = eig(P_prev);
    P_sqrt = V*sqrt(Diag)*transpose(V);
    
    % Compute Sigma points and weights:
    Wi_m_prev = zeros(1,2*nstates+1);
    Wi_m_prev(1) = lambda/(nstates + lambda);
    Wi_c_prev = zeros(1,2*nstates+1);
    Wi_c_prev(1) = lambda/(nstates + lambda) + (1 - alpha^2 + betha);
    
    Xi_prev = zeros(nstates,2*nstates+1);
    Xi_prev(:,1) = x_prev;
    
    for j = 1:nstates
        range = P_sqrt(:,j);
        Xi_prev(:,j) = x_prev + range;
        Xi_prev(:,nstates+j) = x_prev - range;
        Wi_m_prev(:,j) = 1/(2*(nstates + lambda));
        Wi_m_prev(:,nstates+j) = 1/(2*(nstates + lambda));
        Wi_c_prev(:,j) = 1/(2*(nstates + lambda));
        Wi_c_prev(:,nstates+j) = 1/(2*(nstates + lambda));
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
    y_pred = 0;
    for j = 1:2*nstates+1
        y_pred = y_pred + Wi_m_prev(j)*h_Xi_pred_array(:,j);
    end
    
    x = x_pred;
    y = y_pred;
    P = P_pred;
    
    %========UPDATE=========
    if measurement ~= 0
        P_y = 0;
        P_xy = zeros(nstates,1);
        for j = 1:2*nstates+1
            covar_x = f_Xi_pred_array(:,j) - x_pred;
            covar_y = h_Xi_pred_array(:,j) - y_pred;
            P_y = P_y + Wi_c_prev(:,j)*((covar_y)*(covar_y)');
            P_xy = P_xy + Wi_c_prev(:,j)*((covar_x)*(covar_y)');
        end 
        P_y = P_y + R;
        K = P_xy/P_y;
        e = measurement - y_pred;
        x_upd = x_pred + K*e;
        P_upd = P_pred - K*P_y*K';
        P = P_upd;
        x = x_upd;
        
%         y_upd = SSNN(u,x_upd,A,B,Wx,Wfx,Wfu,bf,bx,C,D,Wy,Wgx,Wgu,bg,by);
%         y = y_upd;
        
    end
    
    %====calculate error=====
    error = cl_data_i - y_pred;
end

function [y_model,x_model] = MODEL(aoa_data,x_model_array,A,B,Wx,Wfx,Wfu,bf,bx,C,D,Wy,Wgx,Wgu,bg,by)
    [y_model,x_model] = SSNN(aoa_data,x_model_array,A,B,Wx,Wfx,Wfu,bf,bx,C,D,Wy,Wgx,Wgu,bg,by);
end