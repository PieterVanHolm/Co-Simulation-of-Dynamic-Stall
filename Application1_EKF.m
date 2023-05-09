clc
clear 

%% =============================Initialisation============================
set(0,'DefaultTextInterpreter','latex');
fs=20; % Font size
lw=2; % Line size
ms=10; % Marker size

%===========Measurements=============

addpath('./CycleToCycleVariations','./CycleToCycleVariations/ModelParameters_Application1','./CycleToCycleVariations/ModelParameters_Application1/ss_nn_CycleToCycleVariation_EXP_500iter_3nx_30nn_07NcycTrain_39NcycTot_relerrAv0.2521_abserr0.0399_NLoutput_Application1.mat');
Path = './CycleToCycleVariations/ModelParameters_Application1/ss_nn_CycleToCycleVariation_EXP_500iter_3nx_30nn_07NcycTrain_39NcycTot_relerrAv0.2521_abserr0.0399_NLoutput_Application1.mat';
DATA = load(Path,'aoa16_6_1Hz','cl16_6_1Hz','t16_6_1Hz');

cl_data = DATA.cl16_6_1Hz;
aoa_data  = DATA.aoa16_6_1Hz;
time_data = DATA.t16_6_1Hz;

% %===========Interpolation===========
% %  aoa
% new_aoa_data = zeros(2*length(aoa_data)-1,1);
% for i=1:length(aoa_data)-1
%     new_aoa_data(2*i-1) = aoa_data(i);
%     new_aoa_data(2*i) = (aoa_data(i)+aoa_data(i+1))/2;
% end
% new_aoa_data(end) = aoa_data(end);
% aoa_data = new_aoa_data;
% 
% %  cl
% new_cl_data = zeros(2*length(cl_data)-1,1);
% for i=1:length(cl_data)-1
%     new_cl_data(2*i-1) = cl_data(i);
%     new_cl_data(2*i) = (cl_data(i)+cl_data(i+1))/2;
% end
% new_cl_data(end) = cl_data(end);
% cl_data = new_cl_data;
% 
% %  time
% new_time_data = zeros(2*length(time_data)-1,1);
% for i=1:length(time_data)-1
%     new_time_data(2*i-1) = time_data(i);
%     new_time_data(2*i) = (time_data(i)+time_data(i+1))/2;
% end
% new_time_data(end) = time_data(end);
% time_data = new_time_data;

%=========== Model =============

model = load('ss_nn_CycleToCycleVariation_EXP_500iter_3nx_30nn_07NcycTrain_39NcycTot_relerrAv0.2521_abserr0.0399_NLoutput_Application1.mat');
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

%=========== Available measurements =============

% Array with # of measurement points per cycle:
available_measurements_cl = zeros(length(cl_data),1);
available_measurements_time = zeros(length(time_data),1);

n_per_cycle = 20; % <---------------------------------------------------------------------------------------------------
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
y_test_array = zeros(noutputs,npoints);
% Assign initial conditions
x_array(:,1) = zeros(nstates,1);

% Initialisation Kalman filter matrices:
P_prev = [1 0 0;0 1 0;0 0 1]; % 'prev' stands for 'previous'
Q = [10 0 0;0 10 0;0 0 10];
I = [1 0 0;0 1 0;0 0 1];
R = 0.05;

% Initialize model arrays:
x_model_array = zeros(nstates,length(aoa_data));
y_model_array = zeros(length(aoa_data),1);
% set initial value x(k=0):
x_model_array(:,1) = zeros(nstates,1);

% Error array:
Error_array = zeros(npoints,1);

% % ===================== LOOP ========================
% n_measurements = length(cl_data);
% MAE_array = [];
% n_array = [];
% 
% for n=n_measurements:-1:1
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

%% ===================== LOOP ========================
i = 0;
while i < npoints
    % Increment time index
    i=i+1;
    [y_model_array(i),x_model_array(:,i+1)] = MODEL(aoa_data(i),x_model_array(:,i),A,B,Wx,Wfx,Wfu,bf,bx,C,D,Wy,Wgx,Wgu,bg,by);
    [x_array(:,i+1),y_array(:,i+1),P_prev,Error_array(i),Q] = EKF(aoa_data(i),cl_data(i),y_model_array(i),x_array(:,i),available_measurements_cl(i),P_prev,R,I,Q,A,B,Wx,Wfx,Wfu,bf,bx,C,D,Wy,Wgx,Wgu,bg,by);
end

x_array(:,1) = [];
y_array(:,1) = [];
x_model_array(:,1) = [];

%% ================== REMOVAL OF FIRST CYCLE ==================:
aoa_data = aoa_data(201:end,1);
time_data = time_data(201:end,1);
cl_data = cl_data(201:end,1);
Error_array = Error_array(201:end,1);
x_array = x_array(:,201:end);
y_array = y_array(:,201:end);
x_model_array = x_model_array(:,201:end);
y_model_array = y_model_array(201:end,:);
available_measurements_time = available_measurements_time(201:end);
available_measurements_time = nonzeros(available_measurements_time);

%% ================== MEAN ABSOLUTE ERROR ==================== :
MAE = mean(abs(Error_array))

%end
% ================== PLOT ==================== :

figure('units','normalized','outerposition',[0 0 1 1])
text1 = num2str(n_per_cycle);
title_text = append(text1,' measurements per cycle');

sbpl1 = subplot(4,1,1);
hold on
title(title_text);
plot(time_data,aoa_data,'r');
yline(mean(aoa_data),'r--');
ylabel('Angle of atatck [deg]');
xlabel('Time [s]');
hold off

sbpl2 = subplot(4,1,2:3);
hold on
p1 = plot(time_data,cl_data,'g');
p1.Color(4) = 0.7;
p3 = plot(time_data,y_array,'b');
p2 = plot(time_data,y_model_array,'r','MarkerSize',3);
p2.Color(4) = 0.7;
if n_per_cycle < 5
    xl1 = xline(available_measurements_time(1),'k:');
    for i=2:length(available_measurements_time)
        xline(available_measurements_time(i),'k:');
    end
    ps = [p1,p2,p3,xl1];
    legend(ps,'measurements','model prediction','EKF prediction','measurement','Location','southeast');
else
    ps = [p1,p2,p3];
    legend(ps,'measurements','model prediction','EKF prediction','Location','southeast');
end
ylabel('$C_L$');
xlabel('Time [s]');
ylim([min(y_array),max(y_array)]);
hold off

sbpl3 = subplot(4,1,4);
hold on
p2 = yline(MAE,'r-');
p1 = plot(time_data,Error_array,'k');

mean_text = num2str(MAE);
text = append('Mean Absolute Error = ',mean_text);
legend(text,'Error','Location','southeast');
ylabel('Error in $C_L$');
xlabel('Time [s]');
ylim([min(Error_array),max(Error_array)]);

linkaxes([sbpl1,sbpl2,sbpl2], 'x');
hold off

%% ================== FUNCTIONS ==================== :

function [x,y,P,error,Q,K] = EKF(aoa_data_i,cl_data_i,y_model_array_i,x_array_i,available_measurements_cl_i,P_prev,R,I,Q,A,B,Wx,Wfx,Wfu,bf,bx,C,D,Wy,Wgx,Wgu,bg,by)
    %========ADAPTIVE Q MATRIX DEPENDING ON ERROR BETWEEN MEASUREMENT AND MODEL=========
    if available_measurements_cl_i ~= 0
        mistake = abs(y_model_array_i - available_measurements_cl_i);
        Power = -7.8 + 14*mistake;
        Q = [1 0 0;0 1 0;0 0 0]*10.^(Power);
    end

    %========PREDICTION=========
    x_prev = x_array_i;
    u = aoa_data_i;
    Al = Linearize(u,x_prev,'A',A,B,Wx,Wfx,Wfu,bf,bx,C,D,Wy,Wgx,Wgu,bg,by); % Linearization around x(k/k)
    [y_pred,x_pred] = SSNN(u,x_prev,A,B,Wx,Wfx,Wfu,bf,bx,C,D,Wy,Wgx,Wgu,bg,by); % Prediction of state x(k+1/k)
    P_pred = Al*P_prev*Al' + Q; % Prediction of P(k+1/k)
    x = x_pred;
    P = P_pred;
    y = y_pred;
    %==========UPDATE===========
    K = zeros(3,3);
    if available_measurements_cl_i ~= 0
        Cl = Linearize(u,x_pred,'C',A,B,Wx,Wfx,Wfu,bf,bx,C,D,Wy,Wgx,Wgu,bg,by); % Linearization around x(k+1/k)
        e = available_measurements_cl_i - y_pred;
        S = Cl*P_pred*Cl' + R;
        K = P_pred*Cl'/S; % Update of K(k+1)
        x_upd = x_pred + K*e; % Update of state vector to x(k+1/k+1)
        P_upd = (I - K*Cl)*P_pred; % Update of state uncertainty to P(k+1/k+1)

        x = x_upd;
        P = P_upd;
        
        y_upd = SSNN(u,x_upd,A,B,Wx,Wfx,Wfu,bf,bx,C,D,Wy,Wgx,Wgu,bg,by);
        y = y_upd;
    end

    %====calculate error=====
    error = y_pred - cl_data_i;
    
end

function [y_model_array,x_model_array] = MODEL(aoa,x_model,A,B,Wx,Wfx,Wfu,bf,bx,C,D,Wy,Wgx,Wgu,bg,by)
    [y_model_array,x_model_array] = SSNN(aoa,x_model,A,B,Wx,Wfx,Wfu,bf,bx,C,D,Wy,Wgx,Wgu,bg,by);
end
