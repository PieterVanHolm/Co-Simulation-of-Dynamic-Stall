clear
clc

%% =============================Initialisation============================
set(0,'DefaultTextInterpreter','latex');
fs=20; % Font size
lw=2; % Line size
ms=10; % Marker size

%===========Measurements=============

addpath('CycleToCycleVariations','CycleToCycleVariations\ModelParameters_Application1\');
Path = './CycleToCycleVariations/ModelParameters_Application1/ss_nn_CycleToCycleVariation_EXP_500iter_3nx_30nn_07NcycTrain_39NcycTot_relerrAv0.2521_abserr0.0399_NLoutput_Application1.mat';
DATA = load(Path,'aoa16_6_1Hz','cl16_6_1Hz','t16_6_1Hz','model_nn','');

cl_data = DATA.cl16_6_1Hz;
aoa_data  = DATA.aoa16_6_1Hz;
time_data = DATA.t16_6_1Hz;

% Cycle selection:
start_number = 1200;
end_number = 1600;

three_cycles_cl = cl_data(start_number:end_number);
three_cycles_aoa = aoa_data(start_number:end_number);
three_cycles_time = time_data(start_number:end_number);

%===========Model=============

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

%% =====================EKF========================

%==arrays initialisation==
nstates = 3;
noutputs = 1;
npoints = length(three_cycles_cl(:,1)); % Number of time instances
% Initialise state response array
x_array = zeros(nstates,npoints);
% Initialise state velocity array
y_array = zeros(noutputs,npoints);
y_test_array = zeros(noutputs,npoints);
% Assign initial conditions
x_array(:,1) = ones(nstates,1);

i=0;      % index of time instance

% Initialisation Kalman filter matrices:
P_prev = [1 0 0;0 1 0;0 0 1]; % 'prev' stands for 'previous'
I = [1 0 0;0 1 0;0 0 1];
R = ones(length(three_cycles_cl),1);
R = R.*0.025;

% Initialize model arrays:
x_model_array = zeros(nstates,length(three_cycles_aoa));
y_model_array = zeros(length(three_cycles_aoa),1);
% set initial value x(k=0):
x_model_array(:,1) = zeros(nstates,1);
Error_array = zeros(npoints,1);

% Array with measurement points:
available_measurements_aoa = zeros(length(three_cycles_aoa),1);
available_measurements_cl = zeros(length(three_cycles_cl),1);

step = 100;

available_measurements_aoa(1:step:end) = three_cycles_aoa(1:step:end);
available_measurements_cl(1:step:end) = three_cycles_cl(1:step:end);

i = 0;
while i < npoints
    % Increment time index
    i=i+1;

    %========MODEL=========
    [y_model_array(i),x_model_array(:,i+1)] = SSNN(three_cycles_aoa(i),x_model_array(:,i),A,B,Wx,Wfx,Wfu,bf,bx,C,D,Wy,Wgx,Wgu,bg,by);

    %========ADAPTIVE Q MATRIX DEPENDING ON ERROR BETWEEN MEASUREMENT AND MODEL=========    
    if available_measurements_cl(i) ~= 0
        mistake = abs(available_measurements_cl(i) - y_model_array(i));
        Power = -7.8 + 14*mistake;
        Q = [1 0 0;0 1 0;0 0 1]*.10^(-Power);
    end
    %========PREDICTION=========
    x_prev = x_array(:,i);
    u = three_cycles_aoa(i,1);
    Al = Linearize(u,x_prev,'A',A,B,Wx,Wfx,Wfu,bf,bx,C,D,Wy,Wgx,Wgu,bg,by); % Linearization around x(k/k)

    [y_pred,x_pred] = SSNN(u,x_prev,A,B,Wx,Wfx,Wfu,bf,bx,C,D,Wy,Wgx,Wgu,bg,by); % Prediction of state x(k+1/k)
    P_pred = Al*P_prev*Al' + Q; % Prediction of P(k+1/k)

    %==========UPDATE===========
    Cl = Linearize(u,x_pred,'C',A,B,Wx,Wfx,Wfu,bf,bx,C,D,Wy,Wgx,Wgu,bg,by); % Linearization around x(k+1/k)
    
    e = three_cycles_cl(i) - y_pred; % Error = measurement - prediction
    S = Cl*P_pred*Cl' + R(i);
    K = P_pred*Cl'/S; % Update of K(k+1)
    x_upd = x_pred + K*e; % Update of state vector to x(k+1/k+1)
    P_upd = (I - K*Cl)*P_pred; % Update of state uncertainty to P(k+1/k+1)
    
    y_upd = SSNN(u,x_upd,A,B,Wx,Wfx,Wfu,bf,bx,C,D,Wy,Wgx,Wgu,bg,by);
    
    %====save data=====
    x_array(:,i+1) = x_upd;
    y_array(:,i+1) = y_upd;
    P_prev = P_upd;
    Error_array(i) = y_pred - cl_data(i);
end

x_array(:,1) = [];
y_array(:,1) = [];
x_model_array(:,1) = [];

MAE = mean(abs(Error_array))
%% ==================Removal of first cycle==================:

three_cycles_aoa = three_cycles_aoa(201:end,1);
three_cycles_cl = three_cycles_cl(201:end,1);
x_array = x_array(:,201:end);
y_array = y_array(:,201:end);
x_model_array = x_model_array(:,201:end);
y_model_array = y_model_array(201:end,:);
available_measurements_aoa = available_measurements_aoa(201:end);
available_measurements_cl = available_measurements_cl(201:end);

%% ==================Available measurement array==================== :
forPlot_available_measurements_aoa = [];
forPlot_available_measurements_cl = [];

for i=1:length(available_measurements_cl)
    if available_measurements_aoa(i) ~= 0
        forPlot_available_measurements_aoa = [forPlot_available_measurements_aoa; available_measurements_aoa(i)];
    end
    if available_measurements_cl(i) ~= 0
        forPlot_available_measurements_cl = [forPlot_available_measurements_cl; available_measurements_cl(i)];
    end
end
nmeasurments = length(forPlot_available_measurements_cl)
length_measurement_cycle = round(nmeasurments/3);
%% ==================Plot==================== :

figure(1)
hold on
p1 = plot(three_cycles_aoa(1:201),three_cycles_cl(1:201),'r');
p1.Color(4) = 0.2;
p2 = plot(three_cycles_aoa(1:201),y_model_array(1:201),'r.','MarkerSize',3);
p2.Color(4) = 0.4;
p3 = plot(three_cycles_aoa(1:201),y_array(1:201),'r');
p4 = plot(forPlot_available_measurements_aoa(1:length_measurement_cycle,1),forPlot_available_measurements_cl(1:length_measurement_cycle,1),'rx');

% p5 = plot(three_cycles_aoa(201:400),three_cycles_cl(201:400),'b');
% p5.Color(4) = 0.2;
% p6 = plot(three_cycles_aoa(201:400),y_model_array(201:400),'b.','MarkerSize',3);
% p6.Color(4) = 0.4;
% p7 = plot(three_cycles_aoa(201:400),y_array(201:400),'b');
% p8 = plot(forPlot_available_measurements_aoa(length_measurement_cycle+1:2*length_measurement_cycle,1),forPlot_available_measurements_cl(length_measurement_cycle+1:2*length_measurement_cycle,1),'bx');
% 
% p9 = plot(three_cycles_aoa(401:600),three_cycles_cl(401:600),'g');
% p9.Color(4) = 0.2;
% p10 = plot(three_cycles_aoa(401:600),y_model_array(401:600),'g.','MarkerSize',3);
% p10.Color(4) = 0.8;
% p11 = plot(three_cycles_aoa(401:600),y_array(401:600),'g');
% p12 = plot(forPlot_available_measurements_aoa(2*length_measurement_cycle+1:end,1),forPlot_available_measurements_cl(2*length_measurement_cycle+1:end,1),'gx');
% 
% ps = [p9,p10,p11,p12];
% msrmnts_text = num2str(nmeasurments);
% text = append('Measurement points: ',msrmnts_text);
% legend(ps,'measurements','model prediction','EKF prediction',text,'Location','southeast');
msrmnts_text = num2str(nmeasurments);
text = append('Measurement points: ',msrmnts_text);
legend('measurements','model prediction','EKF prediction',text,'Location','southeast');

ylabel('$C_L$');
xlabel('a.o.a. [deg]');
hold off


