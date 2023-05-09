clear all; close all; clc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INPUTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nameModel = 'ss_nn_results_8sweeps_1000iter_3nx_30nn_4000N_relerrAv0p0138_abserr0p0031_NLoutput';

%%%%%%%%%%         Model Validation on Sinusoidal motion        %%%%%%%%%%%
% settings to create the Sine input (alpha)
offset = 16; amplitude = 8; f = 1.6; Ncyc = 3;              
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% look for the Single Sine CFD result
validationCase1 = ['../DataCFD/Validation/'...
                   'CFD_fullAeroCoeffs_off' num2str(offset) '_amp'...
                   num2str(amplitude) '_freq' num2str(f) 'Hz.mat'];
               
% Load the model               
load(['Model/' nameModel])


% Plot Model Error on Training data
yTrainActual = y_mod_train_1;
timeTrain = linspace(0,round(length(uTrain)/fs),length(uTrain))';
relerrAv = mean(relativeError);

hf = figure;
set(hf,'PaperSize',fliplr(get(hf,'PaperSize')))
set(gcf, 'Position',  [250, 400, 900, 400])

subplot(12,1,[1,2,3,4])
plot(timeTrain(mask),uTrain(mask),'k','LineWidth',0.85)
set(gca,'FontSize',20)
ylabel('\fontsize{26}\alpha\fontsize{19} [°]')
set(gca,'xticklabel',[])
ylim([-7 30])

titleName = ['Rel. err = ' num2str(relerrAv*100,'%-2.2f') '%   '...
             'Abs. err = ' num2str(abserr_train_NN,'%-1.4f'), '\fontsize{4}' newline];
title(titleName,'FontSize',19)

subplot(12,1,[5,6,7,8,9])
mod = plot(timeTrain(mask),yTrainActual(mask),':','LineWidth',1.2);
hold on;
cfd = plot(timeTrain(mask),yTrain(mask),'LineWidth',0.85);
uistack(mod,'top')
set(gca,'FontSize',20)
ylabel('C_l','FontSize',21)
legend([mod; cfd],'Model','CFD','FontSize',18,'Location','NorthWest')
ylim([-0.5 2.5])
set(gca,'xticklabel',[])

subplot(12,1,[10,11,12])
plot(timeTrain(mask), yTrain(mask)-yTrainActual(mask),'k','LineWidth',0.85)
set(gca,'FontSize',20)
xlabel('Time [s]','FontSize',21)
ylabel('Error','FontSize',21,'Position',[-7,0])
ylim([-0.2 0.2])






%%% Validation (Sinusoidal motion)
load(validationCase1)
AOA_CFD = MatrixAeroCoeffsCFD(1,:);
CL_CFD = MatrixAeroCoeffsCFD(2,:);
t_CFD = MatrixAeroCoeffsCFD(4,:);

totalTime = Ncyc/f; 
t = linspace(0,totalTime, fs*totalTime+1)';
lastPeriod = floor(fs*totalTime-fs/f):fs*totalTime;
uSine = offset + amplitude*sin(2*pi*f.*t);


%%% function to get state (x) and output (y)
[ySine,xSine] = fFilter_ss_nn_NLoutput_simple(model_nn,uSine,zeros(nx,1));


figure
mod = plot(uSine(lastPeriod), ySine(lastPeriod), 'LineWidth',1.2);
hold on; grid on
set(gca,'FontSize',16)
xlabel('\fontsize{22}\alpha\fontsize{16} [°]')
ylabel('C_l','FontSize',17)
xlim([offset-amplitude offset+amplitude])
ylim([0 2])
titleName = ['\fontsize{22}\alpha\fontsize{15}_0 = ' num2str(offset) '°    ',...
             '\fontsize{22}\alpha\fontsize{15}_1 = ' num2str(amplitude) '°    '...
             'f = ' num2str(f) ' Hz'];
title(titleName)
cfd = plot(AOA_CFD,CL_CFD, 'LineWidth',1.2);
uistack(mod,'top')
legend([mod; cfd],'Model','CFD','FontSize',15,'Location','NorthWest')



figure
plot(t,ySine, 'LineWidth',1.2)
hold on; grid on
set(gca,'FontSize',16)
xlabel('Time [s]','FontSize',17)
ylabel('C_l','FontSize',17)
xlim([0 t(end)])
ylim([0 2])







% How to extract model parameters
% A   = model_nn.LW{2,2};
% B   = model_nn.IW{2};
% Wx  = model_nn.LW{2,1};
% Wfx = model_nn.LW{1,2};
% Wfu = model_nn.IW{1};
% bf  = model_nn.b{1};
% bx  = model_nn.b{2};
% C   = model_nn.LW{4,2};
% D   = model_nn.IW{4};
% Wy  = model_nn.LW{4,3};
% Wgx = model_nn.LW{3,2};
% Wgu = model_nn.IW{3};
% bg  = model_nn.b{3};
% by  = model_nn.b{4};
