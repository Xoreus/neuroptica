% Script to run both plotting function, saving all figures and pngs in
% their respecable folders ([FOLDER + '/Matlab_Figs/'] and [FOLDER + '/Matlab_Pngs/']))
%
% Author: Simon Geoffroy-Gagnon
% Edit: 2020.03.17

% SINGLE TRAINING NALYSIS

clc; close all; clear;

fig_of_merit_value = 0.75;
print_fig_of_merit = false;
showContour = false;
printMe = false;

N = 4;
for jj = 1:length(N)
    %     F = '/home/simon/Documents/neuroptica/tests/Analysis/average-linsep/N=8';
%         F = '/storage/Research/02.2020-NewPaper/N=32/N=32-OG';
%         F = '/home/simon/Documents/neuroptica/tests/Analysis/average-linsep/N=4_new/N=4_0';
%         F = '/home/simon/Documents/neuroptica/tests/Analysis/IL/N=10_1';
%         F = '/home/simon/Documents/neuroptica/tests/Analysis/IL/N=8_0dB_train/N=8_0'; % Loss Imbalance Plot
%         F = '/home/simon/Documents/neuroptica/tests/Analysis/IL/N=8/N=8_1';
    
%     F = '/home/simon/Documents/neuroptica/tests/Analysis/average-linsep/N=48/N=48_32';
%     F = '/home/simon/Documents/neuroptica/tests/Analysis/Lossy_Training/N=8_0';   
%     F = '/home/simon/Documents/neuroptica/tests/Analysis/N=4/N=4_2_63.750';

    F = '/home/simon/Documents/neuroptica/tests/Analysis/Lossy_Training/N=8_0_old';  % RiR Lossy training
    
    [sim, topo] = load_ONN_data(F);
    makeMatlabDirs(F)
    warning('off', 'MATLAB:table:ModifiedAndSavedVarnames')
    
    if 1 && 0
        phiTheta(F, sim, topo, fig_of_merit_value, showContour, print_fig_of_merit, printMe); % Plots colormap of acc with phi vs theta phase uncert at specific loss/MZI
    end
    if 1 && 1
        loss_phaseUncert(F, sim, topo, fig_of_merit_value, showContour, print_fig_of_merit, printMe) % plots colormap of acc with phase uncert vs loss/MZI
    end
    if 1 && 0
        ONN_Backprop_Plot(F, sim, topo, printMe)
    end
    if 1 && 0
        plotAcc_LPU_lineplot(F, sim, topo, 1, printMe)
    end
    if 1 && 0
        plotAcc_singleModel_Loss_lineplot(F, sim, topo, printMe)
    end
    if 1 && 0
        plotAcc_allModels_SinglePhaseUncert(F, sim, topo, printMe)
    end
    cd('../Matlab-v4')
end
