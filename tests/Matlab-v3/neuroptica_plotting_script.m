% Script to run both plotting function, saving all figures and pngs in
% their respecable folders ([FOLDER + '/Matlab_Figs/'] and [FOLDER + '/Matlab_Pngs/']))
%
% Author: Simon Geoffroy-Gagnon
% Edit: 20.02.2020

% SINGLE TRAINING ANALYSIS

clc; close all; clear;

fig_of_merit_value = 0.75;
print_fig_of_merit = false;
showContour = false;
printMe  = 0;
loss_dB = 0;

N = [64]; % 8, 16, 32];
% N = [8*2]; % 8, 16, 32];
ii = '';
% ii = '-PT-F';
for jj = 1:length(N)
    
%     FOLDER = '/storage/Research/OE_2020/Figs+Datasets/N=32/N=32-Loss+PU/'; % Large range
    FOLDER = '/storage/Research/OE_2020/Figs+Datasets/N=64/N=64-LPU/';
    
    [acc, sim, topo] = load_ONN_data(FOLDER, N(jj), loss_dB);
    makeMatlabDirs(FOLDER)
    warning('off', 'MATLAB:table:ModifiedAndSavedVarnames')
    
    if ~sim.(topo{1}).same_phase_uncert && 0
        phiTheta(FOLDER, sim, acc, topo, fig_of_merit_value, showContour, print_fig_of_merit, printMe); % Plots colormap of acc with phi vs theta phase uncert at specific loss/MZI
    end
    if sim.(topo{1}).same_phase_uncert  && 1
        loss_phaseUncert(FOLDER, sim, acc, topo, fig_of_merit_value, showContour, print_fig_of_merit, printMe) % plots colormap of acc with phase uncert vs loss/MZI
    end
    if ~isempty(sim.(topo{1}).losses) && 0
        ONN_Backprop_Plot(FOLDER, sim, topo, printMe)
    end
    if 1 && 0
        plotAcc_singleModel_AllLoss_lineplot(FOLDER, sim, acc, topo, printMe)
    end
    %     plotAcc_allModels_SinglePhaseUncert(FOLDER, sim, acc, topo)
    %     close all
    cd('../Matlab-v3')
end
