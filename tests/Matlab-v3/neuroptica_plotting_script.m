% Script to run both plotting function, saving all figures and pngs in
% their respecable folders ([FOLDER + '/Matlab_Figs/'] and [FOLDER + '/Matlab_Pngs/']))
%
% Author: Simon Geoffroy-Gagnon
% Edit: 20.02.2020

% SINGLE TRAINING ANALYSIS

clc; close all; clear;

Folder = '/home/simon/Documents/neuroptica/tests/Analysis/linsep/';
Folder = '/home/simon/Documents/neuroptica/tests/Analysis/linsep/N=16';
fig_of_merit_value = 0.75;
print_fig_of_merit = false;
showContour = false;
printMe  = true;
loss_dB = 0;

N = [8*2*2]; % 8, 16, 32];
N = [8]; % 8, 16, 32];
for ii = 1:length(N)
    
%     ActualFolder = ['N=' num2str(N(ii))];
%     ActualFolder = ['N=' num2str(N(ii)), '-newSimValues#3'];
%     ActualFolder = 'N=16_16';
%     FOLDER = [Folder, ActualFolder, '/'];
    FOLDER = '/storage/Research/02.2020-NewPaper/N=16/N=16-Loss+PU/';
    FOLDER = '/storage/Research/02.2020-NewPaper/N=16/N=16-PhiTheta/';
    FOLDER = '/storage/Research/02.2020-NewPaper/N=8/N=8-Backprop/';
%     FOLDER = '/home/simon/Documents/neuroptica/tests/Analysis/linsep/N=32_0-PhiTheta_FINAL/';
%     FOLDER = '/home/simon/Documents/neuroptica/tests/Analysis/linsep/N=32_0-Loss+PU_FINAL/';
    [acc, sim, topo] = load_ONN_data(FOLDER, N(ii), loss_dB);
    makeMatlabDirs(FOLDER)
    warning('off', 'MATLAB:table:ModifiedAndSavedVarnames')
    
    if ~sim.(topo{1}).same_phase_uncert && 0
        phiTheta(FOLDER, sim, acc, topo, fig_of_merit_value, showContour, print_fig_of_merit, printMe); % Plots colormap of acc with phi vs theta phase uncert at specific loss/MZI
    end
    if 0 || 0
        loss_phaseUncert(FOLDER, sim, acc, topo, fig_of_merit_value, showContour, print_fig_of_merit, printMe) % plots colormap of acc with phase uncert vs loss/MZI
    end
    if ~isempty(sim.(topo{1}).losses) && 1
        ONN_Backprop_Plot(FOLDER, sim, topo, printMe)
    end    
%         plotAcc_singleModel_AllLoss(FOLDER, sim, acc, topo, printMe)
    %     plotAcc_allModels_SinglePhaseUncert(FOLDER, sim, acc, topo)
    %     close all
    cd('../Matlab-v3')
end
