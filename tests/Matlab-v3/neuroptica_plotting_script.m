% Script to run both plotting function, saving all figures and pngs in
% their respecable folders ([FOLDER + '/Matlab_Figs/'] and [FOLDER + '/Matlab_Pngs/']))
%
% Author: Simon Geoffroy-Gagnon
% Edit: 20.02.2020

% SINGLE TRAINING ANALYSIS

clc; close all; clear;

Folder = '/home/simon/Documents/neuroptica/tests/Analysis/linsep/';
% Folder = '/home/simon/Documents/neuroptica/tests/Analysis/';

fig_of_merit_value = 0.75;
print_fig_of_merit = true;
showContour = false;
printMe  = false;
loss_dB = 0;

N = 8;
for ii = 1:length(N)
    
    ActualFolder = ['N=' num2str(N(ii)), '-newSimValues-phiTheta'];
    %     ActualFolder = ['N=' num2str(N(ii))];
%     ActualFolder = 'test';
    FOLDER = [Folder, ActualFolder, '/'];
    
    [acc, sim, topo] = load_ONN_data(FOLDER, N(ii), loss_dB);
    makeMatlabDirs(FOLDER)
    warning( 'off', 'MATLAB:table:ModifiedAndSavedVarnames')
    
    if ~sim.(topo{1}).same_phase_uncert
        phiTheta(FOLDER, sim, acc, topo, fig_of_merit_value, showContour, print_fig_of_merit, printMe); % Plots colormap of acc with phi vs theta phase uncert at specific loss/MZI
    end
%     loss_phaseUncert(FOLDER, sim, acc, topo, fig_of_merit_value, showContour, print_fig_of_merit, printMe) % plots colormap of acc with phase uncert vs loss/MZI
%     close all
    if ~isempty(sim.(topo{1}).losses)
        ONN_Accuracy_Plot(FOLDER, sim, topo, printMe)
    end
%         plotAcc_singleModel_AllLoss(FOLDER, sim, acc, topo, printMe)
%         plotAcc_allModels_SinglePhaseUncert(FOLDER, sim, acc, topo)
    %     close all
    cd('../Matlab-v3')
end
