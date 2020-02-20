% Script t]o run both plotting function, saving all figures and pngs in
% their respecable folders ([FOLDER + '/Matlab_Figs/'] and [FOLDER + '/Matlab_Pngs/']))
%
% Author: Simon Geoffroy-Gagnon
% Edit: 20.02.2020

% SINGLE TRAINING ANALYSIS

clc; close all; clear;

Folder = '/home/simon/Documents/neuroptica/tests/Analysis/linsep/';

fig_of_merit_value = 0.75;
print_fig_of_merit = true;
showContour = true;

rng = 4;
for ii = 1:length(rng)
    
    ActualFolder = ['N=' num2str(rng(ii)), '-newSimValues'];
    FOLDER = [Folder, ActualFolder, '/'];
     
    [acc, sim, topo] = load_ONN_data(FOLDER);
    makeMatlabDirs(FOLDER)
    warning( 'off', 'MATLAB:table:ModifiedAndSavedVarnames')
    
    if ~sim.(topo{1}).same_phase_uncert
        phiTheta(FOLDER, sim, acc, topo, fig_of_merit_value, showContour, print_fig_of_merit) % Plots colormap of acc with phi vs theta phase uncert at specific loss/MZI
    end
    loss_phaseUncert(FOLDER, sim, acc, topo, fig_of_merit_value, showContour, print_fig_of_merit) % plots colormap of acc with phase uncert vs loss/MZI
    if ~isempty(sim.(topo{1}).losses)
        ONN_Accuracy_Plot(FOLDER, sim, topo)
    end
    plotAcc_singleModel_AllLoss(FOLDER, sim, acc, topo)
    plotAcc_allModels_SinglePhaseUncert(FOLDER, sim, acc, topo)
    
    cd('../Matlab-v3')
end
