% Script to run both plotting function, saving all figures and pngs in
% their respecable folders ([FOLDER + '/Matlab_Figs/'] and [FOLDER + '/Matlab_Pngs/']))
%
% Author: Simon Geoffroy-Gagnon
% Edit: 25.01.2020

% SINGLE TRAINING ANALYSIS

clc; close all; clear;

Folder = '/home/simon/Documents/neuroptica/tests/Analysis/SingleLossAnalysis/';

rng = 20:38;
rng = 33;
% Dataset = 'Gauss'
% loss_diff = 0
for ii = 1:length(rng)
    
    ActualFolder = sprintf('Loss_Imbalance_figures_rng%d', rng(ii));
    ActualFolder = 'Reck+Diamond+clements_MNIST_N=10_loss-diff=0.1_rng5';
    ActualFolder = 'Loss_Imbalance_figures_rng333_retest';
    %     ActualFolder = 'Reck+Diamond+clements_MNIST_loss-diff=0.1_rng7';
    %     ActualFolder = 'Reck+Diamond+clements_MNIST_loss-diff=0.1_rng5';
    FOLDER = [Folder ActualFolder '/'];
    
    SimulationSettings = load_ONN_data(FOLDER);
    makeMatlabDirs(FOLDER)
    warning( 'off', 'MATLAB:table:ModifiedAndSavedVarnames')
    
    accuracy_colormap(FOLDER, SimulationSettings)
%     ONN_Accuracy_Plot(FOLDER, SimulationSettings)
    
    %     close all;
    %     plotAcc_allModels_SingleLoss(FOLDER, SimulationSettings)
    %     close all;
    %     plotAcc_allModels_SinglePhaseUncert(FOLDER, SimulationSettings)
    %     close all;
%         plotAcc_singleModel_AllLoss(FOLDER, SimulationSettings)
        close all;
    cd('../MATLAB')
end
