% Script to run both plotting function, saving all figures and pngs in
% their respecable folders ([FOLDER + '/Matlab_Figs/'] and [FOLDER + '/Matlab_Pngs/']))
%
% Author: Simon Geoffroy-Gagnon
% Edit: 25.01.2020

% SINGLE TRAINING ANALYSIS

clc; close all; clear;

Folder = '/home/simon/Documents/neuroptica/tests/Analysis/SingleLossAnalysis/';

rng = 2;
% Dataset = 'Gauss'
% loss_diff = 0
for ii = 1:length(rng)

    ActualFolder = sprintf('Reck+Diamond_loss-diff=0.1_rng%d', rng(ii));
    ActualFolder = 'Reck+Diamond_MNIST_loss-diff=0.1_rng2';
    ActualFolder = 'Reck+Diamond_MNIST_loss-diff=1_rng2_TEST';
    FOLDER = [Folder ActualFolder '/'];
    
    SimulationSettings = load_ONN_data(FOLDER);
    makeMatlabDirs(FOLDER)
    
    accuracy_colormap(FOLDER, SimulationSettings)
%     close all;
%     plotAcc_allModels_SingleLoss(FOLDER, SimulationSettings)
%     close all;
%     plotAcc_allModels_SinglePhaseUncert(FOLDER, SimulationSettings)
%     close all;
%     plotAcc_singleModel_AllLoss(FOLDER, SimulationSettings)
%     close all;
    cd('../MATLAB')
end
