% Script to run both plotting function, saving all figures and pngs in
% their respecable folders ([FOLDER + '/Matlab_Figs/'] and [FOLDER + '/Matlab_Pngs/']))
%
% Author: Simon Geoffroy-Gagnon
% Edit: 25.01.2020

% SINGLE TRAINING ANALYSIS

clc; close all; clear;

Folder = '/home/simon/Documents/neuroptica/tests/Analysis/SingleLossAnalysis/';

loss_diff = 0.10;
rng = [2];

for ii = 1:length(rng)
    ActualFolder = sprintf('test');
%     ActualFolder = 'test';
    FOLDER = [Folder ActualFolder '/'];
    
    SimulationSettings = load_ONN_data(FOLDER);
    makeMatlabDirs(FOLDER)
    
    accuracy_colormap(FOLDER, SimulationSettings)
    close all;
    
    plotAcc_allModels_SingleLoss(FOLDER, SimulationSettings)
    close all;
    
    plotAcc_allModels_SinglePhaseUncert(FOLDER, SimulationSettings)
    close all;
    
    plotAcc_singleModel_AllLoss(FOLDER, SimulationSettings)
    close all;
    cd('../MATLAB')
end
