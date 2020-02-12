% Script t]o run both plotting function, saving all figures and pngs in
% their respecable folders ([FOLDER + '/Matlab_Figs/'] and [FOLDER + '/Matlab_Pngs/']))
%
% Author: Simon Geoffroy-Gagnon
% Edit: 25.01.2020

% SINGLE TRAINING ANALYSIS

clc; close all; clear;

Folder = '/home/simon/Documents/neuroptica/tests/Analysis/SingleLossAnalysis/';

fig_of_merit_value = 0.7;
showContour = true;

rng = [394, 4, 5, 6, 7];
rng = [4];
% Dataset = 'Gauss'
% loss_diff = 0
for ii = 1:length(rng)
    
    ActualFolder = 'classONNtest';
    ActualFolder = ['Gaussian_N=4_loss-diff=0.5_rng' num2str(rng(ii))];
%     ActualFolder = 'Gaussian_N=4_loss-diff=0.5_rng394';
    %     ActualFolder = 'phaseUncertTest';
    %     ActualFolder = ['AllTopologies_3DAccMap_MNIST_N=4_loss-diff=0.5_rng' num2str(rng(ii))];
    FOLDER = [Folder, ActualFolder, '/'];
    %     FOLDER = [ActualFolder, '/'];
    
    SimulationSettings = load_ONN_data(FOLDER);
    
    makeMatlabDirs(FOLDER)
    warning( 'off', 'MATLAB:table:ModifiedAndSavedVarnames')
%     if ~SimulationSettings.same_phase_uncert
%         accuracy_colormap_phi_theta_plotAccuracyArea_FoM_maxAcc(FOLDER, SimulationSettings, fig_of_merit_value, showContour) % Plots colormap of acc with phi vs theta phase uncert at specific loss/MZI
%     end
    accuracy_colormap_phaseUncert_plotAccuracyArea_FoM_maxAcc(FOLDER, SimulationSettings, fig_of_merit_value, showContour) % plots colormap of acc with phase uncert vs loss/MZI
    close all
%     ONN_Accuracy_Plot(FOLDER, SimulationSettings)
    close all
%     plotAcc_singleModel_AllLoss(FOLDER, SimulationSettings)
    close all
    
    cd('../MatlabV2')
end
