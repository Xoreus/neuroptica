% Script t]o run both plotting function, saving all figures and pngs in
% their respecable folders ([FOLDER + '/Matlab_Figs/'] and [FOLDER + '/Matlab_Pngs/']))
%
% Author: Simon Geoffroy-Gagnon
% Edit: 25.01.2020

% SINGLE TRAINING ANALYSIS

clc; close all; clear;

Folder = '/home/simon/Documents/neuroptica/tests/Analysis/SingleLossAnalysis/';

fig_of_merit_value = 0.75;
showContour = true;

ActualFolder = ['MNIST_N=4_loss-diff=0.5_rng4'];

FOLDER = [Folder, ActualFolder, '/'];
%     FOLDER = [ActualFolder, '/'];

SimulationSettings = load_ONN_data(FOLDER);

makeMatlabDirs(FOLDER)
warning( 'off', 'MATLAB:table:ModifiedAndSavedVarnames')


if ~SimulationSettings.same_phase_uncert
    accuracy_colormap_phi_theta_plotAccuracyArea_FoM_maxAcc(FOLDER, SimulationSettings, fig_of_merit_value, showContour) % Plots colormap of acc with phi vs theta phase uncert at specific loss/MZI
end
accuracy_colormap_phaseUncert_plotAccuracyArea_FoM_maxAcc(FOLDER, SimulationSettings, fig_of_merit_value, showContour) % plots colormap of acc with phase uncert vs loss/MZI
close all


% ONN_Accuracy_Plot(FOLDER, SimulationSettings)
close all
% plotAcc_singleModel_AllLoss(FOLDER, SimulationSettings)
close all

cd('../MatlabV2')
