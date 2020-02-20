% Script t]o run both plotting function, saving all figures and pngs in
% their respecable folders ([FOLDER + '/Matlab_Figs/'] and [FOLDER + '/Matlab_Pngs/']))
%
% Author: Simon Geoffroy-Gagnon
% Edit: 25.01.2020

% SINGLE TRAINING ANALYSIS

clc; close all; clear;

Folder = '/home/simon/Documents/neuroptica/tests/Analysis/single_loss/';
% Folder = '/home/simon/Documents/neuroptica/tests/Analysis/Noise-single_loss/';
% Folder = '/home/simon/Documents/neuroptica/tests/Analysis/Good_Plots/goood-lossImbalance-plots/';

fig_of_merit_value = 0.9;
showContour = true;
print_fig_of_merit = true;

ActualFolder = ['MNIST_N=4_loss-diff=0_rng1'];

FOLDER = [Folder, ActualFolder, '/'];
%     FOLDER = [ActualFolder, '/'];

SimulationSettings = load_ONN_data(FOLDER);

makeMatlabDirs(FOLDER)
warning( 'off', 'MATLAB:table:ModifiedAndSavedVarnames')

if 1
    if ~SimulationSettings.same_phase_uncert && 0
        accuracy_colormap_phi_theta_plotAccuracyArea_FoM_maxAcc(FOLDER, SimulationSettings, fig_of_merit_value, showContour, print_fig_of_merit) % Plots colormap of acc with phi vs theta phase uncert at specific loss/MZI
    end
    accuracy_colormap_phaseUncert_plotAccuracyArea_FoM_maxAcc(FOLDER, SimulationSettings, fig_of_merit_value, showContour, print_fig_of_merit) % plots colormap of acc with phase uncert vs loss/MZI
    close all
end

% ONN_Accuracy_Plot(FOLDER, SimulationSettings)
% close all
% plotAcc_singleModel_AllLoss(FOLDER, SimulationSettings)r
% close all

cd('../MatlabV2')
