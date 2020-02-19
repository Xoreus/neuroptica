% Script t]o run both plotting function, saving all figures and pngs in
% their respecable folders ([FOLDER + '/Matlab_Figs/'] and [FOLDER + '/Matlab_Pngs/']))
%
% Author: Simon Geoffroy-Gagnon
% Edit: 25.01.2020

% SINGLE TRAINING ANALYSIS

clc; close all; clear;

Folder = '/home/simon/Documents/neuroptica/tests/Analysis/single_loss/';
% Folder = '/home/simon/Documents/neuroptica/tests/Analysis/Good_Plots/useless-DMM/';

fig_of_merit_value = 0.75;
print_fig_of_merit = false;
showContour = true;

rng = [32];
for ii = 1:length(rng)
    
    ActualFolder = ['Gaussian_N=4_loss-diff=0_rng' num2str(rng(ii))];
%     ActualFolder = ['MNIST_N=4_loss-diff=0_rng' num2str(rng(ii))];
%     ActualFolder = ['MNIST_N=10_loss-diff=0_rng' num2str(rng(ii))];
    FOLDER = [Folder, ActualFolder, '/'];
     
    SimulationSettings = load_ONN_data(FOLDER);
    
    makeMatlabDirs(FOLDER)
    warning( 'off', 'MATLAB:table:ModifiedAndSavedVarnames')
    
    if ~SimulationSettings.same_phase_uncert
        accuracy_colormap_phi_theta_plotAccuracyArea_FoM_maxAcc(FOLDER, SimulationSettings, fig_of_merit_value, showContour) % Plots colormap of acc with phi vs theta phase uncert at specific loss/MZI
    end
    accuracy_colormap_phaseUncert_plotAccuracyArea_FoM_maxAcc(FOLDER, SimulationSettings, fig_of_merit_value, showContour, print_fig_of_merit) % plots colormap of acc with phase uncert vs loss/MZI

    ONN_Accuracy_Plot(FOLDER, SimulationSettings)
    plotAcc_singleModel_AllLoss(FOLDER, SimulationSettings)
    plotAcc_allModels_SinglePhaseUncert(FOLDER, SimulationSettings)
    
    cd('../MatlabV2')
end
