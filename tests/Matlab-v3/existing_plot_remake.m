% Script to run both plotting function, saving all figures and pngs in
% their respecable folders ([FOLDER + '/Matlab_Figs/'] and [FOLDER + '/Matlab_Pngs/']))
% 
% Author: Simon Geoffroy-Gagnon
% Edit: 20.02.2020
% 
% SINGLE TRAINING ANALYSIS

clc; close all; clear;


fig_of_merit_value = 0.75;
print_fig_of_merit = false;
showContour = false;
printMe  = true;
loss_dB = 0;

N = 4;
FOLDER  = '/storage/Research/02.2020-NewPaper/N=8/N=8-PhiTheta/';
FOLDER  = '/storage/Research/02.2020-NewPaper/N=4/N=4-PhiTheta/';
[acc, sim, topo] = load_ONN_data(FOLDER, N, loss_dB);
makeMatlabDirs(FOLDER)
warning('off', 'MATLAB:table:ModifiedAndSavedVarnames')
if ~sim.(topo{1}).same_phase_uncert
    phiTheta(FOLDER, sim, acc, topo, fig_of_merit_value, showContour, print_fig_of_merit, printMe); % Plots colormap of acc with phi vs theta phase uncert at specific loss/MZI
end


FOLDER  = '/storage/Research/02.2020-NewPaper/N=8/N=8-Loss+PU/';
FOLDER  = '/storage/Research/02.2020-NewPaper/N=4/N=4-Loss+PU/';
[acc, sim, topo] = load_ONN_data(FOLDER, N, loss_dB);
if 1
    loss_phaseUncert(FOLDER, sim, acc, topo, fig_of_merit_value, showContour, print_fig_of_merit, printMe) % plots colormap of acc with phase uncert vs loss/MZI
end


FOLDER  = '/storage/Research/02.2020-NewPaper/N=8/N=8-OG/';
FOLDER  = '/storage/Research/02.2020-NewPaper/N=4/N=4-OG/';
[acc, sim, topo] = load_ONN_data(FOLDER, N, loss_dB);
if ~isempty(sim.(topo{1}).losses) && 1
    ONN_Accuracy_Plot(FOLDER, sim, topo, printMe)
end
plotAcc_singleModel_AllLoss(FOLDER, sim, acc, topo, printMe)

cd('../Matlab-v3')
