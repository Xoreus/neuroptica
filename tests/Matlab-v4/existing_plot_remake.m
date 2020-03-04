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
for N = [4]
FOLDER  = sprintf('/storage/Research/02.2020-NewPaper/N=%d/N=%d-PhiTheta/', N, N);
[acc, sim, topo] = load_ONN_data(FOLDER, N, loss_dB);
makeMatlabDirs(FOLDER)
warning('off', 'MATLAB:table:ModifiedAndSavedVarnames')
if ~sim.(topo{1}).same_phase_uncert
    phiTheta(FOLDER, sim, acc, topo, fig_of_merit_value, showContour, print_fig_of_merit, printMe); % Plots colormap of acc with phi vs theta phase uncert at specific loss/MZI
end

FOLDER  = sprintf('/storage/Research/02.2020-NewPaper/N=%d/N=%d-Loss+PU/', N, N);
[acc, sim, topo] = load_ONN_data(FOLDER, N, loss_dB);
if 1
    loss_phaseUncert(FOLDER, sim, acc, topo, fig_of_merit_value, showContour, print_fig_of_merit, printMe) % plots colormap of acc with phase uncert vs loss/MZI
end

if N < 8 && 0
    FOLDER  = sprintf('/storage/Research/02.2020-NewPaper/N=%d/N=%d-OG/', N, N);
    [acc, sim, topo] = load_ONN_data(FOLDER, N, loss_dB);
    if ~isempty(sim.(topo{1}).losses) && 1
        ONN_Backprop_Plot(FOLDER, sim, topo, printMe)
    end
    plotAcc_singleModel_AllLoss_lineplot(FOLDER, sim, acc, topo, printMe)
end
cd('../Matlab-v3')
end