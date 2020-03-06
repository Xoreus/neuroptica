% Script to run both plotting function, saving all figures and pngs in
% their respecable folders ([FOLDER + '/Matlab_Figs/'] and [FOLDER + '/Matlab_Pngs/']))
%
% Author: Simon Geoffroy-Gagnon
% Edit: 2020.03.04

% SINGLE TRAINING NALYSIS

clc; close all; clear;

fig_of_merit_value = 0.75;
print_fig_of_merit = false;
showContour = false;
printMe  = true;

N = [4];
for jj = 1:length(N)

    FOLDER = sprintf('/storage/Research/02.2020-NewPaper/N=%d/N=%d-OG',N(jj), N(jj));
    FOLDER = sprintf('/home/simon/Documents/neuroptica/tests/Analysis/MNIST_AddedPhaseNoise/N=%d_0.2', N(jj));
    
    [sim, topo] = load_ONN_data(FOLDER);
    makeMatlabDirs(FOLDER)
    warning('off', 'MATLAB:table:ModifiedAndSavedVarnames')
    
    if 1 && 0
        phiTheta(FOLDER, sim, topo, fig_of_merit_value, showContour, print_fig_of_merit, printMe); % Plots colormap of acc with phi vs theta phase uncert at specific loss/MZI
    end
    if 1  && 0
        loss_phaseUncert(FOLDER, sim, topo, fig_of_merit_value, showContour, print_fig_of_merit, printMe) % plots colormap of acc with phase uncert vs loss/MZI
    end
    if 1 && 1
        ONN_Backprop_Plot(FOLDER, sim, topo, printMe)
    end
    if 1 && 0
        plotAcc_singleModel_AllLoss_lineplot(FOLDER, sim, topo, 10, printMe)
    end
    if 1 && 0
        plotAcc_allModels_SinglePhaseUncert(FOLDER, sim, acc, topo)
    end
    cd('../Matlab-v4')
end
