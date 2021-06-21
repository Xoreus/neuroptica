% Script to run all plotting function, saving all figures and pngs in
% their respecable folders ([FOLDER + '/Matlab_Figs/'] and [FOLDER + '/Matlab_Pngs/']))
%
% Author: Simon Geoffroy-Gagnon
% Edit: 2020.03.17

% SINGLE TRAINING NALYSIS

clc; close all; clear;

fig_of_merit_value = 0.75;
print_fig_of_merit = 0;
showContour = 0;
printMe = 1;

for jj = 1
    % 16x16 sims
%     F = '/storage/Research/OE_2020/Figs+Datasets/N=16/N=16-Loss+PU';
        
    % 8x8 Sims
%     F = '/home/edwar/Documents/Github_Projects/neuroptica/tests/Analysis/N=8/N=8_4';
% F = '/home/edwar/Documents/Github_Projects/neuroptica/tests/Analysis/OE/N=8/N=8_7';
    % 4x4 sims
%      F = '/home/edwar/Documents/Github_Projects/neuroptica/tests/Analysis/N=4/N=4_22'; % OE Paper Lineplots
%      F = '/home/edwar/Documents/Github_Projects/neuroptica/tests/Analysis/OE/N=4/N=4_6';
%      F = '/home/edwar/Documents/Github_Projects/neuroptica/tests/Analysis/OE/N=4/N=4_6_new';
%      F = '/home/edwar/Documents/Github_Projects/neuroptica/tests/Analysis/OE/N=4/N=4_6_500';
%      F = '/home/edwar/Documents/Github_Projects/neuroptica/tests/Analysis/OE/N=4/N=4_6_highLoss';
%      F= '/home/edwar/Documents/Github_Projects/neuroptica/tests/Analysis/OE/N=4/N=4_16_stdDev'
     F='/home/edwar/Documents/Github_Projects/neuroptica/tests/Analysis/PRA/N=4';
    [sim, topo] = load_ONN_data(F);
    makeMatlabDirs(F)
    warning('off', 'MATLAB:table:ModifiedAndSavedVarnames')
    
    if 1 && 0
        phiTheta(F, sim, topo, fig_of_merit_value, showContour, print_fig_of_merit, printMe); % Plots colormap of acc with phi vs theta phase uncert at specific loss/MZI
    end
    if 1 && 0
        loss_phaseUncert(F, sim, topo, fig_of_merit_value, showContour, print_fig_of_merit, printMe) % plots colormap of acc with phase uncert vs loss/MZI
    end
    if 1 && 0
        ONN_Backprop_Plot(F, sim, topo, printMe)
    end
    if 1 && 1
%         plotAcc_LPU_lineplot(F, sim, topo, 1, printMe)
    plotAcc_Reviewer1_LinePlot_Fig(F, sim, topo, 1, printMe)   
    plotAcc_Reviewer1_LinePlot_Fig(F, sim, topo, 2, printMe)   
    
    end
    if 1 && 0
        plotAcc_singleModel_Loss_lineplot(F, sim, topo, printMe)
    end
    if 1 && 0
        plotAcc_allModels_SinglePhaseUncert(F, sim, topo, printMe)
    end
    cd('../Matlab-v4')
end
