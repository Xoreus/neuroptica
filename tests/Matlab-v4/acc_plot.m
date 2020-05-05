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
    % 80x80 sims
%     F = '/home/simon/Documents/neuroptica/tests/Analysis/N=80';
    % 32x32 sims
%         F = '/home/simon/Documents/neuroptica/tests/Analysis/N=32/N=32_1';
%     16x16 sims
%         F = '/home/simon/Documents/neuroptica/tests/Analysis/N=16/N=16_0'; % Show FoM lines
%         F = '/home/simon/Documents/neuroptica/tests/Analysis/N=16/N=16_2-new';
    % 8x8 Sims
    %     F = '/home/simon/Documents/neuroptica/tests/Analysis/N=8/N=8_4';
    % 4x4 sims
%         F = '/home/simon/Documents/neuroptica/tests/Analysis/N=4/N=4_9';
%         F = '/home/simon/Documents/neuroptica/tests/Analysis/N=4/N=4_6_PT_0dB';
%         F = '/home/simon/Documents/neuroptica/tests/Analysis/N=4/N=4_19';
% 8x8 Lossy/Not lossy
F = '/home/simon/Documents/neuroptica/tests/Analysis/N=8/N=8_0_0.5dB-loss';
F = '/home/simon/Documents/neuroptica/tests/Analysis/N=8/N=8_0';
% F = '/home/simon/Documents/neuroptica/tests/Analysis/N=8/N=8_0_0.25dB-loss'
    % Random Phase --> Different Final Acc THESIS
    %         F = '/home/simon/Documents/neuroptica/tests/Analysis/N=4/N=4_2_63.750';
    %         F = '/home/simon/Documents/neuroptica/tests/Analysis/N=4/N=4_2_67.500';
    %         F = '/home/simon/Documents/neuroptica/tests/Analysis/N=4/N=4_2_100.000_2';
     
    % IL_Sensitivity_PhaseNoise THESIS
%     F = '/home/simon/Documents/neuroptica/tests/Analysis/IL_Sensitivity_PhaseNoise/N=8_0'; % 0 IL
%     F = '/home/simon/Documents/neuroptica/tests/Analysis/N=8/N=8_0-newRange';
%     F = '/home/simon/Documents/neuroptica/tests/Analysis/N=8/N=8_0_0.5dB-loss';
%     F = '/home/simon/Documents/neuroptica/tests/Analysis/N=8/N=8_0_0.25dB-loss';
    
    % DIAMOND DMM DIFFERENCE THESIS
%     F = '/home/simon/Documents/neuroptica/tests/Analysis/Diamond_DMM_Difference/N=8_2';
    
    % MNIST Backprop THESIS
%     F = '/home/simon/Documents/neuroptica/tests/Analysis/MNIST/N=4';
%     F = '/home/simon/Documents/neuroptica/tests/Analysis/MNIST/N=10';
    
    [sim, topo] = load_ONN_data(F);
    makeMatlabDirs(F)
    warning('off', 'MATLAB:table:ModifiedAndSavedVarnames')
    
    if 1 && 1
        phiTheta(F, sim, topo, fig_of_merit_value, showContour, print_fig_of_merit, printMe); % Plots colormap of acc with phi vs theta phase uncert at specific loss/MZI
    end
    if 1 && 0
        loss_phaseUncert(F, sim, topo, fig_of_merit_value, showContour, print_fig_of_merit, printMe) % plots colormap of acc with phase uncert vs loss/MZI
    end
    if 1 && 0
        ONN_Backprop_Plot(F, sim, topo, printMe)
    end
    if 1 && 0
        plotAcc_LPU_lineplot(F, sim, topo, 1, printMe)
    end
    if 1 && 0
        plotAcc_singleModel_Loss_lineplot(F, sim, topo, printMe)
    end
    if 1 && 0
        plotAcc_allModels_SinglePhaseUncert(F, sim, topo, printMe)
    end
    cd('../Matlab-v4')
end