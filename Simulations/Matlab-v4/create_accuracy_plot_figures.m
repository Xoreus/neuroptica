% Script to run all plotting function, saving all figures and pngs in
% their respecable folders ([FOLDER + '/Matlab_Figs/'] and [FOLDER + '/Matlab_Pngs/']))
%
% Author: Simon Geoffroy-Gagnon
% Edit: 2020.03.17

clc; close all; clear;

fig_of_merit_value = 0.75;
print_fig_of_merit = 0;
showContour = 0;
printMe = 1;

% has potential for acc vs IL
F = '/home/edwar/Documents/Github_Projects/neuroptica/tests/Analysis/outPorts_mean_pow/N=32_4';
F = '/home/edwar/Documents/Github_Projects/neuroptica/tests/Analysis/N=32';
F = '/home/edwar/Documents/Github_Projects/neuroptica/tests/Analysis/N=32_v2';

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
if 1 && 0
    plotAcc_LPU_lineplot(F, sim, topo, 1, printMe)
end
if 1 && 0
    plotAcc_singleModel_Loss_lineplot(F, sim, topo, printMe)
end
if 1 && 1
    topo = [topo(3),topo(2),topo(1)];
    plotAcc_allModels_SinglePhaseUncert(F, sim, topo, printMe)
end
cd('../Matlab-v4')
