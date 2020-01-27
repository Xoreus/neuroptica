% Script to run both plotting function, saving all figures and pngs in
% their respecable folders ([FOLDER + '/Matlab_Figs/'] and [FOLDER + '/Matlab_Pngs/']))
% 
% Author: Simon Geoffroy-Gagnon
% Edit: 25.01.2020


clc; close all; clear;

Folder = '/home/simon/Documents/neuroptica/tests/Analysis/multiLossAnalysis/';

ActualFolder = 'lossDiff=0.9_Gauss_rng2';
FOLDER = [Folder ActualFolder '/'];

SimulationSettings = load_ONN_data(FOLDER);
makeMatlabDirs(FOLDER)

% plotAcc_singleModel_AllLoss_Multi_A(FOLDER, SimulationSettings)
% plotAcc_singleModel_AllLoss_Multi(FOLDER, SimulationSettings)


plotAcc_allModels_SingleLoss_Multi(FOLDER, SimulationSettings)
close all;



cd('../MATLAB')


% plotAcc_allModels_SingleLoss(FOLDER, SimulationSettings)