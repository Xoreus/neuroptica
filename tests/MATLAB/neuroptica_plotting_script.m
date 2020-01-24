% Script to run both plotting function, saving all figures and pngs in
% their respecable folders ([FOLDER + '/Figures/'])
% 
% Author: Simon Geoffroy-Gagnon
% Edit: 15.01.2020


clc; close all; clear;

Folder = '/home/simon/Documents/neuroptica/tests/Analysis/SingleLossAnalysis/';

ActualFolder = 'loss_diff_0.01_test_rng2'          ;
ActualFolder = 'loss_diff_0.0_test_rng2'          ;
ActualFolder = 'loss_diff_0.1_test_LossMatrix_rng2';
ActualFolder = 'loss_diff_0.01_test_LossMatrix_rng2';
FOLDER = [Folder ActualFolder '/'];

plotAcc_allModels_SingleLoss(FOLDER)
close all;

plotAcc_allModels_SinglePhaseUncert(FOLDER)
close all;

% plotAcc_singleModel_AllLoss(FOLDER)
% plotAcc_allModels_SingleLoss_SingleNonlinearity(FOLDER)
cd('../MATLAB')