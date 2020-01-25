% Script to run both plotting function, saving all figures and pngs in
% their respecable folders ([FOLDER + '/Figures/'])
% 
% Author: Simon Geoffroy-Gagnon
% Edit: 15.01.2020


clc; close all; clear;

Folder = '/home/simon/Documents/neuroptica/tests/Analysis/SingleLossAnalysis/';

ActualFolder = 'diff_00_rng8';
FOLDER = [Folder ActualFolder '/'];

% plotAcc_allModels_SingleLoss(FOLDER)
% close all;

plotAcc_allModels_SinglePhaseUncert(FOLDER)
% close all;
% 
% plotAcc_singleModel_AllLoss(FOLDER)
% cd('../MATLAB')