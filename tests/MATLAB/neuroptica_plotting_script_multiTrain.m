% Script to run both plotting function for the multiTrain simulations, saving all figures and pngs in
% their respecable folders ([FOLDER + '/Matlab_Figs/'] and [FOLDER + '/Matlab_Pngs/'])
% 
% Author: Simon Geoffroy-Gagnon
% Edit: 23.01.2020


clc; close all; clear;

% FOLDER = '~/Documents/neuroptica/tests/Analysis/multiLossAnalysis/QuickTest/';
% FOLDER = '/home/simon/Documents/neuroptica/tests/Analysis/multiLossAnalysis/QuickerTest#2/';
FOLDER = '/home/simon/Documents/neuroptica/tests/Analysis/multiLossAnalysis/test/';

plotAcc_SingleModelAllLossAllTrainings(FOLDER)
plotAcc_SingleLossAllModelsAllTrainings(FOLDER)
cd('../MATLAB')