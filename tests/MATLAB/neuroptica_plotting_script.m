% Script to run both plotting function, saving all figures and pngs in
% their respecable folders ([FOLDER + '/Figures/'])
% 
% Author: Simon Geoffroy-Gagnon
% Edit: 15.01.2020


clc; close all; clear;

% FOLDER = '../nonlinearity_MNIST_analysis_additional_tests2/';
% FOLDER = '../nonlinearity_MNIST_analysis_additional_tests5_1000valSamples/';
% FOLDER = '../nonlinearity_MNIST_analysis_AbsoluteValuesB4NonLin/';
% FOLDER = '../training_at_loss=0dB_iris_N=4_forFigures/';
% FOLDER = '../training_at_every_loss_gauss_N=8/';
FOLDER = '../test_new_code_retrained/';
% FOLDER = '../training_at_every_loss_iris_N=4_forFigures/';

plotAcc_allModels_SingleLoss(FOLDER)
plotAcc_singleModel_AllLoss(FOLDER)
plotAcc_allModels_SingleLoss_SingleNonlinearity(FOLDER)
cd('../MATLAB')