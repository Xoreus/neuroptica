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
% FOLDER = '../GaussianLossDistribution_large_test_singleLoss/';
% FOLDER = '../training_at_every_loss_iris_N=4_forFigures/';
% FOLDER = '../Reck+Diamond_Topology_UniformRandomVarLoss_MNIST/';
% FOLDER = '../Reck+Diamond_Topology_UniformRandomVarLoss_MNIST#2/';
% FOLDER = '../Reck+Diamond_Topology_UniformRandomRandLoss_Gauss/';
FOLDER = '../ReckDMM+Diamond+DoubleReck_Comparisons#2/'

plotAcc_allModels_SingleLoss(FOLDER)
plotAcc_singleModel_AllLoss(FOLDER)
% plotAcc_allModels_SingleLoss_SingleNonlinearity(FOLDER)
cd('../MATLAB')