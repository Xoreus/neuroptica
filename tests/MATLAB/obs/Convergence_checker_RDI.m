clear; clc; close all;

DATASET_NUM = 19;

% Matlab plotter for Acc vs Phase inacc and loss
% FOLDER = 'Convergence_Analysis_ReckDMMInverseReck/';
FOLDER = 'Convergence_Analysis_ReckDMMInverseReck_ThetaDMM=pi/';

RDI_Phases = zeros(64, 2*DATASET_NUM);
for ii = 0:2:2*DATASET_NUM
    RDI = readmatrix([FOLDER, sprintf('phases_for_Reck+DMM+invReck%d.txt', round(ii/2))]);
    RDI_Phases(:, ii+1:(ii+2)) = RDI(:,2:end);
end

disp(round(RDI_Phases(8*7/2:8*7/2+8, :), 2))
