function [N, Models, Nonlin, phase_uncert, loss_dB, iterations, DATASET_NUM] = load_ONN_data(FOLDER)
% Get nonlin, Models, Phase Uncert, loss, iterations, N
N = load([FOLDER, 'N.txt']);
Models = textread([FOLDER, 'ONN_Setups.txt'], '%s', 'delimiter', '\n');
Nonlin = textread([FOLDER, 'Nonlinearities.txt'], '%s', 'delimiter', '\n');
phase_uncert = load([FOLDER, 'PhaseUncert.txt']);
loss_dB = load([FOLDER, 'LossdB.txt']);
iterations = load([FOLDER, 'ITERATIONS.txt']); % How many times we test the same structure
DATASET_NUM = 0;
end