function [N, Models, Nonlin, phase_uncert_train, phase_uncert_test, loss_dB_train,loss_dB_test, iterations, DATASET_NUM] = load_ONN_data_MultiTrain(FOLDER)
% Get nonlin, Models, Phase Uncert, loss, iterations, N
N = load([FOLDER, 'N.txt']);

Models = strsplit(fileread([FOLDER, 'ONN_Setups.txt']), '\n');
Models = Models(1:end-1);

Nonlin = strsplit(fileread([FOLDER, 'Nonlinearities.txt']), '\n');
Nonlin = Nonlin(1:end-1);

phase_uncert_train = load([FOLDER, 'PhaseUncert_train.txt']);
phase_uncert_test = load([FOLDER, 'PhaseUncert_test.txt']);
loss_dB_test = load([FOLDER, 'LossdB_test.txt']);
loss_dB_train = load([FOLDER, 'LossdB_train.txt']);
iterations = load([FOLDER, 'ITERATIONS.txt']); % How many times we test the same structure
DATASET_NUM = 0;
end