% Get all data from single or multiple loss python ONN simulation
%
% Author: Simon Geoffroy-Gagnon
% Edit: 25.01.2020

function [SimulationSettings] = load_ONN_data(FOLDER)
% Get nonlin, Models, Phase Uncert, loss, iterations, N

fid = fopen([FOLDER, 'SimulationSettings.txt'],'rt');
tmp = textscan(fid, '%s %s', 'Headerlines', 1);
fclose(fid);

for ii = 1:length(tmp{1})
    SimulationSettings.(tmp{1}{ii}) = tmp{2}{ii};
end

fold = dir([FOLDER, '*_train.txt']);

if ~isempty(fold)
    losses_dB_train = load([FOLDER, 'losses_dB_train.txt']);
    losses_dB_test = load([FOLDER, 'losses_dB_test.txt']);
    
    phase_uncerts_train = load([FOLDER, 'phase_uncerts_train.txt']);
    phase_uncerts_test = load([FOLDER, 'phase_uncerts_test.txt']);
    
    SimulationSettings.losses_dB_train = losses_dB_train;
    SimulationSettings.losses_dB_test = losses_dB_test;
    
    SimulationSettings.phase_uncerts_train = phase_uncerts_train;
    SimulationSettings.phase_uncerts_test = phase_uncerts_test;
else
    loss_dB = load([FOLDER, 'loss_dB.txt']);
    phase_uncert = load([FOLDER, 'phase_uncert.txt']);
    
    SimulationSettings.loss_dB = loss_dB;
    SimulationSettings.phase_uncerts = phase_uncert;
end

fid = fopen([FOLDER, 'ONN_Setups.txt'], 'rt');
ONN_Setups = textscan(fid, '%s', 'Headerlines', 0);
fclose(fid);
SimulationSettings.ONN_Setups = ONN_Setups{1};

models = get_model_names(SimulationSettings.ONN_Setups);
SimulationSettings.models = models;
end