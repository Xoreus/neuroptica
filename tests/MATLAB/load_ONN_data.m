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

loss_dB = load([FOLDER, 'loss_dB.txt']);
phase_uncert_theta = load([FOLDER, 'phase_uncert_theta.txt']);
phase_uncert_phi = load([FOLDER, 'phase_uncert_phi.txt']);

SimulationSettings.loss_dB = loss_dB;
SimulationSettings.phase_uncert_theta = phase_uncert_theta;
SimulationSettings.phase_uncert_phi = phase_uncert_phi;

fid = fopen([FOLDER, 'ONN_Setups.txt'], 'rt');
ONN_Setups = textscan(fid, '%s', 'Headerlines', 0);
fclose(fid);
SimulationSettings.ONN_Setups = ONN_Setups{1};

models = get_model_names(SimulationSettings.ONN_Setups);
SimulationSettings.models = models;
end