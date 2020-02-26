% Get all data from single or multiple loss python ONN simulation
%
% Author: Simon Geoffroy-Gagnon
% Edit: 25.01.2020

function [acc, sim, topo] = load_ONN_data(FOLDER, N, loss_dB)
fid = fopen([FOLDER, 'all_topologies.txt']);
topo = textscan(fid, '%s');
accuracy = {};
for ii = 1:length(topo{1})
    acc.(topo{1}{ii}) = load([FOLDER, sprintf('acc_%s_loss=%.3f_uncert=0.000_%dFeat.mat', topo{1}{ii}, loss_dB, N)]);
    simulation = load([FOLDER, '/Topologies/', topo{1}{ii}]);
    sim.(topo{1}{ii}) = simulation.(topo{1}{ii});
    accuracy{end + 1} = acc.(topo{1}{ii}).accuracy;
end
for ii = 1:length(accuracy)
    maxAcc(ii) = max(max(max(accuracy{ii})));
end
acc.max_accuracy = max(maxAcc);
topo = topo{1};
end