% Get all data from single or multiple loss python ONN simulation
%
% Author: Simon Geoffroy-Gagnon
% Edit: 25.01.2020

function [acc, sim, topo] = load_ONN_data(FOLDER)
fid = fopen([FOLDER, 'all_topologies.txt']);
topo = textscan(fid, '%s');
accuracy = {};
for ii = 1:length(topo{1})
    acc.(topo{1}{ii}) = load([FOLDER, sprintf('acc_%s_loss=0.000_uncert=0.000_4Feat.mat', topo{1}{ii})]);
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