% Get all data from single or multiple loss python ONN simulation
%
% Author: Simon Geoffroy-Gagnon
% Edit: 25.01.2020

function acc = load_ONN_data(FOLDER)
fid = fopen([FOLDER, 'all_topologies.txt']);
allTopo = textscan(fid, '%s');
accuracy = {};
for ii = 1:length(allTopo{1})
    acc.(allTopo{1}{ii}) = load([FOLDER, sprintf('acc_%s_loss=0.000_uncert=0.000_4Feat.mat', allTopo{1}{ii})]);
    acc.([allTopo{1}{ii}, '_Simulation']) = load([FOLDER, '/Topologies/', allTopo{1}{ii}]);
    accuracy{end + 1} = acc.(allTopo{1}{ii}).accuracy;
end
for ii = 1:length(accuracy)
    maxAcc(ii) = max(max(max(accuracy{ii})));
end
acc.max_accuracy = max(maxAcc);

end