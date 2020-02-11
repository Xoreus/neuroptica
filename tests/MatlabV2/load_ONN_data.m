% Get all data from single or multiple loss python ONN simulation
%
% Author: Simon Geoffroy-Gagnon
% Edit: 25.01.2020

function allTopo = load_ONN_data(FOLDER)

allTopo = load([FOLDER, 'allTopologies.mat']);
allTopo = allTopo.allTopologies;
allTopo.max_accuracy = max(max(max(max(allTopo.accuracy))));
end