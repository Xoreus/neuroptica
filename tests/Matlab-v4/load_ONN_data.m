% Get all data from single or multiple loss python ONN simulation
%
% Author: Simon Geoffroy-Gagnon
% Edit: 25.01.2020

function [sim, topo] = load_ONN_data(FOLDER)
fid = fopen([FOLDER, '/all_topologies.txt']);
topo = textscan(fid, '%s');
accuracy = cell(length(topo{1}), 1);
for ii = 1:length(topo{1})
    simulation = load([FOLDER, '/Topologies/', topo{1}{ii}]);
    sim.(topo{1}{ii}) = simulation.(topo{1}{ii});
    acc = [sim.(topo{1}{ii}).accuracy_LPU, sim.(topo{1}{ii}).accuracy_PT];
    accuracy{ii} = acc;
end
for ii = 1:length(topo{1})
    filename = [FOLDER, '/Data_Fitting/', topo{1}{ii}, '.txt'];
    delimiterIn = ',';
    headerlinesIn = 1;
    bp = importdata(filename,delimiterIn,headerlinesIn);
    
    sim.(topo{1}{ii}).losses = bp.data(:,2);
    sim.(topo{1}{ii}).trn_accuracy = bp.data(:,3);
    sim.(topo{1}{ii}).val_accuracy = bp.data(:,4);

end

for ii = 1:length(accuracy)
    maxAcc(ii) = max(max(max(accuracy{ii})));
end
sim.max_accuracy = max(maxAcc);
topo = topo{1};
end