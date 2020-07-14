% Get all data from single or multiple loss python ONN simulation
%
% Author: Simon Geoffroy-Gagnon
% Edit: 25.01.2020

function [sim, model_names] = load_ONN_data(FOLDER)
model_names = dir([FOLDER, '/Topologies/']);
model_names = {model_names.name};
model_names = model_names(3:end);
accuracy = cell(length(model_names),1);
for ii = 1:length(model_names)
    simulation = load([FOLDER, '/Topologies/', model_names{ii}]);
    sim.(model_names{ii}(1:end-4)) = simulation.(model_names{ii}(1:end-4));
    acc = max(sim.(model_names{ii}(1:end-4)).accuracy_LPU(1,1), sim.(model_names{ii}(1:end-4)).accuracy_PT(1,1));
    accuracy{ii} = acc;
end

for ii = 1:length(model_names)
    filename = [FOLDER, '/', model_names{ii}(1:end-4), '.txt'];
    delimiterIn = ',';
    headerlinesIn = 1;
    if exist(filename, 'file')
        bp = importdata(filename,delimiterIn,headerlinesIn);
        sim.(model_names{ii}(1:end-4)).losses = bp.data(:,2);
        sim.(model_names{ii}(1:end-4)).trn_accuracy = bp.data(:,3);
        sim.(model_names{ii}(1:end-4)).val_accuracy = bp.data(:,4);
    end
end

for ii = 1:length(accuracy)
    maxAcc(ii) = max(max(max(accuracy{ii})));
end

sim.max_accuracy = max(maxAcc);
model_names = cellfun(@(x) x(1:end-4), model_names, 'UniformOutput', false);
sim.topo = model_names;
end