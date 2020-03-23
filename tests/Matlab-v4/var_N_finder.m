% Finds the variance of all the different N values
%
% Author: Simon Geoffroy-Gagnon
% Edit: 2020.03.19

% Gets the average of multiple simulations
%
% Author: Simon Geoffroy-Gagnon
% Edit: 2020.03.11
clear; close all; clc;

fontsz = 64;
printme = true;

FOLDER = '/home/simon/Documents/neuroptica/tests/Analysis/average-linsep-NoDMM';
Ns = [4, 6, 8, 10, 16, 24, 32, 48, 64];

topo = {'R_P', 'C_Q_P', 'E_P', 'R_I_P'};% , 'R_D_I_P', 'R_D_P'};
topology = {'Reck','Diamond','Clements','Reck + Inv. Reck'};% ,'Reck + DMM + Inv. Reck','Reck + DMM'};
for ii = 1:length(topo)
    FoM_PT_var.(topo{ii}) = cell(length(Ns), 1);
    FoM_LPU_var.(topo{ii}) = cell(length(Ns), 1);
    MSE_var.(topo{ii}) = cell(length(Ns), 1);
end

for jj = 1:length(Ns)
    f = [FOLDER, sprintf('/N=%d',Ns(jj))];
    s = dir(f);
    dirNames = {s.name};
    for ii = 3:length(dirNames)
        [sim, topo_cur] = load_ONN_data([f, '/', dirNames{ii}]);
        for tt = 1:length(topo_cur)
            FoM_PT_var.(topo_cur{tt}){jj}( end+1) = sim.(topo_cur{tt}).PT_FoM;
            FoM_LPU_var.(topo_cur{tt}){jj}(end+1) = sim.(topo_cur{tt}).LPU_FoM;
            MSE_var.(topo_cur{tt}){jj}(end+1) =  sim.(topo_cur{tt}).losses(end);
        end
    end
end

for jj = 1:length(Ns)
    for tt = 1:length(topo)
        FoM_LPU_var.(topo{tt}){jj} = sqrt(var(FoM_LPU_var.(topo{tt}){jj}));
    end
    for tt = 1:length(topo)
        FoM_PT_var.(topo{tt}){jj} = sqrt(var(FoM_PT_var.(topo{tt}){jj}));
    end
    for tt = 1:length(topo)
        MSE_var.(topo{tt}){jj} = sqrt(var(MSE_var.(topo{tt}){jj}));
    end
end

for tt = 1:length(topo)
    FoM_PT_var.(topo{tt}) = cell2mat( FoM_PT_var.(topo{tt}));
    FoM_LPU_var.(topo{tt}) = cell2mat( FoM_LPU_var.(topo{tt}));
    MSE_var.(topo{tt}) = cell2mat( MSE_var.(topo{tt}));
end
