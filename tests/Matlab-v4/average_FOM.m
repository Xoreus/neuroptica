% Plot % difference in port pwer output for meshes of different sizes
%
% Author: Simon Geoffroy-Gagnon
% Edit: 2020.03.20
% clc; clear; close all;
close all; clc; clear;

printMe = false;

linewid = 3;
fontsz = 64;
newPaper = false;
sigma = 0;
topo = {'R_P', 'C_Q_P', 'E_P', 'R_I_P'};% , 'R_D_I_P', 'R_D_P'};
topos = [1,3,2,4];
topology = {'Reck','Diamond','Clements','Reck + Inv. Reck'};

for lossy = [0]
    Ns = [4,6,8,10,12,16,32,64];
    for name_idx = 1:length(topo)
        FoM_PT.(topo{name_idx}) = zeros(length(Ns), 1);
        FoM_LPU.(topo{name_idx}) = zeros(length(Ns), 1);
        MSE.(topo{name_idx}) = zeros(length(Ns), 1);
        num.(topo{name_idx}) = zeros(length(Ns), 1);
    end
    
    for jj = 1:length(Ns)
        F = sprintf('/home/simon/Documents/neuroptica/tests/Analysis/%.2f_sigma%.2f_outPorts_mean_pow', lossy, sigma);
        folders = dir([F sprintf('/N=%d_*', Ns(jj))]);
        foldNames = {folders.name};
        
        for n = 1:length(foldNames)
            dirFiles = dir([F '/' foldNames{n} '/Topologies/']);
            dirNames = {dirFiles(3:end).name};
            if length(dirNames) == 6
                dirNames = dirNames([1,2,5,6]);
            end
            for name_idx = 1:length(dirNames)
                sim = load([F '/' foldNames{n} '/Topologies/' dirNames{name_idx}]);
                FoM_PT.(dirNames{name_idx}(1:end-4))(jj) = FoM_PT.(dirNames{name_idx}(1:end-4))(jj) + sim.(dirNames{name_idx}(1:end-4)).PT_FoM;
                FoM_LPU.(dirNames{name_idx}(1:end-4))(jj) = FoM_LPU.(dirNames{name_idx}(1:end-4))(jj) + sim.(dirNames{name_idx}(1:end-4)).LPU_FoM;
                MSE.(dirNames{name_idx}(1:end-4))(jj) = MSE.(dirNames{name_idx}(1:end-4))(jj) + sim.(dirNames{name_idx}(1:end-4)).losses(end);
                num.(dirNames{name_idx}(1:end-4))(jj) = num.(dirNames{name_idx}(1:end-4))(jj) + 1;
            end
        end
        
        lgd = {};
        
        for name_idx = 1:length(dirNames)
            sim = load([F '/' foldNames{n} '/Topologies/' dirNames{name_idx}]);
            lgd{end+1} = sim.(dirNames{name_idx}(1:end-4)).topology;
        end
        
        for name_idx = 1:length(topo)
            FoM_PT.(topo{name_idx}(1:end))(jj) = FoM_PT.(topo{name_idx}(1:end))(jj)/num.(topo{name_idx}(1:end))(jj);
            FoM_LPU.(topo{name_idx}(1:end))(jj) = FoM_LPU.(topo{name_idx}(1:end))(jj)/num.(topo{name_idx}(1:end))(jj);
            MSE.(topo{name_idx}(1:end))(jj) = MSE.(topo{name_idx}(1:end))(jj)/num.(topo{name_idx}(1:end))(jj);
        end
        
    end
    
    
    markers = {'-v','-d','-s','-o'};
    linecolor = [[0.5,0,0],[0,0.5,0],[0,0,0.5],[0.5,0.5,0]];
    figure('Renderer', 'painters', 'Position', [400 400 1800 1300])
    for tt = topos
        
        plot(Ns([1:3,5:end]),FoM_LPU.(topo{tt})([1:3,5:end]), markers{tt}, 'markersize', 20, 'MarkerFaceColor', '#c3c3c3', ...
            'displayName', topology{tt}, 'linewidth', 3)
        hold on
        
    end
    a = get(gca,'YTickLabel');
    set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz*0.8)
    h = gca;
    set(h, 'YTickLabelMode','auto')
    set(h, 'XTickLabelMode','auto')
    xticks(Ns([1,3,5:end]))
    set(gca, 'YScale', 'log')
    xlabel('Structure Size ($N$)', 'fontsize', fontsz, 'interpreter','latex')
    ylabel('FoM $(\mathrm{Rad} \cdot \mathrm{dB})$', 'fontsize', fontsz, 'interpreter','latex')
    legend('-DynamicLegend', 'fontsize', fontsz, 'interpreter','latex', 'location', 'northeast');
    set(gca, 'YGrid', 'off', 'XGrid', 'on')
%     set(gca, 'YScale', 'log')
    h = gca;
    set(h, 'YTickLabelMode','auto')
    set(h, 'XTickLabelMode','auto')
    xticks([4, 8, 12, 16, 32, 64])
    if printme && 1
        pMe(sprintf('../Crop_Me/FoM_LPU%s.pdf', errBar))
    end

end