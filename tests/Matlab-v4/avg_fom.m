% Gets the average of multiple simulations
%
% Author: Simon Geoffroy-Gagnon
% Edit: 2020.03.11
clear; close all; clc;

fontsz = 64;
printme = 1;

FOLDER = '/home/simon/Documents/neuroptica/tests/Analysis/average_linsep_NoDMM';
FOLDER = '/home/simon/Documents/neuroptica/tests/Analysis/Thesis_Sims/average_linsep_NoDMM_NoJump';
Ns = [4, 6, 8, 10, 16, 24, 32, 48, 64, 80, 96];

topo = {'R_P','R_I_P',  'E_P', 'C_Q_P',};% , 'R_D_I_P', 'R_D_P'};
topology = {'Reck','Reck + Inv. Reck','Clements','Diamond'};% ,'Reck + DMM + Inv. Reck','Reck + DMM'};
errBar = '';

for ii = 1:length(topo)
    FoM_PT.(topo{ii}) = zeros(length(Ns), 1);
    FoM_LPU.(topo{ii}) = zeros(length(Ns), 1);
    MSE.(topo{ii}) = zeros(length(Ns), 1);
    num.(topo{ii}) = zeros(length(Ns), 1);
    
    scat_pt.(topo{ii}) = cell(length(Ns), 1);
    scat_lpu.(topo{ii}) = cell(length(Ns), 1);
    scat_mse.(topo{ii}) = cell(length(Ns),1);
end

for jj = 1:length(Ns)
    f = [FOLDER, sprintf('/N=%d',Ns(jj))];
    s = dir(f);
    dirNames = {s.name};
    for ii = 3:length(dirNames)
        [sim, topo_cur] = load_ONN_data([f, '/', dirNames{ii}]);
        
        for tt = 1:length(topo_cur)
            pt_area = (sim.(topo_cur{tt}).phase_uncert_phi(2) - sim.(topo_cur{tt}).phase_uncert_phi(1))^2;
            lpu_area = (sim.(topo_cur{tt}).phase_uncert_phi(2) - sim.(topo_cur{tt}).phase_uncert_phi(1)) * ...
                (sim.(topo_cur{tt}).loss_dB(2) - sim.(topo_cur{tt}).loss_dB(1));
            FoM_PT.(topo_cur{tt})(jj) = FoM_PT.(topo_cur{tt})(jj) + sum(sum(sim.(topo_cur{tt}).accuracy_PT > 0.75*max(max(sim.(topo_cur{tt}).accuracy_PT))))*pt_area;
            FoM_LPU.(topo_cur{tt})(jj) = FoM_LPU.(topo_cur{tt})(jj) + sum(sum(sim.(topo_cur{tt}).accuracy_LPU > 0.75*max(max(sim.(topo_cur{tt}).accuracy_LPU))))*lpu_area;
            MSE.(topo_cur{tt})(jj) = MSE.(topo_cur{tt})(jj) + sim.(topo_cur{tt}).losses(end);
            num.(topo_cur{tt})(jj) = num.(topo_cur{tt})(jj) + 1;
        end
    end
    for tt = 1:length(topo)
        FoM_LPU.(topo{tt})(jj) = FoM_LPU.(topo{tt})(jj)/num.(topo{tt})(jj);
    end
    for tt = 1:length(topo)
        FoM_PT.(topo{tt})(jj) = FoM_PT.(topo{tt})(jj)/num.(topo{tt})(jj);
    end
    for tt = 1:length(topo)
        MSE.(topo{tt})(jj) = MSE.(topo{tt})(jj)/num.(topo{tt})(jj);
    end
end


markers = {'-v','-o','-s','-d' };
linecolor = [[0.5,0,0],[0,0.5,0],[0,0,0.5],[0.5,0.5,0]];
figure('Renderer', 'painters', 'Position', [400 400 1800 1300])
for tt = 1:length(topo)
    if isempty(errBar)
    plot(Ns,FoM_LPU.(topo{tt}), markers{tt}, 'markersize', 20, 'MarkerFaceColor', '#c3c3c3', ...
        'displayName', topology{tt}, 'linewidth', 3)
    else
    errorbar(Ns,FoM_LPU.(topo{tt}), FoM_LPU_var.(topo{tt}), 'displayName', topology{tt}, 'linewidth', 3)
    end
    hold on
    
end
a = get(gca,'YTickLabel');
set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz*0.7)
h = gca;
set(h, 'YTickLabelMode','auto')
set(h, 'XTickLabelMode','auto')
xticks(Ns([1, 4:end]))
set(gca, 'YScale', 'log')
xlabel('Structure Size ($N$)', 'fontsize', fontsz*0.8, 'interpreter','latex')
ylabel('FoM $(\mathrm{Rad} \cdot \mathrm{dB})$', 'fontsize', fontsz*0.8, 'interpreter','latex')
legend('-DynamicLegend', 'fontsize', fontsz*0.7, 'interpreter','latex', 'location', 'northeast');
set(gca, 'YGrid', 'off', 'XGrid', 'on')

if printme && 1
    pMe_lineplot(['../Crop_Me' sprintf('/FoM_LPU%s.pdf', errBar)])
end

figure('Renderer', 'painters', 'Position', [400 400 1800 1300])
for tt = 1:length(topo)
    if isempty(errBar)
    plot(Ns,FoM_PT.(topo{tt}),  markers{tt},  'markersize', 20, 'MarkerFaceColor', '#c3c3c3', ...
        'displayName', topology{tt}, 'linewidth', 3)
    else
    errorbar(Ns,FoM_PT.(topo{tt}),FoM_PT_var.(topo{tt}), ...
        'displayName', topology{tt}, 'linewidth', 3)
    end
    hold on
end
a = get(gca,'YTickLabel');
set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz*0.7)
h = gca;
set(h, 'YTickLabelMode','auto')
set(h, 'XTickLabelMode','auto')
xticks(Ns([1, 4:end]))
xlabel('Structure Size ($N$)', 'fontsize', fontsz*0.8, 'interpreter','latex')
ylabel('FoM $(\mathrm{Rad}^2)$', 'fontsize', fontsz*0.8, 'interpreter','latex')
set(gca, 'YScale', 'log')
set(gca, 'YGrid', 'off', 'XGrid', 'on')
legend('-DynamicLegend', 'fontsize', fontsz*0.7, 'interpreter','latex', 'location', 'northeast');
if printme && 1
    pMe_lineplot(['../Crop_Me' sprintf('/FoM_PT%s.pdf', errBar)])
end


figure('Renderer', 'painters', 'Position', [400 400 1800 1300])
for tt = 1:length(topo)
    if isempty(errBar)
    plot(Ns,MSE.(topo{tt}), markers{tt}, 'markersize', 20, 'MarkerFaceColor', '#c3c3c3', ...
        'displayName', topology{tt}, 'linewidth', 3)
    else
    errorbar(Ns,MSE.(topo{tt}),MSE_var.(topo{tt}),...
        'displayName', topology{tt}, 'linewidth', 3)
    end
    hold on
end
legend('-DynamicLegend', 'fontsize', fontsz*0.7, 'interpreter','latex', 'location', 'northwest');
a = get(gca,'YTickLabel');
set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz*0.7)
h = gca;
set(h, 'YTickLabelMode','auto')
set(h, 'XTickLabelMode','auto')
xticks(Ns([1, 4:end]))
xlabel('Structure Size ($N$)', 'fontsize', fontsz*0.8, 'interpreter','latex')
ylabel('Final $\mathcal{L}_{\mathrm{(MSE)}}$', 'fontsize', fontsz*0.8, 'interpreter','latex')
% axis square
grid('on')
if printme && 1
    pMe_lineplot(['../Crop_Me' sprintf('/MSE%s.pdf', errBar)])
end