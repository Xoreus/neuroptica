% Gets the average of multiple simulations
%
% Author: Simon Geoffroy-Gagnon
% Edit: 2020.03.11
clear; close all; clc;

fontsz = 54;
printme = false;

FOLDER = '/home/simon/Documents/neuroptica/tests/Analysis/average-linsep';
Ns = [4, 6, 8, 10, 16];
Ns = [4, 6, 8, 10, 16];

topo = {'R_P', 'C_Q_P', 'E_P', 'R_I_P', 'R_D_I_P', 'R_D_P'};
for ii = 1:length(topo)
    FoM_PT.(topo{ii}) = zeros(length(Ns), 1);
    FoM_LPU.(topo{ii}) = zeros(length(Ns), 1);
    MSE.(topo{ii}) = zeros(length(Ns), 1);
    num.(topo{ii}) = zeros(length(Ns), 1);
end

for jj = 1:length(Ns)
    f = [FOLDER, sprintf('/N=%d',Ns(jj))];
    s = dir(f);
    dirNames = {s.name};
    for ii = 3:length(dirNames)
        [sim, topo] = load_ONN_data([f, '/', dirNames{ii}]);
        for tt = 1:length(topo)
            FoM_PT.(topo{tt})(jj) = FoM_PT.(topo{tt})(jj) + sim.(topo{tt}).PT_FoM;
            FoM_LPU.(topo{tt})(jj) = FoM_LPU.(topo{tt})(jj) + sim.(topo{tt}).LPU_FoM;
            MSE.(topo{tt})(jj) = MSE.(topo{tt})(jj) + sim.(topo{tt}).losses(end);
            num.(topo{tt})(jj) = num.(topo{tt})(jj) + 1;
        end
    end
    fprintf('\tN = %d\n',Ns(jj))
    for tt = 1:length(topo)
        FoM_LPU.(topo{tt})(jj) = FoM_LPU.(topo{tt})(jj)/num.(topo{tt})(jj);
        fprintf('%-10s Loss Phase Uncert Avg FoM:\t %10.4f\n',topo{tt}, FoM_LPU.(topo{tt})(jj))
    end
    fprintf('\n')
    for tt = 1:length(topo)
        FoM_PT.(topo{tt})(jj) = FoM_PT.(topo{tt})(jj)/num.(topo{tt})(jj);
        fprintf('%-10s Phi Theta Avg FoM:\t %10.4f\n',topo{tt}, FoM_PT.(topo{tt})(jj))
    end
    fprintf('\n')
    for tt = 1:length(topo)
        MSE.(topo{tt})(jj) = MSE.(topo{tt})(jj)/num.(topo{tt})(jj);
        fprintf('%-10s Avg final MSE:\t %10.4f\n',topo{tt}, MSE.(topo{tt})(jj))
    end
    fprintf('\n\n')
end

figure('Renderer', 'painters', 'Position', [400 400 1800 1300])
for tt = 1:length(topo)
    plot(Ns,FoM_LPU.(topo{tt}), 'displayName', sim.(sim.topo{tt}).topology, 'linewidth', 3)
    hold on
end
a = get(gca,'YTickLabel');
set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz*0.7)
h = gca;
set(h, 'YTickLabelMode','auto')
set(h, 'XTickLabelMode','auto')
xticks(Ns)
xlabel('Structure Size ($N$)', 'fontsize', fontsz*0.8, 'interpreter','latex')
ylabel('FoM $(\mathrm{Rad} \cdot \mathrm{dB})$', 'fontsize', fontsz*0.8, 'interpreter','latex')
legend('-DynamicLegend', 'fontsize', fontsz*0.8, 'interpreter','latex', 'location', 'northeast');
grid('on')
if printme
    pMe([FOLDER '/FoM_LPU.pdf'])
end

figure('Renderer', 'painters', 'Position', [400 400 1800 1300])
for tt = 1:length(topo)
    plot(Ns,FoM_PT.(topo{tt}), 'displayName', sim.(sim.topo{tt}).topology, 'linewidth', 3)
    hold on
end
a = get(gca,'YTickLabel');
set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz*0.7)
h = gca;
set(h, 'YTickLabelMode','auto')
set(h, 'XTickLabelMode','auto')
xticks(Ns)
xlabel('Structure Size ($N$)', 'fontsize', fontsz*0.8, 'interpreter','latex')
ylabel('FoM $(\mathrm{Rad}^2)$', 'fontsize', fontsz*0.8, 'interpreter','latex')
legend('-DynamicLegend', 'fontsize', fontsz*0.8, 'interpreter','latex', 'location', 'northeast');
grid('on')
if printme
    pMe([FOLDER '/FoM_PT.pdf'])
end


figure('Renderer', 'painters', 'Position', [400 400 1800 1300])
for tt = 1:length(topo)
    plot(Ns,MSE.(topo{tt}), 'displayName', sim.(sim.topo{tt}).topology, 'linewidth', 3)
    hold on
end
legend('-DynamicLegend', 'fontsize', fontsz*0.8, 'interpreter','latex', 'location', 'northwest');
a = get(gca,'YTickLabel');
set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz*0.7)
h = gca;
set(h, 'YTickLabelMode','auto')
set(h, 'XTickLabelMode','auto')
xticks(Ns)
xlabel('Structure Size ($N$)', 'fontsize', fontsz*0.8, 'interpreter','latex')
ylabel('Final $\mathcal{L}_{\mathrm{(MSE)}}$', 'fontsize', fontsz*0.8, 'interpreter','latex')
% axis square
grid('on')
if printme
    pMe([FOLDER '/MSE.pdf'])
end