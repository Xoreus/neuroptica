% Function to plot all the pan-N plots
% like Figure of Merit and final MSE loss
%
% Author: Simon Geoffroy-Gagnon
% Edit: 2020.03.09
close all;clear;clc

F = '/home/simon/Documents/neuroptica/tests/Analysis/linsep-Thesis';
F = '/home/simon/Documents/neuroptica/tests/Analysis/linsep_final';

Ns = [4, 6, 8, 10, 12, 16, 32, 64];
fig_of_merit_value = 0.75;

f = sprintf('/N=%d_0/', Ns(1));
FOLDER = [F, f];
[sim, topo] = load_ONN_data(FOLDER);

c = cell(length(topo),1);
FoM_PT = cell2struct(c, topo);
FoM_LPU = cell2struct(c, topo);
MSE = cell2struct(c, topo);

for jj = 1:length(Ns)
    switch Ns(jj)
        case 4
            datasets = 0:119;
        case 6
            datasets = 0:119;
        case 8
            datasets = 0:119;
        case 10
            datasets = 0:9;
        case 12
            datasets = 0:9;
        case 16
            datassets = [0, 1, 10:16];
        case 32
            datasets = 0:1;
        case 64
            datasets = 0;
    end
    for ii = datasets
        f = sprintf('/N=%d_%d/', Ns(jj), ii);
        FOLDER = [F, f];
        [sim, topo] = load_ONN_data(FOLDER);
        for tt = 1:length(sim.topo)
            PT = sim.(sim.topo{tt}).accuracy_PT;
            LPU = sim.(sim.topo{tt}).accuracy_LPU;
            
            % Calculate "area" of contour map as a figure of merit
            sim.(sim.topo{tt}).FoM_PT = sum(sum(PT >= sim.max_accuracy*fig_of_merit_value)) * (sim.(sim.topo{tt}).phase_uncert_phi(2) - ...
                sim.(sim.topo{tt}).phase_uncert_phi(1)) * (sim.(sim.topo{tt}).phase_uncert_theta(2) - sim.(sim.topo{tt}).phase_uncert_theta(1));
            
            % Calculate "area" of contour map as a figure of merit
            sim.(sim.topo{tt}).FoM_LPU = sum(sum(LPU >= sim.max_accuracy*fig_of_merit_value)) * (sim.(sim.topo{tt}).phase_uncert_phi(2) - ...
                sim.(sim.topo{tt}).phase_uncert_phi(1)) * (sim.(sim.topo{tt}).phase_uncert_theta(2) - sim.(sim.topo{tt}).phase_uncert_theta(1));
            
            
            FoM_PT.(sim.topo{tt})(jj) = FoM_PT.(sim.topo{tt})(jj) + sim.(sim.topo{tt}).FoM_PT;
            
            FoM_LPU.(sim.topo{tt})(jj) = FoM_LPU.(sim.topo{tt})(jj) + sim.(sim.topo{tt}).FoM_LPU;
            
            MSE.(sim.topo{tt})(jj) = MSE.(sim.topo{tt})(jj) + sim.(sim.topo{tt}).losses(end);
        end
    end
    MSE.(sim.topo{tt})(jj) = MSE.(sim.topo{tt})(jj)/length(datasets);
    FoM_LPU.(sim.topo{tt})(jj) = FoM_LPU.(sim.topo{tt})(jj)/length(datasets);
    FoM_PT.(sim.topo{tt})(jj) = FoM_PT.(sim.topo{tt})(jj)/length(datasets);
end
fontsz = 64;

linecolor = {[0, 0.5, 0],[0, 0, 0.5],[0.5,0,0]};
linestyle = {'-v','-d','-s'};
markercolor = {'#c3c3c3','k','#888888'};
figure('Renderer', 'painters', 'Position', [400 400 1900 1400])
for tt = 1:length(sim.topo{1})
    plot(Ns, FoM_LPU.(sim.topo{1}{tt}), linestyle{tt}, 'color',linecolor{tt}, 'markersize', 20, 'MarkerFaceColor', ...
        markercolor{tt},'linewidth', 3, 'displayName', sim.(sim.topo{1}{tt}).topology)
    hold on
end
legend('-DynamicLegend', 'fontsize', fontsz*0.8, 'interpreter','latex', 'location', 'northeast');
a = get(gca,'YTickLabel');
set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz*0.7)
h = gca;
set(h, 'YTickLabelMode','auto')
set(h, 'XTickLabelMode','auto')
xticks(Ns)
xlabel('Structure Size ($N$)', 'fontsize', fontsz*0.8, 'interpreter','latex')
ylabel('FoM $\sigma, Loss/MZI$', 'fontsize', fontsz*0.8, 'interpreter','latex')
set(gca, 'YScale', 'log')

linecolor = {[0, 0.5, 0],[0, 0, 0.5],[0.5,0,0]};
linestyle = {'-v','-d','-s'};
markercolor = {'#c3c3c3','k','#888888'};
figure('Renderer', 'painters', 'Position', [400 400 1900 1400])
for tt = 1:length(sim.topo{1})
    plot(Ns, MSE.(sim.topo{1}{tt}), linestyle{tt}, 'color',linecolor{tt}, 'markersize', 20, 'MarkerFaceColor', ...
        markercolor{tt},'linewidth', 3, 'displayName', sim.(sim.topo{1}{tt}).topology)
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
ylabel('Final Loss Function (MSE)', 'fontsize', fontsz*0.8, 'interpreter','latex')

linecolor = {[0, 0.5, 0],[0, 0, 0.5],[0.5,0,0]};
linestyle = {'-v','-d','-s'};
markercolor = {'#c3c3c3','k','#888888'};
figure('Renderer', 'painters', 'Position', [400 400 1900 1400])
for tt = 1:length(sim.topo{1})
    plot(Ns, FoM_PT.(sim.topo{1}{tt}), linestyle{tt}, 'color',linecolor{tt}, 'markersize', 20, 'MarkerFaceColor', ...
        markercolor{tt},'linewidth', 3, 'displayName', sim.(sim.topo{1}{tt}).topology)
    hold on
end
legend('-DynamicLegend', 'fontsize', fontsz*0.8, 'interpreter','latex', 'location', 'northeast');
a = get(gca,'YTickLabel');
set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz*0.7)
h = gca;
set(h, 'YTickLabelMode','auto')
set(h, 'XTickLabelMode','auto')
xticks(Ns)
ylabel('FoM ($\sigma_\theta = \sigma_\phi$) (Rad)', 'fontsize', fontsz, 'interpreter','latex')
% ylabel('$\sigma_\phi$ (Rad)', 'fontsize', fontsz, 'interpreter','latex')
set(gca, 'YScale', 'log')