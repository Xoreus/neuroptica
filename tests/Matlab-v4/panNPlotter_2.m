% Function to plot all the pan-N plots
% like Figure of Merit and final MSE loss
%
% Author: Simon Geoffroy-Gagnon
% Edit: 2020.03.10
close all;clear;clc

F = '/home/simon/Documents/neuroptica/tests/Analysis/linsep-Thesis';
F = '/home/simon/Documents/neuroptica/tests/Analysis/linsep_final';
% F = '/home/simon/Documents/neuroptica/tests/Analysis/newPaperLinSep';
% Ns = [4,8,16,32,64];

Ns = [4,6,8,10,12,14,16,20,24,28,32];
Ns = [4,6,8,10,12,16,20,28,32];


fig_of_merit_value = 0.75;

f = sprintf('/N=%d_1/', Ns(1));
f = sprintf('/N=%d_0/', Ns(1));
FOLDER = [F, f];
[sim, topo] = load_ONN_data(FOLDER);

c = cell(length(topo),1);
FoM_PT = cell2struct(c, topo);
FoM_LPU = cell2struct(c, topo);
MSE = cell2struct(c, topo);

for N = Ns
    f = sprintf('/N=%d_0/', N);
    FOLDER = [F, f];
    [sim, topo] = load_ONN_data(FOLDER);
    for tt = 1:length(sim.topo{1})
        PT = sim.(sim.topo{1}{tt}).accuracy_PT;
        LPU = sim.(sim.topo{1}{tt}).accuracy_LPU;
        
        % Calculate "area" of contour map as a figure of merit
        sim.(sim.topo{1}{tt}).FoM_PT = sum(sum(PT >= sim.max_accuracy*fig_of_merit_value)) * (sim.(sim.topo{1}{tt}).phase_uncert_phi(2) - ...
            sim.(sim.topo{1}{tt}).phase_uncert_phi(1)) * (sim.(sim.topo{1}{tt}).phase_uncert_theta(2) - sim.(sim.topo{1}{tt}).phase_uncert_theta(1));
        
        FoM_PT.(sim.topo{1}{tt})(end+1) = sim.(sim.topo{1}{tt}).FoM_PT;
        
        % Calculate "area" of contour map as a figure of merit
        sim.(sim.topo{1}{tt}).FoM_LPU = sum(sum(LPU >= sim.max_accuracy*fig_of_merit_value)) * (sim.(sim.topo{1}{tt}).phase_uncert_phi(2) - ...
            sim.(sim.topo{1}{tt}).phase_uncert_phi(1)) * (sim.(sim.topo{1}{tt}).phase_uncert_theta(2) - sim.(sim.topo{1}{tt}).phase_uncert_theta(1));
        
        FoM_LPU.(sim.topo{1}{tt})(end+1) = sim.(sim.topo{1}{tt}).FoM_LPU;
        
        MSE.(sim.topo{1}{tt})(end+1) = sim.(sim.topo{1}{tt}).losses(end);
    end
end

fontsz = 64;

figure('Renderer', 'painters', 'Position', [400 400 1900 1400])
for tt = 1:length(sim.topo{1})
        plot(Ns, FoM_LPU.(sim.topo{1}{tt}), '-s', 'markersize', 20, 'MarkerFaceColor', ...
        '#c3c3c3','linewidth', 3, 'displayName', sim.(sim.topo{1}{tt}).topology)
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