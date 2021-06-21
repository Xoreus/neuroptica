% Function to load all the phases for a set of R_D_I_P and see the variance
%
% Author: Simon Geoffroy-Gagnon
% Edit: 2020.03.17
close all; clc; clear;

N = 4;
fontsz = 64;
FOLDER = sprintf('/home/edwar/Documents/Github_Projects/neuroptica/tests/Analysis/phaseConvergence_RDP/N=4', N);
phases = {};

comp_layer =  'DMM'; % 'Reck'; %

if strcmp(comp_layer, 'Inv. Reck')
    layer_idx = 3;
elseif strcmp(comp_layer, 'Reck')
    layer_idx = 1;
else
    layer_idx = 2;
end
dirF = dir(FOLDER);
dirNames = {dirF(3:end).name};
for dname = 1:length(dirNames)
    fold = [FOLDER sprintf('/%s/Topologies',dirNames{dname})];
    load([fold, '/R_D_P.mat'])
    phases{end+1} = R_D_P.phases{layer_idx};
end
theta = zeros(length(phases{1}),length(phases));
phi = zeros(length(phases{1}),length(phases));
for ii = 1:length(phases)
    theta(:,ii) = phases{ii}(:,1);
    phi(:,ii) = phases{ii}(:,2);
end
vart = var(theta');
varp = var(phi');

meant = mean(theta');
meanp = mean(phi');
fprintf('N=%d, mean theta = %.3f, var theta = %.3f\n', N, mean(meant), var(meant))

samples_p = meanp + varp.*randn(100000, 1);
samples_t = meant + vart.*randn(100000, 1);
%%
colors = {[0 0 0.5], [0 0.5 0], [0.5 0 0], [0.5 0.5 0.5], [0.25, 0.25, 0],[1,1,0],[1,0,1],[0.25,0.25,0.25]};
xlims = [-15, 15];
fontsz = 64;
lgd = {};
figure('Renderer', 'painters', 'Position', [400 400 1900 1400])
for ii = 1:length(samples_p(1,:))
    h = histfit(samples_p(:,ii));
    set(h(2), 'color', colors{mod(ii, length(colors))+1})
    set(h(2), 'linewidth',3)
    delete(h(1))
    lgd{end+1} = sprintf('%s $\\phi_{%d}$',comp_layer, ii);
    hold on
end
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',fontsz*0.7)
a = get(gca,'YTickLabel');
set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz*0.7)
h = gca;
set(h, 'YTickLabelMode','auto')
set(h, 'XTickLabelMode','auto')
axis square
xlabel('$\phi$ phase (Rad)', 'fontsize', fontsz, 'interpreter','latex')
ylabel('', 'fontsize', fontsz, 'interpreter','latex')
yt = get(gca, 'YTick');
set(gca, 'YTick', yt, 'YTickLabel', yt/numel(samples_t(:,1)))
legend(lgd, 'fontsize', fontsz*.8, 'interpreter','latex', 'location', 'northwest');
yticks('')
xlim(xlims)
pMe_lineplot(sprintf('../Crop_Me/phi_phaseConvergence_%s_N=%d.pdf', comp_layer, N))


figure('Renderer', 'painters', 'Position', [400 400 1900 1400])
lgd = {};
for ii = 1:length(samples_t(1,:))
    h = histfit(samples_t(:,ii));
    set(h(2), 'color', colors{mod(ii, length(colors))+1})
    set(h(2), 'linewidth',3)
    lgd{end+1} = sprintf('%s $\\theta_{%d}$',comp_layer, ii);
    delete(h(1))
    hold on
end
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',fontsz*0.7)
a = get(gca,'YTickLabel');
set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz*0.7)
h = gca;
set(h, 'YTickLabelMode','auto')
set(h, 'XTickLabelMode','auto')
axis square
xlabel('$\theta$ phase (Rad)', 'fontsize', fontsz, 'interpreter','latex')
ylabel('', 'fontsize', fontsz, 'interpreter','latex')
yticks('')

legend(lgd, 'fontsize', fontsz*.8, 'interpreter','latex', 'location', 'northwest');
xlim(xlims)
pMe_lineplot(sprintf('../Crop_Me/theta_phaseConvergence_%s_N=%d.pdf', comp_layer, N))
