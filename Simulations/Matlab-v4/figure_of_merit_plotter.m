
diamond = [0.17438, 0.04641, 0.02072, 0.019844, 0.00984, 0.00611]; % Loss-PU
reck =    [0.10625, 0.03938, 0.01556, 0.011094, 0.00391, 0.00250]; % Loss-PU
N =       [4,       8,       12,      16,      32,      64];

diamond = [0.17438, 0.04641, 0.02072, 0.0156,  0.00984, 0.00611]; % Loss-PU
reck =    [0.10625, 0.03938, 0.01556, 0.0102,  0.00391, 0.00250]; % Loss-PU
N =       [4,       8,       12,      16,      32,      64];

close all
figure('Renderer', 'painters', 'Position', [400 400 1800 1300])

% plot(N, diamond, 'linewidth', 3)
% hold on
% plot(N, reck, 'linewidth', 3)


plot(N, reck, 'r-v', 'markersize', 20, 'MarkerFaceColor', '#c3c3c3','linewidth', 3)
hold on
plot(N, diamond, 'b-d', 'markersize', 20, 'MarkerFaceColor', 'k', 'linewidth', 3)

fontsz = 64;

% title('Figure of Merit', 'fontsize', fontsz, 'interpreter','latex');
lgd = {'Reck', 'Diamond'};
legend(lgd, 'fontsize', fontsz, 'interpreter','latex');

a = get(gca,'YTickLabel');
set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz*0.7)
h = gca;
set(h, 'YTickLabelMode','auto')
set(h, 'XTickLabelMode','auto')
xticks(N)
xlabel('Structure Size ($N$)', 'fontsize', fontsz, 'interpreter','latex')
ylabel('FoM $(\mathrm{Rad} \cdot \mathrm{dB})$', 'fontsize', fontsz, 'interpreter','latex')
% axis square
grid('on')
pMe_lineplot('/home/edwar/Documents/Github_Projects/neuroptica/tests/Crop_Me/FoM.pdf')
%%
diamond_bp  = [0.1927, 0.2067, 0.2086, 0.1948, 0.1530, 0.1848];
reck_bp     = [0.1935, 0.2276, 0.3572, 0.6381, 1.2113, 2.4012];
clements_bp = [0.1784, 0.2572, 0.3388, 0.6363, 1.1830, 2.3933];
N =           [4,      8,      12,     16,     32,     64];
figure('Renderer', 'painters', 'Position', [400 400 1800 1300])

plot(N, reck_bp, 'r-v', 'markersize', 20, 'MarkerFaceColor', '#c3c3c3','linewidth', 3)
hold on
plot(N, diamond_bp, 'b-d', 'markersize', 20, 'MarkerFaceColor', 'k', 'linewidth', 3)

% plot(N, clements_bp, 'b-*', 'markersize', 20, 'MarkerFaceColor', 'k', 'linewidth', 3)
fontsz = 64;

% title('Figure of Merit', 'fontsize', fontsz, 'interpreter','latex');
lgd = {'Reck', 'Diamond'};
legend(lgd, 'fontsize', fontsz, 'interpreter','latex', 'location', 'northwest');

a = get(gca,'YTickLabel');
set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz)
h = gca;
set(h, 'YTickLabelMode','auto')
set(h, 'XTickLabelMode','auto')
xticks(N)
xlabel('Structure Size ($N$)', 'fontsize', fontsz, 'interpreter','latex')
ylabel('Final $\mathcal{L}_{\mathrm{(MSE)}}$', 'fontsize', fontsz, 'interpreter','latex')

% axis square
grid('on')
pMe_lineplot('/home/edwar/Documents/Github_Projects/neuroptica/tests/Crop_Me/MSE-N.pdf')

%%
