diamond = [0.315625, 0.0675, 0.022813, 0.010781]; % Phi Theta
reck = [0.22125, 0.045781, 0.020625, 0.008281]; % Phi Theta



diamond = [0.17438, 0.04641, 0.05361, 0.01972, 0.01984, 0.00984]; % Loss-PU
reck =    [0.10625, 0.03938, 0.03972, 0.01556, 0.01109, 0.00391]; % Loss-PU
N =       [4,       8,       10,      12,      16,      32];


diamond = [0.17438, 0.04641, 0.01972, 0.0247, 0.01984, 0.00984]; % Loss-PU
reck =    [0.10625, 0.03938, 0.01556, 0.0206, 0.01109, 0.00391]; % Loss-PU
N =       [4,       8,       12,      14,     16,      32];

diamond = [0.17438, 0.04641, 0.02072, 0.01984, 0.00984, 0.00611]; % Loss-PU
reck =    [0.10625, 0.03938, 0.01556, 0.01109, 0.00391, 0.00250]; % Loss-PU
N =       [4,       8,       12,      16,      32,      64];

diamond = [0.17438, 0.04641, 0.02072, 0.013125, 0.00984, 0.00611]; % Loss-PU
reck =    [0.10625, 0.03938, 0.01556, 0.007500, 0.00391, 0.00250]; % Loss-PU
N =       [4,       8,       12,      16,      32,      64];

diamond = [0.17438, 0.04641, 0.02072, 0.019844, 0.00984, 0.00611]; % Loss-PU
reck =    [0.10625, 0.03938, 0.01556, 0.011094, 0.00391, 0.00250]; % Loss-PU
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
legend(lgd, 'fontsize', fontsz*0.8, 'interpreter','latex');

a = get(gca,'YTickLabel');
set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz*0.7)
h = gca;
set(h, 'YTickLabelMode','auto')
set(h, 'XTickLabelMode','auto')
xticks(N)
xlabel('Structure Size ($N$)', 'fontsize', fontsz*0.8, 'interpreter','latex')
ylabel('FoM $(\mathrm{Rad} \cdot \mathrm{dB})$', 'fontsize', fontsz*0.8, 'interpreter','latex')
% axis square
grid('on')
pMe('/storage/Research/02.2020-NewPaper/FoM.pdf')
%%
figure('Renderer', 'painters', 'Position', [400 400 1800 1300])

% plot(N, diamond, 'linewidth', 3)
% hold on
% plot(N, reck, 'linewidth', 3)


plot(N, reck, 'r-v', 'markersize', 20, 'MarkerFaceColor', '#c3c3c3','linewidth', 3)
hold on
plot(N, diamond, 'b-d', 'markersize', 20, 'MarkerFaceColor', 'k', 'linewidth', 3)
set(gca, 'YScale', 'log')
fontsz = 64;

% title('Figure of Merit', 'fontsize', fontsz, 'interpreter','latex');
lgd = {'Reck', 'Diamond'};
legend(lgd, 'fontsize', fontsz*0.8, 'interpreter','latex');

a = get(gca,'YTickLabel');
set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz*0.7)
h = gca;
set(h, 'YTickLabelMode','auto')
set(h, 'XTickLabelMode','auto')
xticks(N)
xlabel('Structure Size ($N$)', 'fontsize', fontsz*0.8, 'interpreter','latex')
ylabel('log(FoM) $(\mathrm{Rad} \cdot \mathrm{dB})$', 'fontsize', fontsz*0.8, 'interpreter','latex')

% axis square
grid('on')
pMe('/storage/Research/02.2020-NewPaper/FoM-log.pdf')

%%
if 0
figure('Renderer', 'painters', 'Position', [400 400 1800 1300])

diamond_n = diamond./diamond;
reck_n = reck./diamond;
bar(N, diamond_n, 'linewidth', 3)
hold on
bar(N+1, reck_n, 'linewidth', 3)

title('Figure of Merit')
lgd = {'Diamond','Reck'};
legend(lgd, 'fontsize', fontsz, 'interpreter','latex');

a = get(gca,'YTickLabel');
set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz*0.7)
h = gca;
set(h, 'YTickLabelMode','auto')
set(h, 'XTickLabelMode','auto')
xticks([4,8,16,32])
xlabel('$N$', 'fontsize', fontsz, 'interpreter','latex')
ylabel('Figure of Merit', 'fontsize', fontsz, 'interpreter','latex')
end