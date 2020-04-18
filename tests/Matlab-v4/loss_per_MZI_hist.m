% Code do display histogram of different possible MZI IL
%
% Author: Simon Geoffroy-Gagnon
% Edit: 2020.04.10
clear; close all; clc
%% Initial Values
s = 100000;
loss_dB = 0.5;
loss_diff = 0.5;
loss_min = 0.5;
fontsz = 64;
%% 1). Max(0.5, N(0.5, 0.5))

l = normrnd(loss_dB, loss_diff, s, 1);
l_min = loss_min*ones(s, 1);

loss = max([l_min, l]'); %#ok<*UDIM>

figure('Renderer', 'painters', 'Position', [400 400 1900 1400])
histogram(loss)
title('Max(0.5, N(0.5, 0.5))','FontName','Times','fontsize',fontsz, 'interpreter','latex')
xlabel('Loss/MZI (dB)', 'fontsize', fontsz, 'interpreter','latex')
ylabel('Occurences (\#)', 'fontsize', fontsz, 'interpreter','latex')
a = get(gca,'YTickLabel');
set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz)
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',fontsz)
h = gca;
set(h, 'YTickLabelMode','auto')
set(h, 'XTickLabelMode','auto')
%% 2). 0.5 + abs(N(0, 0.5))


l = abs(normrnd(0, loss_diff, s, 1));
l_min = loss_min*ones(s, 1);

loss = loss_min + l;

figure('Renderer', 'painters', 'Position', [400 400 1900 1400])
histogram(loss)
title('0.5 + abs(N(0, 0.5))','FontName','Times','fontsize',fontsz, 'interpreter','latex')
xlabel('Loss/MZI (dB)', 'fontsize', fontsz, 'interpreter','latex')
ylabel('Occurences (\#)', 'fontsize', fontsz, 'interpreter','latex')
a = get(gca,'YTickLabel');
set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz)
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',fontsz)
h = gca;
set(h, 'YTickLabelMode','auto')
set(h, 'XTickLabelMode','auto')