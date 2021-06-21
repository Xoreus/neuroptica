% Function to plot DNN accuracy and loss in the same way as ONN plotter

function plot_dnn(FOLDER)
close all

l = load([FOLDER '/' 'losses.txt']);
trnacc = load([FOLDER '/' 'trn_acc.txt']);
valacc = load([FOLDER '/' 'val_acc.txt']);
fontsz = 50;

figure('Renderer', 'painters', 'Position', [400 400 1900 1400])
model = 'Digital Neural Network';

yyaxis left
plot(l(:, 1:end), 'linewidth', 3)
ylabel('Loss Function (MSE)', 'fontsize', fontsz, 'interpreter','latex')
yyaxis right
plot(trnacc(:, 1:end), '--', 'linewidth', 3)
hold on
plot(valacc(:, 1:end), '-', 'linewidth', 3)
ylabel('Accuracy (\%)', 'fontsize', fontsz, 'interpreter','latex')
xlabel('Epoch', 'fontsize', fontsz, 'interpreter','latex')
ylim([0 100])
a = get(gca,'YTickLabel');
set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz)

legend({'Loss Function','Training Accuracy','Validation Accuracy'}, 'fontsize', fontsz, 'interpreter', ...
    'latex', 'location', 'northeast');
yyaxis left
yyaxis right
ylim([0 100])

title(sprintf('Accuracy of %s', model), 'fontsize', fontsz, 'interpreter','latex')
drawnow;
% savefig([sprintf('DNN_loss+acc-plot.fig')])
% saveas(gcf, [sprintf('DNN_loss+acc-plot.png')])
pMe(sprintf('/home/edwar/Documents/Github_Projects/Thesis/Figures/DNN_Iris_0h_maxAcc%d.pdf', max(valacc)))