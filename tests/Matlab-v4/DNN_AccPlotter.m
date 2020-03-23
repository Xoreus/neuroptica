% Plot DNN loss/val acc/trn acc
close all
hidden = 2;
F = sprintf('/home/simon/Documents/neuroptica/tests/Analysis/DNN/iris-%dh-og', hidden);
l = load([F '/losses.txt']);
trnacc = load([F '/trn_acc.txt']);
valacc = load([F '/val_acc.txt']);
fontsz = 64;

figure('Renderer', 'painters', 'Position', [400 400 1900 1400])
model = 'Digital Neural Network';

yyaxis left
plot(l(:, 1:end), 'linewidth', 3)
ylabel('Loss Function (MSE)', 'fontsize', fontsz, 'interpreter','latex')
h = gca;
set(h, 'YTickLabelMode','auto')
set(h, 'XTickLabelMode','auto')

yyaxis right
plot(trnacc(:, 1:end), '--', 'linewidth', 3)
hold on
plot(valacc(:, 1:end), '-', 'linewidth', 3)
ylabel('Accuracy (\%)', 'fontsize', fontsz, 'interpreter','latex')
xlabel('Epoch', 'fontsize', fontsz, 'interpreter','latex')
ylim([0 100])


a = get(gca,'YTickLabel');
set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz*0.7)
h = gca;
set(h, 'YTickLabelMode','auto')
set(h, 'XTickLabelMode','auto')

l = legend({'Loss Function','Training Accuracy','Validation Accuracy'}, 'fontsize', fontsz, 'interpreter','latex', 'location', 'east');
set(l, 'position', get(l, 'position') + [-.015 -.15 0 0])
yyaxis left
yyaxis right
ylim([0 100])

title(sprintf('Accuracy of %s', model), 'fontsize', fontsz, 'interpreter','latex')
drawnow;
% savefig(sprintf('DNN_loss+acc-plot.fig'))
% saveas(gcf, sprintf('DNN_loss+acc-plot.png'))

set(gcf,'Units','inches');
screenposition = [12.4271  4.7917   18.8542   12.9583];
set(gcf, 'PaperPosition',[0 0 screenposition(3:4)], 'PaperSize', [screenposition(3:4)]);
fname = [F sprintf('/../DNN_loss+acc_h=%d.pdf', hidden)];
print('-dpdf','-painters', fname)