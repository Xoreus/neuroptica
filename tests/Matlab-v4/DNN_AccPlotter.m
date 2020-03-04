% Plot DNN loss/val acc/trn acc
close all

l = load('losses.txt');
trnacc = load('trn_acc.txt');
valacc = load('val_acc.txt');
fontsz = 44;

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
set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz*0.7)


legend({'Loss Function','Training Accuracy','Validation Accuracy'}, 'fontsize', fontsz, 'interpreter','latex', 'location', 'east');
yyaxis left
yyaxis right
ylim([0 100])

title(sprintf('Accuracy of %s', model), 'fontsize', fontsz, 'interpreter','latex')
drawnow;
savefig([sprintf('DNN_loss+acc-plot.fig')])
saveas(gcf, [sprintf('DNN_loss+acc-plot.png')])

set(gcf,'Units','inches');
screenposition = [12.4271  4.7917   18.8542   12.9583];
set(gcf, 'PaperPosition',[0 0 screenposition(3:4)], 'PaperSize',[screenposition(3:4)]);

print -dpdf -painters DNN_loss+acc.pdf