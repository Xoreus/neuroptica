% Function to take in data from a Neuroptica simulation (created in python
% with either ONN_Topologies_Analysis_Retrained.py or
% plotAcc_singleModel_AllLoss.py
% Plots the accuracy for all models and a single loss with varying phase
% uncertainties
%
% Author: Simon Geoffroy-Gagnon
% Edit: 2020.03.17


function plotAcc_singleModel_Loss_lineplot(FOLDER, sim, topo, printMe)
fontsz = 64;
figure('Renderer', 'painters', 'Position', [400 400 1900 1400])
hold on
lgd = {};

for t = 1:length(topo)
    simulation = sim.(topo{t});
    accuracy = sim.(topo{t}).accuracy_LPU;
    lgd{end+1} = simulation.topology;
    same_phaseUncert = accuracy(1, :);
    plot(simulation.loss_dB, same_phaseUncert, 'linewidth', 3)
end
% lgd{end-1} = 'Reck + Inv. Reck';

a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',fontsz)
a = get(gca,'YTickLabel');
set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz)
h = gca;
set(h, 'YTickLabelMode','auto')
set(h, 'XTickLabelMode','auto')
axis square
xlabel('Loss/MZI (dB)', 'fontsize', fontsz, 'interpreter','latex')
ylabel('Accuracy (\%)', 'fontsize', fontsz, 'interpreter','latex')

h = gca;
ylim([0 100])
set(h, 'YTickLabelMode', 'auto')
set(h, 'XTickLabelMode','auto')
legend(lgd, 'fontsize', fontsz*.8, 'interpreter','latex', 'location', 'southwest');
% axis tight
% axis square
ylim([0, 100])
box on

if printMe
    fname = [FOLDER, '/LPM_lineplot.pdf'];
    set(gcf,'Units','inches');
    savefig([FOLDER, sprintf('/Matlab_Figs/LPM_lineplot.fig')])
    saveas(gcf, [FOLDER, sprintf('/LPM_lineplot.png')])
    
    screenposition = [4 4 18 17];
    % screenposition = get(gcf, 'position');
    set(gcf, 'PaperPosition',[0 0 screenposition(3:4)], 'PaperSize',[screenposition(3:4)]);
    print('-dpdf','-painters', fname)
end

end