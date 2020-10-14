% Function to take in data from a Neuroptica simulation (created in python
% with either ONN_Topologies_Analysis_Retrained.py or
% plotAcc_singleModel_AllLoss.py
% Plots the accuracy for all models and a single loss with varying phase
% uncertainties
%
% Author: Simon Geoffroy-Gagnon
% Edit: 2020.05.07

function plotAcc_Reviewer1_LinePlot_Fig(FOLDER, sim, topo, loss_idx, printMe)
fontsz = 64;
figure('Renderer', 'painters', 'Position', [400 400 1900 1400])

for t = 1:length(topo)
    simulation = sim.(topo{t});
    accuracy = simulation.accuracy_LPU;
    
    plot(simulation.phase_uncert_theta, accuracy(:, loss_idx), 'linewidth', 3)
    axis square
    hold on
end
%     legend_ = create_legend_single_model(simulation.loss_dB([1, 6, 11, 16, 21]));
lgd = {'Diamond','Reck'};
legend(lgd, 'fontsize', fontsz*.8, 'interpreter','latex', 'location', 'northeast');

ylim([0, 100])

h = gca;
set(h, 'YTickLabelMode', 'auto')
set(h, 'XTickLabelMode','auto')

fprintf('%s acc = %.3f\n', simulation.topo, max(max(max(accuracy))))

xlabel('$\sigma_\phi,\;\sigma_\theta$ (Rad)', 'fontsize', fontsz, 'interpreter','latex')
ylabel('Accuracy (\%)', 'fontsize', fontsz, 'interpreter','latex')

title(sprintf('%.2f dB Loss/MZI',sim.C_Q_P.loss_dB(loss_idx)), 'fontsize', fontsz, 'interpreter','latex')
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',fontsz*0.75)
a = get(gca,'YTickLabel');
set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz*0.75)
h = gca;

axis square
%     set(h, 'YTickLabelMode','auto')
set(h, 'XTickLabelMode','auto')
%     xlim([0 0.5])
saveas(gcf, [FOLDER, sprintf('/Model=%s_lineplot.png', simulation.topology)])
if printMe
    pMe(['../Crop_Me' '/' sprintf('%.3f_dB_il-all_meshes_lineplot.pdf', simulation.loss_dB(loss_idx))])
end
end
