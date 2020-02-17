% Script to plot the Loss/Training Accuracuy/Validation Accurracy or an ONN
% model
%
% Author: Simon Geoffroy-Gagnon
% Edit: 15.02.2020


function ONN_Accuracy_Plot(FOLDER, SimulationSettings)
fontsz = 44;
epochs = 1000;

for model_idx = 1:size(SimulationSettings.ONN_setup, 1)
    figure('Renderer', 'painters', 'Position', [400 400 1900 1400])
    modelTopo = sprintf('%s',strrep(SimulationSettings.ONN_setup(model_idx, :), '_', '\_'));
    Model_acc = load([FOLDER, modelTopo, '.mat']);
    model = Model_acc.(modelTopo);
    
    yyaxis left
    plot(model.losses(:, 1:epochs), 'linewidth', 3)
    ylabel('Loss Function (MSE)', 'fontsize', fontsz, 'interpreter','latex')
    
    yyaxis right
    plot(model.trn_accuracy(:, 1:epochs), '--', 'linewidth', 3)
    hold on
    plot(model.val_accuracy(:, 1:epochs), '-', 'linewidth', 3)
    ylabel('Accuracy (\%)', 'fontsize', fontsz, 'interpreter','latex')
    xlabel('Epoch', 'fontsize', fontsz, 'interpreter','latex')
    ylim([0 100])
    a = get(gca,'YTickLabel');
    set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz*0.9)
    
    legend({'Loss Function','Training Accuracy','Validation Accuracy'}, 'fontsize', fontsz, 'interpreter','latex', 'location', 'east');
    
    title(sprintf('Accuracy of %s Topology', model.topology), 'fontsize', fontsz, 'interpreter','latex')
    drawnow;
    savefig([FOLDER, sprintf('Matlab_Figs/%s_loss&acc-plot.fig', model.topology)])
    saveas(gcf, [FOLDER, sprintf('Matlab_Pngs/%s_loss&acc-plot.png',  model.topology)])
end