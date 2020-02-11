% Script to plot the Loss/Training Accuracuy/Validation Accurracy or an ONN
% model
%
% Author: Simon Geoffroy-Gagnon
% Edit: 03.02.2020


function ONN_Accuracy_Plot(FOLDER, SimulationSettings)
Data_Fitting = 'Data_Fitting/';
fontsz = 44;
figure('Renderer', 'painters', 'Position', [400 400 1900 1400])
for model_idx = 1:size(SimulationSettings.ONN_setup, 1)
    modelTopo = sprintf('%s',strrep(SimulationSettings.ONN_setup(model_idx, :), ' ', ''));
    Model_acc = load([FOLDER, modelTopo, '.mat']);
    model = Model_acc.(modelTopo);
    
    yyaxis left
    plot(model.losses, 'linewidth', 3)
    ylabel('Loss Function (MSE)', 'fontsize', fontsz, 'interpreter','latex')
    
    a = get(gca,'XTickLabel');
    set(gca,'XTickLabel',a,'FontName','Times','fontsize',fontsz*0.9)
    a = get(gca,'YTickLabel');
    set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz*0.9)
    ylim([.15 .5])
    yyaxis right
    plot(model.trn_accuracy, '--', 'linewidth', 3)
    hold on
    plot(model.val_accuracy, '-', 'linewidth', 3)
    ylabel('Accuracy (\%)', 'fontsize', fontsz, 'interpreter','latex')
    xlabel('Epoch', 'fontsize', fontsz, 'interpreter','latex')
    ylim([0 100])
    a = get(gca,'XTickLabel');
    set(gca,'XTickLabel',a,'FontName','Times','fontsize',fontsz*0.9)
    a = get(gca,'YTickLabel');
    set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz*0.9)
    
    legend({'Loss Function','Training Accuracy','Validation Accuracy'}, 'fontsize', fontsz, 'interpreter','latex', 'location', 'best');
    
    %     title(sprintf('Loss and Training/Validation Accuracy\n ONN model with %s topology\nDataset: %s\n Loss/MZI = %.3f dB with a standard deviation of %s dB\n Phase Uncertainty = %.3f Radians',...
    %         SimulationSettings.models{ii}, SimulationSettings.dataset_name, SimulationSettings.loss_dB(1),...
    %         SimulationSettings.loss_diff, SimulationSettings.phase_uncert_theta(1)), 'fontsize', fontsz, 'interpreter','latex')
    title(sprintf('Accuracy of Model with %s Topology', model.topology), 'fontsize', fontsz, 'interpreter','latex')
    
    savefig([FOLDER, sprintf('Matlab_Figs/AccuracyPlot-Model=%s_Loss=%.3f.fig',  model.topology, ...
        min(SimulationSettings.loss_dB))])
    saveas(gcf, [FOLDER, sprintf('Matlab_Pngs/AccuracyPlot-Model=%s_Loss=%.3f.png',  model.topology, ...
        min(SimulationSettings.loss_dB))])
    hold off
end