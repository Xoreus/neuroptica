% Script to plot the Loss/Training Accuracuy/Validation Accurracy or an ONN
% model
%
% Author: Simon Geoffroy-Gagnon
% Edit: 03.02.2020


function ONN_Accuracy_Plot(FOLDER, SimulationSettings)
Data_Fitting = 'Data_Fitting/';
fontsz = 28;
figure('Renderer', 'painters', 'Position', [400 400 1900 1400])
for ii = 1:length(SimulationSettings.ONN_Setups)
    Acc_Data = readtable([FOLDER, Data_Fitting, sprintf('%s_loss=0.000dB_uncert=0.000Rad_%sFeatures.txt', SimulationSettings.ONN_Setups{ii}, SimulationSettings.N)]);
%     Acc_Data = readtable([FOLDER, Data_Fitting, sprintf('R_D_P_loss=0.000dB_uncert=0.000Rad_10Features.txt')])
    
    yyaxis left
    plot(Acc_Data.Losses, 'linewidth', 2)
    ylabel('Losses (MSE)')
    a = get(gca,'XTickLabel');
    set(gca,'XTickLabel',a,'FontName','Times','fontsize',fontsz/1.2)
    
    a = get(gca,'YTickLabel');
    set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz/1.2)
    yyaxis right
    plot(Acc_Data.TrainingAccuracy, 'b--', 'linewidth', 2)
    hold on
    plot(Acc_Data.ValidationAccuracy, '-', 'linewidth', 2)
    ylabel('Accuracy')
    ylim([0 100])
    a = get(gca,'XTickLabel');
    set(gca,'XTickLabel',a,'FontName','Times','fontsize',fontsz/1.2)
    
    a = get(gca,'YTickLabel');
    set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz/1.2)
    legend({'Losses','Training Accuracy', 'Validation Accuracy'}, 'fontsize', fontsz, 'interpreter','latex', 'location', 'east');
    
    title(sprintf('Loss and Training/Validation Accuracy\n ONN model with %s topology\nDataset: %s\n Loss/MZI = %.3f dB with a standard deviation of %s dB\n Phase Uncertainty = %.3f Radians',...
        SimulationSettings.models{ii}, SimulationSettings.dataset_name, SimulationSettings.loss_dB(1),...
        SimulationSettings.loss_diff, SimulationSettings.phase_uncerts(1)), 'fontsize', 1.5*fontsz, 'interpreter','latex')
    
    savefig([FOLDER, sprintf('Matlab_Figs/AccuracyPlot-Model=%s_Loss=%.3f.fig', SimulationSettings.ONN_Setups{ii}, ...
        min(SimulationSettings.loss_dB))])
    saveas(gcf, [FOLDER, sprintf('Matlab_Pngs/AccuracyPlot-Model=%s_Loss=%.3f.png', SimulationSettings.ONN_Setups{ii}, ...
        min(SimulationSettings.loss_dB))])
    hold off
end