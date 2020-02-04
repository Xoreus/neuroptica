% Function to take in data from a Neuroptica simulation (created in python
% with either ONN_Topologies_Analysis_Retrained.py or
% plotAcc_singleModel_AllLoss.py
% Plots the accuracy for all models and a single loss with varying phase
% uncertainties
%
% Author: Simon Geoffroy-Gagnon
% Edit: 25.01.2020


function plotAcc_singleModel_AllLoss(FOLDER, SimulationSettings)
fontsz = 28;
figure('Renderer', 'painters', 'Position', [400 400 1900 1400])
step_sz = 3;
SimulationSettings.loss_dB = SimulationSettings.loss_dB(1:step_sz:end);

for model_idx = 1:length(SimulationSettings.ONN_Setups)
    
    Model_acc = load([FOLDER, sprintf('acc_%s_loss=%.3f_uncert=%.3f_%sFeat.txt', ...
        SimulationSettings.ONN_Setups{model_idx}, SimulationSettings.loss_dB(1), SimulationSettings.phase_uncerts(1), ...
        SimulationSettings.N)]);
    
    plot(SimulationSettings.phase_uncerts, Model_acc(:, 1:step_sz:end), 'linewidth', 3)
    
    legend_ = create_legend_single_model(SimulationSettings.loss_dB);
    legend(legend_, 'fontsize', fontsz, 'interpreter','latex', 'location', 'best');
    
    ylim([0, 100])
    
    a = get(gca,'XTickLabel');
    set(gca,'XTickLabel',a,'FontName','Times','fontsize',fontsz/1.2)
    a = get(gca,'YTickLabel');
    set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz/1.2)
    
    xlabel('Phase Uncertainty $(\sigma)$', 'fontsize', fontsz, 'interpreter','latex')
    ylabel('Accuracy (\%)', 'fontsize', fontsz, 'interpreter','latex')
    
    title(sprintf('Accuracy of Model with %s Topology\n Loss Standard Deviation $\\sigma_{Loss} = $ %s dB/MZI',SimulationSettings.models{model_idx},...
        SimulationSettings.loss_diff), 'fontsize', 1.5*fontsz, 'interpreter','latex')
    
    
    savefig([FOLDER, sprintf('Matlab_Figs/Model=%s_Loss=[%.3f-%.2f].fig', SimulationSettings.ONN_Setups{model_idx}, ...
        min(SimulationSettings.loss_dB), max(SimulationSettings.loss_dB))])
    saveas(gcf, [FOLDER, sprintf('Matlab_Pngs/Model=%s_Loss=[%.3f-%.2f].png', SimulationSettings.ONN_Setups{model_idx}, ...
        min(SimulationSettings.loss_dB), max(SimulationSettings.loss_dB))])
end
