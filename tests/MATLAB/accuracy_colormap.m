% Function to take inmax data from a Neuroptica simulation (created in python
% with either ONN_Topologies_Analysis_Retrained.py or
% plotAcc_singleModel_AllLoss.py
% Plots the accuracy for all models and a single loss with varying phase
% uncertainties
%
% Author: Simon Geoffroy-Gagnon
% Edit: 25.01.2020


function accuracy_colormap(FOLDER, SimulationSettings)
fontsz = 28;
figure('Renderer', 'painters', 'Position', [400 400 1900 1400])
for model_idx = 1:length(SimulationSettings.ONN_Setups)
    
    Model_acc = load([FOLDER, sprintf('acc_%s_loss=%.3f_uncert=%.3f_%sFeat.txt', ...
        SimulationSettings.ONN_Setups{model_idx}, SimulationSettings.loss_dB(1), SimulationSettings.phase_uncerts(1), SimulationSettings.N)]);
    
    imagesc(SimulationSettings.loss_dB, SimulationSettings.phase_uncerts, Model_acc)
    set(gca,'YDir','normal')
    c = colorbar;
    c.Label.String = 'Accuracy (%)';
    caxis([20 80]) 
    
    a = get(gca,'XTickLabel');
    set(gca,'XTickLabel',a,'FontName','Times','fontsize',fontsz/1.2)
    
    a = get(gca,'YTickLabel');
    set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz/1.2)
    
    ylabel('Phase Uncertainty $(\sigma)$', 'fontsize', fontsz, 'interpreter','latex')
    xlabel('Loss (dB/MZI)', 'fontsize', fontsz, 'interpreter','latex')
    
    title(sprintf('Accuracy of Model with %s Topology\n Loss Standard Deviation $\\sigma_{Loss} = $ %s dB/MZI',SimulationSettings.models{model_idx},...
        SimulationSettings.loss_diff), 'fontsize', 1.5*fontsz, 'interpreter','latex')
    
    savefig([FOLDER, sprintf('Matlab_Figs/ColorMap-Model=%s_Loss=[%.3f-%.2f].fig', SimulationSettings.ONN_Setups{model_idx}, ...
        min(SimulationSettings.loss_dB), max(SimulationSettings.loss_dB))])
    saveas(gcf, [FOLDER, sprintf('Matlab_Pngs/ColorMap-Model=%s_Loss=[%.3f-%.2f].png', SimulationSettings.ONN_Setups{model_idx}, ...
        min(SimulationSettings.loss_dB), max(SimulationSettings.loss_dB))])
end
