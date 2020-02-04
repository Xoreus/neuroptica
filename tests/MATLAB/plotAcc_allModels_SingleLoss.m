% Function to take in data from a Neuroptica simulation (created in python
% with either ONN_Topologies_Analysis_Retrained.py or
% plotAcc_allModels_SingleLoss.py
% Plots the accuracy for all models and a single loss with varying phase
% uncertainties
%
% Author: Simon Geoffroy-Gagnon
% Edit: 25.01.2020


function plotAcc_allModels_SingleLoss(FOLDER, SimulationSettings)
fontsz = 28;
figure('Renderer', 'painters', 'Position', [400 400 1900 1400])
for l_idx = 1:length(SimulationSettings.loss_dB)
    
    for model_idx = 1:length(SimulationSettings.ONN_Setups)
        
        Model_acc = load([FOLDER, sprintf('acc_%s_loss=%.3f_uncert=%.3f_%sFeat.txt', ...
            SimulationSettings.ONN_Setups{model_idx}, SimulationSettings.loss_dB(1), SimulationSettings.phase_uncerts(1), SimulationSettings.N)]);
        
        plot(SimulationSettings.phase_uncerts, Model_acc(:, l_idx), 'linewidth', 3)
        hold on
        
    end
    hold off
    
    legend_ = create_legend_single_loss(SimulationSettings.models);
    
    legend(legend_, 'fontsize', fontsz, 'interpreter','latex', 'location', 'best');
    ylim([0, 100])
    
    a = get(gca,'XTickLabel');
    set(gca,'XTickLabel',a,'FontName','Times','fontsize',fontsz/1.2, 'interpreter','latex')
    
    a = get(gca,'YTickLabel');
    set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz/1.2, 'interpreter','latex')
    
    xlabel('Phase Uncertainty $(\sigma)$', 'fontsize', fontsz, 'interpreter','latex')
    ylabel('Accuracy (\%)', 'fontsize', fontsz, 'interpreter','latex')
    
    title(sprintf('Accuracy of Models with Loss = %.2f dB/MZI, $\\sigma_{Loss} =$ %s dB/MZI', SimulationSettings.loss_dB(l_idx), SimulationSettings.loss_diff),...
        'fontsize', 1.5*fontsz, 'interpreter','latex')
    
    
    savefig([FOLDER, sprintf('Matlab_Figs/AllModels_Loss=%.3f.fig', SimulationSettings.loss_dB(l_idx))])
    saveas(gcf, [FOLDER, sprintf('Matlab_Pngs/AllModels_Loss=%.3f.png', SimulationSettings.loss_dB(l_idx))])
end

end