% Function to take in data from a Neuroptica simulation (created in python
% with either ONN_Topologies_Analysis_Retrained.py or
% plotAcc_allModels_SinglePhaseUncert.py
% Plots the accuracy for all models and a single phase uncert with loss
%
% Author: Simon Geoffroy-Gagnon
% Edit: 25.01.2020


function plotAcc_allModels_SinglePhaseUncert(FOLDER, SimulationSettings)
fontsz = 28;
figure('Renderer', 'painters', 'Position', [400 400 1900 1400])
for p_idx = 1:length(SimulationSettings.phase_uncerts)
        

    for model_idx = 1:length(SimulationSettings.ONN_Setups)
        legend_ = create_legend_single_loss(SimulationSettings.models);
        if ~contains(SimulationSettings.ONN_Setups{model_idx}, 'N')
        Model_acc = load([FOLDER, sprintf('acc_%s_loss=%.3f_uncert=%.3f_%sFeat.txt', ...
            SimulationSettings.ONN_Setups{model_idx}, SimulationSettings.loss_dB(1), ...
            SimulationSettings.phase_uncerts(1), SimulationSettings.N)]);
        
            plot(SimulationSettings.loss_dB, Model_acc(p_idx, :), 'linewidth', 3)
            
            hold on
        end
    end
    hold off
    legend(legend_, 'fontsize', fontsz,  'interpreter','latex');
    ylim([0, 100])
    
    a = get(gca,'XTickLabel');
    set(gca,'XTickLabel',a,'FontName','Times','fontsize',fontsz/1.2)

    a = get(gca,'YTickLabel');
    set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz/1.2)
    
    xlabel(sprintf('Loss (dB/MZI, $\\sigma_{Loss} = $ %s dB/MZI)', SimulationSettings.loss_diff), 'fontsize', fontsz, 'interpreter','latex')
    ylabel('Accuracy (\%)', 'fontsize', fontsz, 'interpreter','latex')

    title(sprintf('Accuracy of Models with Phase Uncertainty $\\sigma_{Phase\\; Uncert}$ = %.2f Rad', SimulationSettings.phase_uncerts(p_idx)),...
        'fontsize', 1.5*fontsz, 'interpreter','latex')
    
    
    savefig([FOLDER, sprintf('Matlab_Figs/AllModels_PhaseUncert=%.3f.fig', SimulationSettings.phase_uncerts(p_idx))])
    saveas(gcf, [FOLDER, sprintf('Matlab_Pngs/AllModels_PhaseUncert=%.3f.png', SimulationSettings.phase_uncerts(p_idx))])
end

end