% Function to take in data from a Neuroptica simulation (created in python
% with either ONN_Topologies_Analysis_Retrained.py or
% plotAcc_singleModel_AllLoss.py
% Plots the accuracy for all models and a single loss with varying phase
% uncertainties
%
% Author: Simon Geoffroy-Gagnon
% Edit: 25.01.2020


function plotAcc_singleModel_AllLoss_Multi(FOLDER, SimulationSettings)
fontsz = 28;
figure('Renderer', 'painters', 'Position', [400 400 1900 1400])

for model_idx = 1:length(SimulationSettings.ONN_Setups)
    for ii = 1:length(SimulationSettings.losses_dB_train([1,end]))
        for jj = 1:length(SimulationSettings.phase_uncerts_train([1,end]))
            
            
            Model_acc = load([FOLDER, sprintf('acc_%s_loss=%.3f_uncert=%.3f_%sFeat.txt', ...
                SimulationSettings.ONN_Setups{model_idx}, SimulationSettings.losses_dB_train(ii), ...
                SimulationSettings.phase_uncerts_train(jj), SimulationSettings.N)]);
            
            plot(SimulationSettings.phase_uncerts_test, Model_acc, 'linewidth', 3)
            
            legend_ = create_legend_single_model(SimulationSettings.losses_dB_test);
            legend(legend_, 'fontsize', fontsz, 'interpreter','latex');
            
            ylim([0, 100])
            
            a = get(gca,'XTickLabel');
            set(gca,'XTickLabel',a,'FontName','Times','fontsize',fontsz/1.2)
            
            a = get(gca,'YTickLabel');
            set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz/1.2)
            
            xlabel('Phase Uncertainty $(\sigma)$', 'fontsize', fontsz, 'interpreter','latex')
            ylabel('Accuracy (\%)', 'fontsize', fontsz, 'interpreter','latex')
            
            title(sprintf('Accuracy of Model with %s topology\n Loss Difference of $\\pm$ %s dB',strrep(SimulationSettings.models{model_idx}, '_','\_')...
                ,SimulationSettings.loss_diff), 'fontsize', 1.5*fontsz, 'interpreter','latex')
            
            
            savefig([FOLDER, sprintf('Matlab_Figs/Model=%s_Loss=[%.3f-%.2f].fig', SimulationSettings.ONN_Setups{model_idx}, ...
                min(SimulationSettings.losses_dB_train), max(SimulationSettings.losses_dB_train))])
            saveas(gcf, [FOLDER, sprintf('Matlab_Pngs/Model=%s_Loss=[%.3f-%.2f].png', SimulationSettings.ONN_Setups{model_idx}, ...
                min(SimulationSettings.losses_dB_train), max(SimulationSettings.losses_dB_train))])
        end
    end
end
