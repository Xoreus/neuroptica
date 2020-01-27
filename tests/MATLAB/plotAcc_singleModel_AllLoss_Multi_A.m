% Function to take in data from a Neuroptica simulation (created in python
% with either ONN_Topologies_Analysis_Retrained.py or
% ONN_Topologies_Analysis.py
% Plots the accuracy for a single model trained at a specific loss/phase
% uncert, at all losses with varying phase uncert
%
% This will create a ridiculous amnt of data, so we'll save the fig in a
% separate dir than the pngs.
%
% Author: Simon Geoffroy-Gagnon
% Edit: 25.01.2020


function plotAcc_singleModel_AllLoss_Multi_A(FOLDER, SimulationSettings)
fontsz = 28;

SimulationSettings.losses_dB_train = SimulationSettings.losses_dB_train([1]);

for l_idx = 1:length(SimulationSettings.losses_dB_train)
    figure('Renderer', 'painters', 'Position', [400 400 1900 1400])
    for model_idx = 1:length(SimulationSettings.ONN_Setups)
        legend_ = create_legend_single_model(SimulationSettings.losses_dB_test);
        for p_idx = 1:length(SimulationSettings.phase_uncerts_train)
            
            Model_acc = load([FOLDER, sprintf('acc_%s_loss=%.3f_uncert=%.3f_%sFeat.txt', ...
                SimulationSettings.ONN_Setups{model_idx}, SimulationSettings.losses_dB_train(l_idx), ...
                SimulationSettings.phase_uncerts_train(p_idx), SimulationSettings.N)]);
            
            plot(SimulationSettings.phase_uncerts_test, Model_acc(:,:), 'linewidth', 3)
            
            legend(legend_, 'fontsize', fontsz, 'interpreter','latex');
            ylim([0, 100])
            
            a = get(gca,'XTickLabel');
            set(gca,'XTickLabel',a,'FontName','Times','fontsize',fontsz/1.2)
            
            a = get(gca,'YTickLabel');
            set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz/1.2)
            
            ylim([0, 100])
            
            xlabel('Phase Uncertainty $(Rad, \sigma)$', 'fontsize', fontsz, 'interpreter','latex')
            ylabel('Accuracy (\%)', 'fontsize', fontsz, 'interpreter','latex')
            
            title(sprintf('Accuracy of Model with %s Topology \nTrained at Loss = %.3f$\\pm$%s dB/MZI, Phase Uncert = %.3f Rad', ...
                SimulationSettings.models{model_idx}, SimulationSettings.losses_dB_test(l_idx), SimulationSettings.loss_diff,...
                SimulationSettings.phase_uncerts_train(p_idx)),'fontsize', 1.5*fontsz, 'interpreter','latex')
            
            savefig([FOLDER, sprintf('Matlab_Figs/%s_trainedAtLoss=%.3f_phaseUncert=%.3f.fig',...
                SimulationSettings.ONN_Setups{model_idx}, SimulationSettings.losses_dB_train(l_idx), SimulationSettings.phase_uncerts_train(p_idx))])
            saveas(gcf, [FOLDER, sprintf('Matlab_Pngs/%s_trainedAtLoss=%.3f_phaseUncert=%.3f.png',...
                SimulationSettings.ONN_Setups{model_idx}, SimulationSettings.losses_dB_train(l_idx), SimulationSettings.phase_uncerts_train(p_idx))])
        end
    end
end
end