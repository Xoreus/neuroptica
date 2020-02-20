% Function to take in data from a Neuroptica simulation (created in python
% with either ONN_Topologies_Analysis_Retrained.py or
% ONN_Topologies_Analysis.py
% Plots the accuracy for all models and a single loss with varying phase
% uncertainties
%
% Author: Simon Geoffroy-Gagnon
% Edit: 25.01.2020


function plotAcc_allModels_SingleLoss_Multi(FOLDER, SimulationSettings)
fontsz = 28;
            
figure('Renderer', 'painters', 'Position', [400 400 1900 1400])
for ii = 1:length(SimulationSettings.losses_dB_train([1,end]))
    for jj = 1:length(SimulationSettings.phase_uncerts_train([1,end]))
        for l_idx = 1:length(SimulationSettings.losses_dB_test)            
            for model_idx = 1:length(SimulationSettings.ONN_Setups)
                legend_ = create_legend_single_loss(SimulationSettings.ONN_Setups);
                if ~contains(SimulationSettings.ONN_Setups{model_idx}, 'N')
                    Model_acc = load([FOLDER, sprintf('acc_%s_loss=%.3f_uncert=%.3f_%sFeat.txt', ...
                        SimulationSettings.ONN_Setups{model_idx}, SimulationSettings.losses_dB_train(ii), ...
                        SimulationSettings.phase_uncerts_train(jj), SimulationSettings.N)]);
                    
                    plot(SimulationSettings.phase_uncerts_test, Model_acc(:, l_idx), 'linewidth', 3)
                    hold on
                end
            end
            
            legend(legend_, 'fontsize', fontsz, 'interpreter','latex');
            ylim([0, 100])
            
            a = get(gca,'XTickLabel');
            set(gca,'XTickLabel',a,'FontName','Times','fontsize',fontsz/1.2)
            
            a = get(gca,'YTickLabel');
            set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz/1.2)
            
            ylim([0, 100])
            
            xlabel('Phase Uncertainty $(Rad, \sigma)$', 'fontsize', fontsz, 'interpreter','latex')
            ylabel('Accuracy (\%)', 'fontsize', fontsz, 'interpreter','latex')
            
            title(sprintf('Accuracy of models with loss = %.3f$\\pm$%s dB/MZI\nTrained at Loss = %.3f dB/MZI, phase uncertaity = %.3f Rad',...
                SimulationSettings.losses_dB_test(l_idx), SimulationSettings.loss_diff, SimulationSettings.losses_dB_train(ii), SimulationSettings.phase_uncerts_train(jj))...
                ,'fontsize', 1.5*fontsz, 'interpreter','latex')
            
            
            savefig([FOLDER, sprintf('Matlab_Figs/AllModels_trainedAtLoss=%.3f_PhaseUncert=%.3f_Loss=%.3f.fig',...
                SimulationSettings.losses_dB_train(ii), SimulationSettings.phase_uncerts_train(jj), SimulationSettings.losses_dB_test(l_idx))])
            saveas(gcf, [FOLDER, sprintf('Matlab_Pngs/AllModels_trainedAtLoss=%.3f_PhaseUncert=%.3f_WithLoss=%.3f.png',...
                SimulationSettings.losses_dB_train(ii), SimulationSettings.phase_uncerts_train(jj), SimulationSettings.losses_dB_test(l_idx))])
            hold off;
        end
    end
end
end