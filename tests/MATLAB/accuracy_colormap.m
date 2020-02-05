% Function to take inmax data from a Neuroptica simulation (created in python
% with either ONN_Topologies_Analysis_Retrained.py or
% plotAcc_singleModel_AllLoss.py
% Plots the accuracy for all models and a single loss with varying phase
% uncertainties
%
% Author: Simon Geoffroy-Gagnon
% Edit: 25.01.2020

%TODO: Create line plots on top of the accuracy map

function accuracy_colormap(FOLDER, SimulationSettings)
fontsz = 28;
figure('Renderer', 'painters', 'Position', [400 400 1800 1300])
colormaps = {'hot'};
for model_idx = 1:length(SimulationSettings.ONN_Setups)
    for ii = 1:length(colormaps)
        
        Model_acc = load([FOLDER, sprintf('acc_%s_loss=%.3f_uncert=%.3f_%sFeat.mat', ...
            SimulationSettings.ONN_Setups{model_idx}, SimulationSettings.loss_dB(1), SimulationSettings.phase_uncert_theta(1), SimulationSettings.N)]);
        accuracy = Model_acc.accuracy;
        
        for loss_idx = 1:size(accuracy, 3)
            pcolor(SimulationSettings.phase_uncert_theta, SimulationSettings.phase_uncert_phi, squeeze(accuracy(:,:,loss_idx)));
            shading('interp');
            set(gca,'YDir','normal')
            c = colorbar;
            c.Label.String = 'Accuracy (%)';
            caxis([20 100])
            colormap(colormaps{ii});
            
            a = get(gca,'XTickLabel');
            set(gca,'XTickLabel',a,'FontName','Times','fontsize',fontsz)
            a = get(gca,'YTickLabel');
            set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz)
            
            xlabel('Theta Phase Uncertainty $(\sigma_\theta)$', 'fontsize', 1.5*fontsz, 'interpreter','latex')
            ylabel('Phi Phase Uncertainty $(\sigma_\phi)$', 'fontsize', 1.5*fontsz, 'interpreter','latex')
            
            title(sprintf('Accuracy of Model with %s Topology\n Loss Standard Deviation $\\sigma_{Loss} = $ %s dB/MZI\n At Loss per MZI of %.3f',SimulationSettings.models{model_idx},...
                SimulationSettings.loss_diff, SimulationSettings.loss_dB(loss_idx)), 'fontsize', 1.5*fontsz, 'interpreter','latex')
            
            drawnow;
            
            savefig([FOLDER, sprintf('Matlab_Figs/ColorMap-Model=%s_Loss=%.3f.fig', SimulationSettings.ONN_Setups{model_idx}, ...
                SimulationSettings.loss_dB(loss_idx))])
            saveas(gcf, [FOLDER, sprintf('Matlab_Pngs/ColorMap-Model=%s_Loss=%.3f.png', SimulationSettings.ONN_Setups{model_idx}, ...
                SimulationSettings.loss_dB(loss_idx))])
        end
    end
end
