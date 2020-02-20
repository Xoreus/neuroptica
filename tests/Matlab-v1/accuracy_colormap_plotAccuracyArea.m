% Function to take inmax data from a Neuroptica simulation (created in python
% with either ONN_Topologies_Analysis_Retrained.py or
% plotAcc_singleModel_AllLoss.py
% Plots the accuracy for all models and a single loss with varying phase
% uncertainties
% Not also plots a line following 90% of the max accuracy
%
% Author: Simon Geoffroy-Gagnon
% Edit: 06.02.2020

function accuracy_colormap_plotAccuracyArea(FOLDER, SimulationSettings, fig_of_merit_value)
fontsz = 38;
colormaps =  {'hot'}; % this is the one simon likes % {'jet'}; % this is the one farhad likes %

for model_idx = 1:length(SimulationSettings.ONN_Setups)
    
    for kk = 1:length(colormaps)
        if strcmp(colormaps(kk), 'hot')
            contourColor = 'w';
        else
            contourColor = 'k';
        end
        
        Model_acc = load([FOLDER, sprintf('acc_%s_loss=%.3f_uncert=%.3f_%sFeat.mat', ...
            SimulationSettings.ONN_Setups{model_idx}, SimulationSettings.loss_dB(1), SimulationSettings.phase_uncert_theta(1), SimulationSettings.N)]);
        accuracy = Model_acc.accuracy;
        
        for loss_idx = 1:size(accuracy, 3)
            figure('Renderer', 'painters', 'Position', [400 400 1800 1300])
            
            curr_acc = squeeze(accuracy(:,:,loss_idx));
            h = pcolor(SimulationSettings.phase_uncert_theta, SimulationSettings.phase_uncert_phi, curr_acc); %,'HandleVisibility','off');
            h.Annotation.LegendInformation.IconDisplayStyle = 'off';
            
            hold on
            % Create contour map of the section that is above .9 of max
            % accuracy
            C = contour(SimulationSettings.phase_uncert_theta,SimulationSettings.phase_uncert_phi, curr_acc, ...
                [max(max(curr_acc))*fig_of_merit_value max(max(curr_acc))*fig_of_merit_value], contourColor , 'linewidth',4);
            % Calculate "area" of contour map as a figure of merit
            fig_of_merit = sum(sum(curr_acc >= max(max(curr_acc))*fig_of_merit_value))*(SimulationSettings.phase_uncert_phi(2) - ...ss
                SimulationSettings.phase_uncert_phi(1)) * (SimulationSettings.phase_uncert_theta(2) - SimulationSettings.phase_uncert_theta(1)) * ...
                (max(max(curr_acc))/SimulationSettings.max_accuracy)^2;
            
            % Create legend for contour map
            lgd = ['Above ', num2str(max(max(curr_acc))*fig_of_merit_value, 3), '\% accuracy'];
            
            legend(lgd, 'fontsize', fontsz, 'interpreter','latex');
            
            shading('interp');
            set(gca,'YDir','normal')
            c = colorbar;
            c.Label.Interpreter = 'latex';
            c.Label.String = 'Accuracy (\%)';
            c.Label.FontSize = fontsz;
            caxis([20 100])
            colormap(colormaps{kk});
            
            a = get(gca,'XTickLabel');
            set(gca,'XTickLabel',a,'FontName','Times','fontsize',fontsz*0.9)
            a = get(gca,'YTickLabel');
            set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz*0.9)
            
            xlabel('Theta Phase Uncertainty $(\sigma_\theta)$', 'fontsize', fontsz, 'interpreter','latex')
            ylabel('Phi Phase Uncertainty $(\sigma_\phi)$', 'fontsize', fontsz, 'interpreter','latex')
            title(sprintf(['Accuracy of Model with %s Topology\nLoss/MZI = %.2f dB, $\\sigma_{Loss/MZI} = $ %.2f dB\nFigure of Merit: %.4f'],SimulationSettings.models{model_idx},...
                SimulationSettings.loss_dB(loss_idx), str2double(SimulationSettings.loss_diff), fig_of_merit), 'fontsize', fontsz, 'interpreter','latex')
            
            savefig([FOLDER, sprintf('Matlab_Figs/ColorMap-FigureOfMerit-Model=%s_Loss=%.3f_FoM=%.3f_cmap=%s.fig', SimulationSettings.ONN_Setups{model_idx}, ...
                SimulationSettings.loss_dB(loss_idx), fig_of_merit_value, colormaps{kk})])
            saveas(gcf, [FOLDER, sprintf('Matlab_Pngs/ColorMap-FigureOfMerit-Model=%s_Loss=%.3f_FoM=%.3f_cmap=%s.png', SimulationSettings.ONN_Setups{model_idx}, ...
                SimulationSettings.loss_dB(loss_idx), fig_of_merit_value, colormaps{kk})])
            close(gcf)
        end
    end
end
