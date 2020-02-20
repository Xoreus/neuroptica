% Function to take inmax data from a Neuroptica simulation (created in python
% with either ONN_Topologies_Analysis_Retrained.py or
% plotAcc_singleModel_AllLoss.py
% Plots the accuracy for all models and a single loss with varying phase
% uncertainties
% Not also plots a line following some percentage (fig_of_merit_value) of the max accuracy
%
% Author: Simon Geoffroy-Gagnon
% Edit: 06.02.2020

function accuracy_colormap_phaseUncert_plotAccuracyArea_FoM_maxAcc(FOLDER, SimulationSettings, fig_of_merit_value, showContour)
fontsz = 44;
colormaps = {'jet'}; % this is the one farhad likes % {'hot'}; % this is the one simon likes

for model_idx = 1:length(SimulationSettings.ONN_Setups)
    for kk = 1:length(colormaps)
        figure('Renderer', 'painters', 'Position', [400 400 1800 1300])
        
        if strcmp(colormaps(kk), 'hot')
            contourColor = 'w';
        else
            contourColor = 'k';
        end
        
        Model_acc = load([FOLDER, sprintf('acc_%s_loss=%.3f_uncert=%.3f_%sFeat.mat', ...
            SimulationSettings.ONN_Setups{model_idx}, SimulationSettings.loss_dB(1), SimulationSettings.phase_uncert_theta(1), SimulationSettings.N)]);
        accuracy = Model_acc.accuracy;
        
        if size(accuracy, 1) ~= 1
            for ii = 1:length(SimulationSettings.phase_uncert_phi)
                same_phaseUncert(ii, :) = accuracy(ii,ii,:);
            end
        else
            same_phaseUncert = squeeze(accuracy);
        end
        h = pcolor(SimulationSettings.loss_dB, SimulationSettings.phase_uncert_phi, same_phaseUncert);
        h.Annotation.LegendInformation.IconDisplayStyle = 'off';
        
        hold on
        % Create contour map of the section that is above .9 of max accuracy
        if showContour
            C = contour(SimulationSettings.loss_dB, SimulationSettings.phase_uncert_phi, same_phaseUncert, ...
                [SimulationSettings.max_accuracy*fig_of_merit_value SimulationSettings.max_accuracy*fig_of_merit_value], contourColor , 'linewidth',4);
            % Create legend for contour map
            lgd = ['Above ', num2str(SimulationSettings.max_accuracy*fig_of_merit_value, 4), '\% accuracy'];
            
            legend(lgd, 'fontsize', fontsz, 'interpreter','latex');
        end
        % Calculate "area" of contour map as a figure of merit
        area_of_merit = sum(sum(same_phaseUncert >= SimulationSettings.max_accuracy*fig_of_merit_value)) * (SimulationSettings.phase_uncert_phi(2) - ...
            SimulationSettings.phase_uncert_phi(1)) * (SimulationSettings.phase_uncert_theta(2) - SimulationSettings.phase_uncert_theta(1));
        
        
        shading('interp');
        set(gca,'YDir','normal')
        c = colorbar;
        c.Label.Interpreter = 'latex';
        c.Label.String = 'Accuracy (\%)';
        c.Label.FontSize = fontsz;
        caxis([20 100])
        colormap(colormaps{kk});
        
        xticks(SimulationSettings.loss_dB)
        a = get(gca,'XTickLabel');
        set(gca,'XTickLabel',a,'FontName','Times','fontsize',fontsz*0.9)
        a = get(gca,'YTickLabel');
        set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz*0.9)
        
        xlabel('Loss/MZI (dB)', 'fontsize', fontsz, 'interpreter','latex')
        ylabel('Phase Uncertainty $(\sigma_{\phi,\theta})$', 'fontsize', fontsz, 'interpreter','latex')
        
        %         title(sprintf(['Accuracy of Model with %s Topology\n Loss Standard Deviation $\\sigma_{Loss} = $ %s dB/MZI\nFigure of Merit: %.3f'],...
        %             SimulationSettings.models{model_idx}, SimulationSettings.loss_diff, area_of_merit), ...
        %             'fontsize', fontsz, 'interpreter','latex')
        
        title(sprintf(['Accuracy of Model with %s Topology\n Loss Standard Deviation $\\sigma_{Loss} = $ %s dB/MZI'],...
            SimulationSettings.models{model_idx}, SimulationSettings.loss_diff), ...
            'fontsize', fontsz, 'interpreter','latex')
        
        title(sprintf(['Accuracy of Model with %s Topology'],SimulationSettings.models{model_idx}), 'fontsize', fontsz, 'interpreter','latex')
        
        savefig([FOLDER, sprintf('Matlab_Figs/ColorMap-TotalPhaseUncert-FigureOfMerit-Model=%s_Loss=%.3f_FoM=%.3f_cmap=%s-totalMaxAcc.fig', SimulationSettings.ONN_Setups{model_idx}, ...
            fig_of_merit_value, colormaps{kk})])
        saveas(gcf, [FOLDER, sprintf('Matlab_Pngs/ColorMap-TotalPhaseUncert-FigureOfMerit-Model=%s_Loss=%.3f_FoM=%.3f_cmap=%s-totalMaxAcc.png', SimulationSettings.ONN_Setups{model_idx}, ...
            fig_of_merit_value, colormaps{kk})])
        close(gcf)
    end
end
end
