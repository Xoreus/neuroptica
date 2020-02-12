% Function to take inmax data from a Neuroptica simulation (created in python
% with either ONN_Topologies_Analysis_Retrained.py or
% plotAcc_singleModel_AllLoss.py
% Plots the accuracy for all models and a single loss with varying phase
% uncertainties
% Not also plots a line following some percentage (fig_of_merit_value) of the max accuracy
%
% Author: Simon Geoffroy-Gagnon
% Edit: 06.02.2020

function accuracy_colormap_phi_theta_plotAccuracyArea_FoM_maxAcc(FOLDER, SimulationSettings, fig_of_merit_value, showContour)
fontsz = 44;
colormaps = {'jet'}; % this is the one farhad likes % {'hot'}; % this is the one simon likes

for model_idx = 1:size(SimulationSettings.ONN_setup, 1)
    for kk = 1:length(colormaps)
        if strcmp(colormaps(kk), 'hot')
            contourColor = 'w';
        else
            contourColor = 'k';
        end
        
        modelTopo = sprintf('%s',strrep(SimulationSettings.ONN_setup(model_idx, :), ' ', ''));
        Model_acc = load([FOLDER, modelTopo, '.mat']);
        model = Model_acc.(modelTopo);
        accuracy = model.accuracy;
        
        for loss_idx = 1:size(accuracy, 3)
            figure('Renderer', 'painters', 'Position', [400 400 1800 1300])
            
            curr_acc = squeeze(accuracy(:,:,loss_idx));
            h = pcolor(model.phase_uncert_theta, model.phase_uncert_phi, curr_acc); %,'HandleVisibility','off');
            h.Annotation.LegendInformation.IconDisplayStyle = 'off';
            
            hold on
            % Create contour map of the section that is above .9 of max accuracy
            if showContour
                contour(SimulationSettings.phase_uncert_theta,SimulationSettings.phase_uncert_phi, curr_acc, ...
                    [SimulationSettings.max_accuracy*fig_of_merit_value SimulationSettings.max_accuracy*fig_of_merit_value], contourColor , 'linewidth', 4);
                % Create legend for contour map
                lgd = ['Above ', num2str(SimulationSettings.max_accuracy*fig_of_merit_value, 4), '\% accuracy'];
                legend(lgd, 'fontsize', fontsz, 'interpreter','latex');
            end
            % Calculate "area" of contour map as a figure of merit
            area_of_merit = sum(sum(curr_acc >= SimulationSettings.max_accuracy*fig_of_merit_value)) * (SimulationSettings.phase_uncert_phi(2) - ...
                SimulationSettings.phase_uncert_phi(1)) * (SimulationSettings.phase_uncert_theta(2) - SimulationSettings.phase_uncert_theta(1));
            
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
            
            title(sprintf(['Accuracy of Model with %s Topology\nLoss/MZI = %.2f dB, $\\sigma_{Loss/MZI} = $ %.2f dB\nFigure of Merit: %.4f'],model.topology,...
                SimulationSettings.loss_dB(loss_idx), SimulationSettings.loss_diff, area_of_merit), 'fontsize', fontsz, 'interpreter','latex')
%             title(sprintf('Accuracy of Model with %s Topology', model.topology), 'fontsize', fontsz, 'interpreter','latex')
            
            %             title(sprintf(['Accuracy of Model with %s Topology\nLoss/MZI = %.2f dB, $\\sigma_{Loss/MZI} = $ %.2f dB'],SimulationSettings.models{model_idx},...
            %                 SimulationSettings.loss_dB(loss_idx), str2double(SimulationSettings.loss_diff)), 'fontsize', fontsz, 'interpreter','latex')
            
            savefig([FOLDER, sprintf('Matlab_Figs/ColorMap-AreaOfMerit-Model=%s_Loss=%.3f_FoM=%.3f_cmap=%s-totalMaxAcc.fig', model.topology, ...
                SimulationSettings.loss_dB(loss_idx), fig_of_merit_value, colormaps{kk})])
            saveas(gcf, [FOLDER, sprintf('Matlab_Pngs/ColorMap-AreaOfMerit-Model=%s_Loss=%.3f_FoM=%.3f_cmap=%s-totalMaxAcc.png', model.topology, ...
                SimulationSettings.loss_dB(loss_idx), fig_of_merit_value, colormaps{kk})])
            close(gcf)
        end
    end
end
