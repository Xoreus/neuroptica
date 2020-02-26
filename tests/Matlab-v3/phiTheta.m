% Function to take inmax data from a Neuropstica simulation (created in python
% with either ONN_Topologies_Analysis_Retrained.py or
% plotAcc_singleModel_AllLoss.py
% Plots the accuracy for all models and a single loss with varying phase
% uncertainties
% Not also plots a line following some percentage (fig_of_merit_value) of the max accuracy
%
% Author: Simon Geoffroy-Gagnon
% Edit: 15.02.2020

function fom = phiTheta(FOLDER, sim, acc, topo, fig_of_merit_value, showContour, print_fig_of_merit, printMe)
fontsz = 44;
fom = zeros(1, length(topo));

for t = 1:length(topo)
    a_of_m = [];
    accuracy = acc.(topo{t}).accuracy;
    simulation = sim.(topo{t});
    
    for loss_idx = 1:4:size(accuracy, 3)
        figure('Renderer', 'painters', 'Position', [400 400 1800 1300])
        
        curr_acc = squeeze(accuracy(:,:,loss_idx));
        h = pcolor(simulation.phase_uncert_theta, simulation.phase_uncert_phi, curr_acc); %,'HandleVisibility','off');
        h.Annotation.LegendInformation.IconDisplayStyle = 'off';
        
        hold on
        % Create contour map of the section that is above .9 of max accuracy
        if showContour
            contour(simulation.phase_uncert_theta, simulation.phase_uncert_phi, curr_acc, ...
                [acc.max_accuracy*fig_of_merit_value acc.max_accuracy*fig_of_merit_value], 'k' , 'linewidth', 4);
            % Create legend for contour map
            lgd = ['Above ', num2str(acc.max_accuracy*fig_of_merit_value, 4), '\% accuracy'];
            legend(lgd, 'fontsize', fontsz, 'interpreter','latex');
        end
        % Calculate "area" of contour map as a figure of merit
        area_of_merit = sum(sum(curr_acc >= acc.max_accuracy*fig_of_merit_value)) * (simulation.phase_uncert_phi(2) - ...
            simulation.phase_uncert_phi(1)) * (simulation.phase_uncert_theta(2) - simulation.phase_uncert_theta(1));
        
%         shading('interp');
        set(h, 'EdgeColor', 'none');
        set(gca,'YDir','normal')
        c = colorbar;
        c.Label.Interpreter = 'latex';
        c.Label.String = 'Accuracy (\%)';
        c.Label.FontSize = fontsz;
        caxis([100/(simulation.N+1) 100])
        colormap('jet');
        
        a = get(gca,'XTickLabel');
        set(gca,'XTickLabel',a,'FontName','Times','fontsize',fontsz*0.7)
%         a = get(gca,'YTickLabel');
%         set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz*0.7)
        
        xlabel('$\sigma_\theta$ (Rad)', 'fontsize', fontsz, 'interpreter','latex')
        ylabel('$\sigma_\phi$ (Rad)', 'fontsize', fontsz, 'interpreter','latex')
        
        if print_fig_of_merit
            title(sprintf(['Accuracy of %s Topology\nLoss/MZI = %.2f dB, $\\sigma_{Loss/MZI} = $ %.2f dB\nFigure of Merit: %.6f'],simulation.topology,...
                simulation.loss_dB(loss_idx), simulation.loss_diff, area_of_merit), 'fontsize', fontsz, 'interpreter','latex')
        else
            title(sprintf('Accuracy of %s Topology', simulation.topology), 'fontsize', fontsz, 'interpreter','latex')
        end
        
        savefig([FOLDER, sprintf('Matlab_Figs/%s_phiThetaUncert_loss=%.2fdB.fig', simulation.onn_topo, simulation.loss_dB(loss_idx))])
        saveas(gcf, [FOLDER, sprintf('Matlab_Pngs/%s_phiThetaUncert_loss=%.2fdB.png', simulation.onn_topo, simulation.loss_dB(loss_idx))])

        if printMe        
            pMe([FOLDER, sprintf('%s_phiThetaUncert_loss=%.2fdB.png', simulation.onn_topo, simulation.loss_dB(loss_idx))])
        end
        if area_of_merit == 0
            break
        end
        
    end
    fom(t) = sum(a_of_m);
end
