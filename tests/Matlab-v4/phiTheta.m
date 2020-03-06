% Function to take inmax data from a Neuropstica simulation (created in python
% with either ONN_Topologies_Analysis_Retrained.py or
% plotAcc_singleModel_AllLoss.py
% Plots the accuracy for all models and a single loss with varying phase
% uncertainties
% Not also plots a line following some percentage (fig_of_merit_value) of the max accuracy
%
% Author: Simon Geoffroy-Gagnon
% Edit: 15.02.2020

function fom = phiTheta(FOLDER, sim, topo, fig_of_merit_value, showContour, print_fig_of_merit, printMe)
fontsz = 64;
fom = zeros(1, length(topo));

for t = 1:length(topo)
    a_of_m = [];
    accuracy = sim.(topo{t}).accuracy_PT;
    simulation = sim.(topo{t});
    
    for loss_idx = 1:size(accuracy, 3)
        figure('Renderer', 'painters', 'Position', [400 400 1800 1300])
        
        curr_acc = squeeze(accuracy(:,:,loss_idx));
        h = pcolor(simulation.phase_uncert_theta, simulation.phase_uncert_phi, curr_acc); %,'HandleVisibility','off');
        h.Annotation.LegendInformation.IconDisplayStyle = 'off';
        
        hold on
        % Create contour map of the section that is above .9 of max accuracy
        if showContour
            contour(simulation.phase_uncert_theta, simulation.phase_uncert_phi, curr_acc, ...
                [sim.max_accuracy*fig_of_merit_value sim.max_accuracy*fig_of_merit_value], 'k' , 'linewidth', 4);
            % Create legend for contour map
            lgd = ['Above ', num2str(sim.max_accuracy*fig_of_merit_value, 4), '\% accuracy'];
            legend(lgd, 'fontsize', fontsz, 'interpreter','latex');
        end
        % Calculate "area" of contour map as a figure of merit
        area_of_merit = sum(sum(curr_acc >= sim.max_accuracy*fig_of_merit_value)) * (simulation.phase_uncert_phi(2) - ...
            simulation.phase_uncert_phi(1)) * (simulation.phase_uncert_theta(2) - simulation.phase_uncert_theta(1));
        
        %         shading('interp');
        set(h, 'EdgeColor', 'none');
        set(gca,'YDir','normal')
        c = colorbar;
        c.Label.Interpreter = 'latex';
        c.Label.String = 'Accuracy (\%)';
        c.Label.FontSize = fontsz;
        caxis([100/(simulation.N) 100])
        colormap('jet');
        

        a = get(gca,'YTickLabel');
        set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz)
        h = gca;
        set(h, 'YTickLabelMode','auto')
        set(h, 'XTickLabelMode','auto')
        ytickformat('%.1f')
        xtickformat('%.1f')
        xlabel('$\sigma_\theta$ (Rad)', 'fontsize', fontsz, 'interpreter','latex')
        ylabel('$\sigma_\phi$ (Rad)', 'fontsize', fontsz, 'interpreter','latex')
        axis square
        if print_fig_of_merit
            title(sprintf(['Accuracy of %s Topology\nLoss/MZI = %.2f dB, $\\sigma_{Loss/MZI} = $ %.2f dB\nFigure of Merit: %.6f'],simulation.topology,...
                simulation.loss_dB(loss_idx), simulation.loss_diff, area_of_merit), 'fontsize', fontsz, 'interpreter','latex')
        else
            title(sprintf('%d$\\times$%d %s', simulation.N, simulation.N, simulation.topology), 'fontsize', fontsz, 'interpreter','latex')
        end
        
        savefig([FOLDER, sprintf('/Matlab_Figs/%s_phiThetaUncert_N=%d.fig', simulation.topology, simulation.N)])
        saveas(gcf, [FOLDER, sprintf('/%s_phiThetaUncert_N=%d.png', simulation.topology, simulation.N)])
        
        if printMe
            pMe([FOLDER, sprintf('/%s_phiThetaUncert_N=%d.pdf', simulation.topology, simulation.N)])
        end
        if area_of_merit == 0
            break
        end
        disp(area_of_merit)
        
    end
    fom(t) = sum(a_of_m);
end
