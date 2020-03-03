% Function to take inmax data from a Neuroptica simulation (created in python
% with either ONN_Topologies_Analysis_Retrained.py or
% plotAcc_singleModel_AllLoss.py
% Plots the accuracy for all models and a single loss with varying phase
% uncertainties
% Not also plots a line following some percentage (fig_of_merit_value) of the max accuracy
%
% Author: Simon Geoffroy-Gagnon
% Edit: 15.02.2020

function loss_phaseUncert(FOLDER, sim, acc, topo, fig_of_merit_value, showContour, print_fig_of_merit, printMe)
fontsz = 64;

for t = 1:length(topo)
    accuracy = acc.(topo{t}).accuracy;
    simulation = sim.(topo{t});
    
    figure('Renderer', 'painters', 'Position', [400 400 1800 1300])
    same_phaseUncert = [];
    if size(accuracy, 1) ~= 1
        for ii = 1:length(simulation.phase_uncert_phi)
            same_phaseUncert(ii, :) = accuracy(ii,ii,:);
        end
    else
        same_phaseUncert = squeeze(accuracy);
    end
    
    h = pcolor(simulation.loss_dB, simulation.phase_uncert_theta, same_phaseUncert);
    h.Annotation.LegendInformation.IconDisplayStyle = 'off';
    
    hold on
    % Create contour map of the section that is above .9 of max accuracy
    if showContour
        C = contour(simulation.loss_dB, simulation.phase_uncert_phi, same_phaseUncert, ...
            [acc.max_accuracy*fig_of_merit_value acc.max_accuracy*fig_of_merit_value], 'k' , 'linewidth',4);
        % Create legend for contour map
        lgd = ['Above ', num2str(acc.max_accuracy*fig_of_merit_value, 4), '\% accuracy'];
        
        legend(lgd, 'fontsize', fontsz, 'interpreter','latex');
    end
    % Calculate "area" of contour map as a figure of merit
    area_of_merit = sum(sum(same_phaseUncert >= acc.max_accuracy*fig_of_merit_value)) * (simulation.phase_uncert_theta(2) - ...
        simulation.phase_uncert_theta(1)) * (simulation.phase_uncert_theta(2) - simulation.phase_uncert_theta(1));
    
    set(h, 'EdgeColor', 'none');
    
    set(gca,'YDir','normal')
    c = colorbar;
    c.Label.Interpreter = 'latex';
    c.Label.String = 'Accuracy (\%)';
    c.Label.FontSize = fontsz;
    caxis([100/(simulation.N + 1) 100])
    colormap('jet');
    
    a = get(gca,'XTickLabel');
    set(gca,'XTickLabel',a,'FontName','Times','fontsize',fontsz*0.7)
    a = get(gca,'YTickLabel');
    set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz*0.7)
    h = gca;
    set(h, 'YTickLabelMode','auto')
    set(h, 'XTickLabelMode','auto')
    
    xlabel('Loss/MZI (dB)', 'fontsize', fontsz, 'interpreter','latex')
    ylabel('$\sigma_\phi = \sigma_\theta$ (Rad)', 'fontsize', fontsz, 'interpreter','latex')
    %     ylabel('$\sigma$ (Rad)', 'fontsize', fontsz, 'interpreter','latex')
    
    if print_fig_of_merit
        title(sprintf(['Accuracy of %s Topology\n Loss Standard Deviation $\\sigma_{Loss} = $ %.2f dB\nFigure of Merit: %.5f'],...
            simulation.topology, simulation.loss_diff, area_of_merit), 'fontsize', fontsz, 'interpreter','latex')
    else
            title(sprintf('%d$\\times$%d %s', simulation.N, simulation.N, simulation.topology), 'fontsize', fontsz, 'interpreter','latex')
    end
    
    savefig([FOLDER, sprintf('Matlab_Figs/%s_power-phaseUncert_N=%d.fig', simulation.topology, simulation.N)])
    saveas(gcf, [FOLDER, sprintf('%s_power-phaseUncert_N=%d.png', simulation.topology, simulation.N)])
    
    if printMe
        pMe([FOLDER, simulation.topology, sprintf('-loss_phaseNoise_N=%d.pdf', simulation.N)])
    end
    disp(area_of_merit)
end
end
