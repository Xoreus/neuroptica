% Function to take inmax data from a Neuroptica simulation (created in python
% with either ONN_Topologies_Analysis_Retrained.py or
% plotAcc_singleModel_AllLoss.py
% Plots the accuracy for all models and a single loss with varying phase
% uncertainties
% Not also plots a line following some percentage (fig_of_merit_value) of the max accuracy
%
% Author: Simon Geoffroy-Gagnon
% Edit: 15.02.2020

function loss_phaseUncert(FOLDER, sim, topo, fig_of_merit_value, showContour, print_fig_of_merit, printMe)
fontsz = 64;

for t = 1:length(topo)
    accuracy = sim.(topo{t}).accuracy_LPU;
    simulation = sim.(topo{t});
    
    figure('Renderer', 'painters', 'Position', [400 400 1800 1300])
    same_phaseUncert = squeeze(accuracy);
    if simulation.N == 48
        h = pcolor(simulation.loss_dB(1:31), simulation.phase_uncert_theta(1:17), same_phaseUncert(1:17, 1:31));
    else
        h = pcolor(simulation.loss_dB, simulation.phase_uncert_theta, same_phaseUncert);
    end
    
    h.Annotation.LegendInformation.IconDisplayStyle = 'off';
    
    hold on
    % Create contour map of the section that is above .9 of max accuracy
    if showContour
        C = contour(simulation.loss_dB, simulation.phase_uncert_phi, same_phaseUncert, ...
            [sim.max_accuracy*fig_of_merit_value sim.max_accuracy*fig_of_merit_value], 'k' , 'linewidth',4);
        % Create legend for contour map
        lgd = ['Above ', '75' , '\% accuracy'];
        % num2str(sim.max_accuracy*fig_of_merit_value, 4)
        legend(lgd, 'fontsize', fontsz*0.8, 'interpreter','latex');
    end
    % Calculate "area" of contour map as a figure of merit
    area_of_merit = sum(sum(same_phaseUncert >= sim.max_accuracy*fig_of_merit_value)) * (simulation.phase_uncert_theta(2) - ...
        simulation.phase_uncert_theta(1)) * (simulation.phase_uncert_theta(2) - simulation.phase_uncert_theta(1));
    
    set(h, 'EdgeColor', 'none');
    
    set(gca,'YDir','normal')
    c = colorbar;
    c.Label.Interpreter = 'latex';
    c.Label.String = 'Accuracy (\%)';
    c.Label.FontSize = fontsz;
    caxis([50 95])
    caxis([100/(simulation.N) 100])
    colormap('jet');
    
    a = get(gca,'YTickLabel');
    set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz*0.7)
    a = get(gca,'XTickLabel');
    set(gca,'XTickLabel',a,'FontName','Times','fontsize',fontsz*0.7)
    h = gca;
    set(h, 'YTickLabelMode','auto')
    set(h, 'XTickLabelMode','auto')
    axis square
%     ytickformat('%.1f')
%     xtickformat('%.1f')
    xlabel('Loss/MZI (dB)', 'fontsize', fontsz, 'interpreter','latex')
    ylabel('$\sigma_\phi,\;\sigma_\theta$ (Rad)', 'fontsize', fontsz, 'interpreter','latex')
    %     ylabel('$\sigma$ (Rad)', 'fontsize', fontsz, 'interpreter','latex')
    
    
    if print_fig_of_merit
        title(sprintf('%d$\\times$%d %s\nFoM: %.4f $\\mathrm{Rad} \\cdot \\mathrm{dB}$', simulation.N, simulation.N, ...
            simulation.topology,  area_of_merit), 'fontsize', fontsz, 'interpreter','latex')
    else
        if strcmp(sim.topo, 'R_D_I_P')
            simulation.topology = 'Reck + DMM + Inv. Reck';
        end
%         title(sprintf('%d$\\times$%d %s', simulation.N, simulation.N, simulation.topology), 'fontsize', fontsz, 'interpreter','latex')
%         title(sprintf('FoM in $\\mathrm{Rad} \\cdot \\mathrm{dB}$'), 'fontsize', fontsz, 'interpreter','latex')
    end

    if printMe
        pMe([sprintf('../Crop_Me/%s_lossPhaseUncert_N=%d.pdf', simulation.topo, simulation.N)])
    end
    fprintf('%s LPU FoM = %.6f\n', simulation.topo, area_of_merit)
end
end
