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
        if simulation.N == 48
            h = pcolor(simulation.phase_uncert_theta, simulation.phase_uncert_phi, curr_acc); %,'HandleVisibility','off');
        else
            h = pcolor(simulation.phase_uncert_theta, simulation.phase_uncert_phi, curr_acc); %,'HandleVisibility','off');
        end
        
        h.Annotation.LegendInformation.IconDisplayStyle = 'off';
        
        hold on
        % Create contour map of the section that is above .9 of max accuracy
        if showContour
            contour(simulation.phase_uncert_theta, simulation.phase_uncert_phi, curr_acc, ...
                [sim.max_accuracy*fig_of_merit_value sim.max_accuracy*fig_of_merit_value], 'k' , 'linewidth', 4);
            % Create legend for contour map
            lgd = ['Above ', '75', '\% accuracy'];
            % num2str(sim.max_accuracy*fig_of_merit_value, 4)
            legend(lgd, 'fontsize', fontsz*0.8, 'interpreter','latex');
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
        set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz*0.7)
        a = get(gca,'XTickLabel');
        set(gca,'XTickLabel',a,'FontName','Times','fontsize',fontsz*0.7)
        h = gca;
%         ytickformat('%.1f')
%         xtickformat('%.1f')
        xlabel('$\sigma_\theta$ (Rad)', 'fontsize', fontsz, 'interpreter','latex')
        ylabel('$\sigma_\phi$ (Rad)', 'fontsize', fontsz, 'interpreter','latex')
        set(h, 'YTickLabelMode','auto')
        set(h, 'XTickLabelMode','auto')

        axis square
        
        if print_fig_of_merit
            title(sprintf('%d$\\times$%d %s\nFoM: %.4f $\\mathrm{Rad}^2$', simulation.N, simulation.N, ...
                simulation.topo,  area_of_merit), 'fontsize', fontsz, 'interpreter','latex')
        else
            if strcmp(sim.topo{t}, 'R_D_I_P')
                simulation.topology = 'Reck + DMM + Inv. Reck';
            elseif strcmp(sim.topo{t}, 'R_D_I_P')
                simulation.topology = 'Reck + Inv. Reck';
            end
%             title(sprintf('%d$\\times$%d %s', simulation.N, simulation.N, simulation.topology), 'fontsize', fontsz, 'interpreter','latex')
            title(sprintf('FoM in $\\mathrm{Rad}^2$'), 'fontsize', fontsz, 'interpreter','latex')
        end
        
        
%         savefig([FOLDER, sprintf('/Matlab_Figs/%s_phiThetaUncert_N=%d.fig', simulation.topo, simulation.N)])
%         saveas(gcf, [FOLDER, sprintf('/%s_phiThetaUncert_N=%d.png', simulation.topo, simulation.N)])
  
        if printMe
            pMe([sprintf('../Crop_Me/%s_phiThetaUncert_N=%d.pdf', simulation.topo, simulation.N)])
        end
        if area_of_merit == 0
            break
        end
        fprintf('%s PT FoM = %.4f\n', simulation.topo, area_of_merit)
        
    end
    fom(t) = sum(a_of_m);
end
