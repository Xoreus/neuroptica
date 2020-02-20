% Function to take in data from a Neuroptica simulation (created in python
% with either ONN_Topologies_Analysis_Retrained.py or
% plotAcc_singleModel_AllLoss.py
% Plots the accuracy for all models and a single loss with varying phase
% uncertainties
%
% Author: Simon Geoffroy-Gagnon
% Edit: 25.01.2020


function plotAcc_singleModel_AllLoss(FOLDER, sim, acc, topo)
fontsz = 44;
step_sz = 1;

for t = 1:length(topo)
    simulation = sim.(topo{t});
    accuracy = acc.(topo{t}).accuracy;
    
    figure('Renderer', 'painters', 'Position', [400 400 1900 1400])
    
    if ~simulation.same_phase_uncert
        for ii = 1:length(simulation.phase_uncert_phi)
            same_phaseUncert(ii, :) = accuracy(ii,ii,1:step_sz:end);
        end
    else
        accuracy = squeeze(accuracy);
        same_phaseUncert = accuracy(:, 1:step_sz:end);
    end
    
    plot(simulation.phase_uncert_theta, same_phaseUncert, 'linewidth', 3)
    
    legend_ = create_legend_single_model(simulation.loss_dB(1:step_sz:end));
    legend(legend_, 'fontsize', fontsz, 'interpreter','latex', 'location', 'best');
    axis tight
    ylim([0, 100])
    
    disp(max(max(max(accuracy))))
    
    xlabel('$(\sigma_{\phi,\theta})$ (Rad)', 'fontsize', fontsz, 'interpreter','latex')
    ylabel('Accuracy (\%)', 'fontsize', fontsz, 'interpreter','latex')
    
    title(sprintf('Accuracy of %s Topology',simulation.topology), 'fontsize', fontsz, 'interpreter','latex')
    a = get(gca,'XTickLabel');
    set(gca,'XTickLabel',a,'FontName','Times','fontsize',fontsz*0.9)
    a = get(gca,'YTickLabel');
    set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz*0.8)
    savefig([FOLDER, sprintf('Matlab_Figs/Model=%s_lineplot.fig', simulation.topology)])
    saveas(gcf, [FOLDER, sprintf('Matlab_Pngs/Model=%s_lineplot.png', simulation.topology)])
end
