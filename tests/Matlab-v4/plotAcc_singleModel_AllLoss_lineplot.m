% Function to take in data from a Neuroptica simulation (created in python
% with either ONN_Topologies_Analysis_Retrained.py or
% plotAcc_singleModel_AllLoss.py
% Plots the accuracy for all models and a single loss with varying phase
% uncertainties
%
% Author: Simon Geoffroy-Gagnon
% Edit: 25.01.2020


function plotAcc_singleModel_AllLoss_lineplot(FOLDER, sim, topo, step_sz, printMe)
fontsz = 64;

for t = 1:length(topo)
    simulation = sim.(topo{t});
    accuracy = sim.(topo{t}).accuracy_LPU;
    
    figure('Renderer', 'painters', 'Position', [400 400 1900 1400])
    same_phaseUncert = [];
    if ~simulation.same_phase_uncert
        for ii = 1:length(simulation.phase_uncert_phi)
            same_phaseUncert(ii, :) = accuracy(ii,ii,1:step_sz:end);
        end
    else
        accuracy = squeeze(accuracy);
        same_phaseUncert = accuracy(:, 1:step_sz:end);
        %         same_phaseUncert = accuracy(:, [1, 6, 11, 16, 21]);
    end
    
    plot(simulation.phase_uncert_theta, same_phaseUncert, 'linewidth', 3)
    axis square
    
    %     legend_ = create_legend_single_model(simulation.loss_dB([1, 6, 11, 16, 21]));
    legend_ = create_legend_single_model(simulation.loss_dB(1:step_sz:end));
    legend(legend_, 'fontsize', fontsz*.8, 'interpreter','latex', 'location', 'northeast');
    
    ylim([0, 100])
    
    h = gca;
    ylim([0 100])
    set(h, 'YTickLabelMode', 'auto')
    set(h, 'XTickLabelMode','auto')
    
    disp(max(max(max(accuracy))))
    
    xlabel('$\sigma_\phi = \sigma_\theta$ (Rad)', 'fontsize', fontsz, 'interpreter','latex')
    ylabel('Accuracy (\%)', 'fontsize', fontsz, 'interpreter','latex')
    
    title(sprintf('%s',simulation.topology), 'fontsize', fontsz, 'interpreter','latex')
    a = get(gca,'XTickLabel');
    set(gca,'XTickLabel',a,'FontName','Times','fontsize',fontsz*0.7)
    a = get(gca,'YTickLabel');
    set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz*0.7)
    h = gca;
    set(h, 'YTickLabelMode','auto')
    set(h, 'XTickLabelMode','auto')
    axis square
    savefig([FOLDER, sprintf('/Matlab_Figs/Model=%s_lineplot.fig', simulation.topology)])
    saveas(gcf, [FOLDER, sprintf('/Model=%s_lineplot.png', simulation.topology)])
    if printMe
        pMe([FOLDER, '/' simulation.topology, '-all-losses_lineplot.pdf'])
    end
end
