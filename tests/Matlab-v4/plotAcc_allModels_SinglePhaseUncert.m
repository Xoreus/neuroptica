% Function to take in data from a Neuroptica simulation (created in python
% with either ONN_Topologies_Analysis_Retrained.py or
% plotAcc_allModels_SinglePhaseUncert.py
% Plots the accuracy for all models and a single phase uncert with loss
%
% Author: Simon Geoffroy-Gagnon
% Edit: 20.02.2020

function plotAcc_allModels_SinglePhaseUncert(FOLDER, sim, acc, topo)
fontsz = 44;
step_sz = 1;
legend_ = {};
figure('Renderer', 'painters', 'Position', [400 400 1900 1400])

for t = 1:length(topo)
    simulation = sim.(topo{t});
    accuracy = acc.(topo{t}).accuracy;
    for p_idx = 1:length(simulation.phase_uncert_theta(1))
        
        legend_{end+1} = simulation.topology;
        same_phaseUncert = [];
        
        if ~simulation.same_phase_uncert
            for ii = 1:length(simulation.phase_uncert_phi)
                same_phaseUncert(ii, :) = accuracy(ii,ii,1:step_sz:end);
            end
        else
            accuracy = squeeze(accuracy);
            same_phaseUncert = accuracy(:, 1:step_sz:end);
        end
        plot(simulation.loss_dB(1:step_sz:end), same_phaseUncert(p_idx, :), 'linewidth', 3)
        
        hold on
    end
end
hold off
legend(legend_, 'fontsize', fontsz,  'interpreter','latex', 'location', 'best');


xlabel(sprintf('Loss (dB/MZI)'), 'fontsize', fontsz, 'interpreter','latex')
ylabel('Accuracy (\%)', 'fontsize', fontsz, 'interpreter','latex')

title(sprintf('Accuracy vs Loss/MZI'), 'fontsize', 1.5*fontsz, 'interpreter','latex')
axis('tight')
ylim([0, 100])

a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',fontsz*0.9)
a = get(gca,'YTickLabel');
set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz*0.8)

savefig([FOLDER, sprintf('Matlab_Figs/AllModels_loss.fig')])
saveas(gcf, [FOLDER, sprintf('Matlab_Pngs/AllModels_loss.png')])

if printMe
    pMe([FOLDER, 'singlePhaseUncert.pdf'])
end

end