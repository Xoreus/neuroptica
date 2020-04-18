% Function to take in data from a Neuroptica simulation (created in python
% with either ONN_Topologies_Analysis_Retrained.py or
% plotAcc_allModels_SingleLoss.py
% Plots the accuracy for all models and a single loss with varying phase
% uncertainties
%
% Author: Simon Geoffroy-Gagnon
% Edit: 25.01.2020


function plotAcc_allModels_SingleLoss(F, sim, topo, printMe)
fontsz = 64;
step_sz = 1;
legend_ = {};
figure('Renderer', 'painters', 'Position', [400 400 1900 1400])

for t = topo([2,end])
    simulation = sim.(t{1});
    accuracy = sim.(t{1}).accuracy_LPU;
    for p_idx = 1
        
        legend_{end+1} = simulation.topology;
        same_phaseUncert = [];
        
        accuracy = squeeze(accuracy);
        same_phaseUncert = accuracy(:, 1:step_sz:end);
        
        plot(simulation.phase_uncert_theta(1:step_sz:end-20), same_phaseUncert(p_idx, 1:end-20), 'linewidth', 3)
        
        hold on
    end
end
hold off
legend(legend_, 'fontsize', fontsz*0.8,  'interpreter','latex', 'location', 'best');


xlabel(sprintf('Loss/MZI (dB)'), 'fontsize', fontsz, 'interpreter','latex')
ylabel('Accuracy (\%)', 'fontsize', fontsz, 'interpreter','latex')

% title(sprintf('Accuracy vs Loss/MZI'), 'fontsize', fontsz, 'interpreter','latex')
axis('tight')
ylim([0, 100])

a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',fontsz*0.7)
a = get(gca,'YTickLabel');
set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz*0.7)

h = gca;
set(h, 'YTickLabelMode','auto')
set(h, 'XTickLabelMode','auto')
% axis square
savefig([FOLDER, sprintf('/Matlab_Figs/AllModels_loss.fig')])
saveas(gcf, [FOLDER, sprintf('/Matlab_Pngs/AllModels_loss.png')])

if printMe
    pMe([FOLDER, '/singleLoss.pdf'])
end

end