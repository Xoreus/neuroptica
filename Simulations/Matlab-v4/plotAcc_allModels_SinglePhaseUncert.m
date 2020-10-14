% Function to take in data from a Neuroptica simulation (created in python
% with either ONN_Topologies_Analysis_Retrained.py or
% plotAcc_allModels_SinglePhaseUncert.py
% Plots the accuracy for all models and a single phase uncert with loss
%
% Author: Simon Geoffroy-Gagnon
% Edit: 20.02.2020

function plotAcc_allModels_SinglePhaseUncert(~, sim, topo, printMe)
fontsz = 64;
step_sz = 1;
stop_at = 8;
legend_ = {};
figure('Renderer', 'painters', 'Position', [200 200 1200 800])
for t = topo
    simulation = sim.(t{1});
    accuracy = sim.(t{1}).accuracy_LPU;
    for p_idx = 1
        
        legend_{end+1} = simulation.topo;
        same_phaseUncert = [];
        
        accuracy = squeeze(accuracy);
        same_phaseUncert = accuracy(:, 1:step_sz:end);
        
        plot(simulation.loss_dB(1:step_sz:end - stop_at), same_phaseUncert(p_idx, 1:end - stop_at), 'linewidth', 3)
        
        hold on
    end
end
hold off
legend_ = {'Reck','Clements','Diamond'};
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
% savefig([FOLDER, sprintf('/Matlab_Figs/AllModels_loss.fig')])
% saveas(gcf, [FOLDER, sprintf('/Matlab_Pngs/AllModels_loss.png')])

if printMe
    pMe(['../Crop_Me', '/singlePhaseUncert.pdf'])
end

end