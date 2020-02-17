% Function to take in data from a Neuroptica simulation (created in python
% with either ONN_Topologies_Analysis_Retrained.py or
% plotAcc_allModels_SinglePhaseUncert.py
% Plots the accuracy for all models and a single phase uncert with loss
%
% Author: Simon Geoffroy-Gagnon
% Edit: 25.01.2020


function plotAcc_allModels_SinglePhaseUncert(FOLDER, SimulationSettings)
fontsz = 44;
step_sz = 1;
legend_ = {};
figure('Renderer', 'painters', 'Position', [400 400 1900 1400])
for p_idx = 1:length(SimulationSettings.phase_uncert_theta(1))
    for model_idx = 1:size(SimulationSettings.ONN_setup, 1)
        
        modelTopo = sprintf('%s',strrep(SimulationSettings.ONN_setup(model_idx, :), ' ', ''));
        Model_acc = load([FOLDER, modelTopo, '.mat']);
        model = Model_acc.(modelTopo);
        accuracy = model.accuracy;
        legend_{end+1} = model.topology;
        if ~model.same_phase_uncert
            for ii = 1:length(SimulationSettings.phase_uncert_phi)
                same_phaseUncert(ii, :) = accuracy(ii,ii,1:step_sz:end);
            end
        else
            accuracy = squeeze(accuracy);
            same_phaseUncert = accuracy(:, 1:step_sz:end);
        end
        plot(SimulationSettings.loss_dB, same_phaseUncert(p_idx, :), 'linewidth', 3)
        
        hold on
    end
    
    hold off
    legend(legend_, 'fontsize', fontsz,  'interpreter','latex', 'location', 'best');
    
    a = get(gca,'XTickLabel');
    set(gca,'XTickLabel',a,'FontName','Times','fontsize',fontsz*0.9)
    a = get(gca,'YTickLabel');
    set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz*0.9)
    
    xlabel(sprintf('Loss (dB/MZI)'), 'fontsize', fontsz, 'interpreter','latex')
    ylabel('Accuracy (\%)', 'fontsize', fontsz, 'interpreter','latex')
    
    title(sprintf('Accuracy vs Loss/MZIs'), 'fontsize', 1.5*fontsz, 'interpreter','latex')
    
    axis('tight')
    ylim([0, 100])
    
    savefig([FOLDER, sprintf('Matlab_Figs/AllModels_PhaseUncert=%.3f.fig', SimulationSettings.phase_uncert_theta(p_idx))])
    saveas(gcf, [FOLDER, sprintf('Matlab_Pngs/AllModels_PhaseUncert=%.3f.png', SimulationSettings.phase_uncert_theta(p_idx))])
end

end