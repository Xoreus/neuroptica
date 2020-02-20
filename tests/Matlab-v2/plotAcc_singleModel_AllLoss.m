% Function to take in data from a Neuroptica simulation (created in python
% with either ONN_Topologies_Analysis_Retrained.py or
% plotAcc_singleModel_AllLoss.py
% Plots the accuracy for all models and a single loss with varying phase
% uncertainties
%
% Author: Simon Geoffroy-Gagnon
% Edit: 25.01.2020


function plotAcc_singleModel_AllLoss(FOLDER, SimulationSettings)
fontsz = 44;
step_sz = 2;
% SimulationSettings.loss_dB = SimulationSettings.loss_dB;

for model_idx = 1:size(SimulationSettings.ONN_setup, 1)
    figure('Renderer', 'painters', 'Position', [400 400 1900 1400])

    modelTopo = sprintf('%s',strrep(SimulationSettings.ONN_setup(model_idx, :), ' ', ''));
    Model_acc = load([FOLDER, modelTopo, '.mat']);
    model = Model_acc.(modelTopo);
    accuracy = model.accuracy;
    if ~model.same_phase_uncert
        for ii = 1:length(SimulationSettings.phase_uncert_phi)
            same_phaseUncert(ii, :) = accuracy(ii,ii,1:step_sz:end);
        end
    else
        accuracy = squeeze(accuracy);
        same_phaseUncert = accuracy(:, 1:step_sz:end);
    end
    
    plot(SimulationSettings.phase_uncert_theta, same_phaseUncert, 'linewidth', 3)
    
    legend_ = create_legend_single_model(SimulationSettings.loss_dB(1:step_sz:end));
    legend(legend_, 'fontsize', fontsz, 'interpreter','latex', 'location', 'best');
    axis tight
    ylim([0, 100])
    
    disp(max(max(max(accuracy))))
    
    xlabel('$(\sigma_{\phi,\theta})$ (Rad)', 'fontsize', fontsz, 'interpreter','latex')
    ylabel('Accuracy (\%)', 'fontsize', fontsz, 'interpreter','latex')
    
    %     title(sprintf('Accuracy of Model with %s Topology\n Loss Standard Deviation $\\sigma_{Loss} = $ %s dB/MZI',SimulationSettings.models{model_idx},...
    %         SimulationSettings.loss_diff), 'fontsize', fontsz, 'interpreter','latex')
    title(sprintf('Accuracy of %s Topology',model.topology), 'fontsize', fontsz, 'interpreter','latex')
    a = get(gca,'XTickLabel');
    set(gca,'XTickLabel',a,'FontName','Times','fontsize',fontsz*0.9)
    a = get(gca,'YTickLabel');
    set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz*0.8)
    savefig([FOLDER, sprintf('Matlab_Figs/Model=%s_Loss=[%.3f-%.2f].fig', model.topology, ...
        min(SimulationSettings.loss_dB), max(SimulationSettings.loss_dB))])
    saveas(gcf, [FOLDER, sprintf('Matlab_Pngs/Model=%s_Loss=[%.3f-%.2f].png', model.topology, ...
        min(SimulationSettings.loss_dB), max(SimulationSettings.loss_dB))])
end
