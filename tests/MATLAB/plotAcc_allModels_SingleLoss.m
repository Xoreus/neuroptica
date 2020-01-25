% Function to take in data from a Neuroptica simulation (created in python
% with either ONN_Topologies_Analysis_Retrained.py or
% plotAcc_allModels_SingleLoss.py
% Plots the accuracy for all models and a single loss with varying phase
% uncertainties
%
% Author: Simon Geoffroy-Gagnon
% Edit: 24.01.2020


function plotAcc_allModels_SingleLoss(FOLDER)
fontsz = 28;

if ~exist([FOLDER, 'Matlab_Figs'], 'dir')
    mkdir([FOLDER, 'Matlab_Figs'])
end
if ~exist([FOLDER, 'Matlab_Pngs'], 'dir')
    mkdir([FOLDER, 'Matlab_Pngs'])
end


[N, Models, Nonlin, phase_uncert, loss_dB, ~, DATASET_NUM] = load_ONN_data(FOLDER);
models = get_model_names(Models);

for l_idx = 1:length(loss_dB)
    figure('Renderer', 'painters', 'Position', [400 400 1900 1400])
    for model_idx = 1:length(Models)
        legend_ = create_legend_single_loss(models, Nonlin);
        if ~contains(Models{model_idx}, 'N')
            Model_acc = load([FOLDER, sprintf('acc_%s_loss=%.3f_uncert=%.3f_%dFeat_%s_set%d.txt', ...
                Models{model_idx}, loss_dB(1), phase_uncert(1), N, Nonlin{1}, DATASET_NUM)]);
            
            plot(phase_uncert, Model_acc(:, l_idx), 'linewidth', 3)
            hold on
        end
    end
    legend(legend_, 'fontsize', fontsz, 'interpreter','latex');
    ylim([0, 100])
    
    a = get(gca,'XTickLabel');
    set(gca,'XTickLabel',a,'FontName','Times','fontsize',fontsz/1.2)

    a = get(gca,'YTickLabel');
    set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz/1.2)

    xlabel('Phase Uncertainty $(\sigma)$', 'fontsize', fontsz, 'interpreter','latex')
    ylabel('Accuracy (\%)', 'fontsize', fontsz, 'interpreter','latex')
    title(sprintf('Accuracy of models with loss $\\approx$ %.2f dB/MZI', loss_dB(l_idx)), 'fontsize', 1.5*fontsz, 'interpreter','latex')
    
    
    savefig([FOLDER, sprintf('Matlab_Figs/AllModels_Loss=%.3f.fig', loss_dB(l_idx))])
    saveas(gcf, [FOLDER, sprintf('Matlab_Pngs/AllModels_Loss=%.3f.png', loss_dB(l_idx))])
end

end