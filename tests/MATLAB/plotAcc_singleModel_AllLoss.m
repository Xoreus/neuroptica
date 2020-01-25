% Function to take in data from a Neuroptica simulation (created in python
% with either ONN_Topologies_Analysis_Retrained.py or
% plotAcc_singleModel_AllLoss.py
% Plots the accuracy for all models and a single loss with varying phase
% uncertainties
%
% Author: Simon Geoffroy-Gagnon
% Edit: 24.01.2020


function plotAcc_singleModel_AllLoss(FOLDER)
fontsz = 28;

if ~exist([FOLDER, 'Matlab_Figs'], 'dir')
    mkdir([FOLDER, 'Matlab_Figs'])
end
if ~exist([FOLDER, 'Matlab_Pngs'], 'dir')
    mkdir([FOLDER, 'Matlab_Pngs'])
end

[N, Models, Nonlin, phase_uncert, loss_dB, ~, DATASET_NUM] = load_ONN_data(FOLDER);
models = get_model_names(Models);

for model_idx = 1:length(Models)
    for nonlin_idx = 1:length(Nonlin)
        figure('Renderer', 'painters', 'Position', [400 400 1900 1400])
        
        Model_acc = load([FOLDER, sprintf('acc_%s_loss=%.3f_uncert=%.3f_%dFeat_%s_set%d.txt', ...
            Models{model_idx}, loss_dB(1), phase_uncert(1),  N, Nonlin{1}, DATASET_NUM)]);
        
        plot(phase_uncert, Model_acc, 'linewidth',2)
        
        legend_ = create_legend_single_model(Models{model_idx}, Nonlin{nonlin_idx}, loss_dB);
        legend(legend_, 'fontsize', fontsz, 'interpreter','latex');
        
        ylim([0, 100])
        
        a = get(gca,'XTickLabel');
        set(gca,'XTickLabel',a,'FontName','Times','fontsize',fontsz/1.2)
        
        a = get(gca,'YTickLabel');
        set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz/1.2)
        
        xlabel('Phase Uncertainty $(\sigma)$', 'fontsize', fontsz, 'interpreter','latex')
        ylabel('Accuracy (\%)', 'fontsize', fontsz, 'interpreter','latex')
        title(sprintf('Accuracy of model with %s topology',strrep(models{model_idx}, '_','\_'),...
            strrep(Nonlin{nonlin_idx}, '_','\_')), 'fontsize', fontsz, 'interpreter','latex')

        
        savefig([FOLDER, sprintf('Matlab_Figs/Model=%s_Loss=[%.3f-%.2f].fig',Models{model_idx}, ...
            min(loss_dB), max(loss_dB))])
        saveas(gcf, [FOLDER, sprintf('Matlab_Pngs/Model=%s_Loss=[%.3f-%.2f].png',Models{model_idx}, ...
            min(loss_dB), max(loss_dB))])
    end
end
