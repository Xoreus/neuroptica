% Function to take in data from a Neuroptica simulation (created in python
% with either ONN_Topologies_Analysis_Retrained.py or
% ONN_Topologies_Analysis.py
% Plots the accuracy for a single models and all losses with varying phase
% uncertainties
%
% Author: Simon Geoffroy-Gagnon
% Edit: 15.01.2020


function plotAcc_singleModel_AllLoss(FOLDER)

if ~exist([FOLDER, 'Matlab_Figures'], 'dir')
    mkdir([FOLDER, 'Matlab_Figures/'])
end

[N, Models, Nonlin, phase_uncert, loss_dB, ~, DATASET_NUM] = load_ONN_data(FOLDER);

for model_idx = 1:length(Models)
    if ~contains(Models{model_idx}, 'N')
        figure
        Model_acc = load([FOLDER, sprintf('acc_%s_loss=%.3f_uncert=%.3f_%dFeat_%s_set%d.txt', ...
            Models{model_idx}, loss_dB(1)*0, phase_uncert(1)*0, N, Nonlin{1}, DATASET_NUM)]);
        
        plot(phase_uncert, Model_acc, 'linewidth',2)
        
        legend_ = create_legend_single_model(loss_dB);
        legend(legend_);
        ylim([0, 100])
        xlabel('Phase Uncertainty (\sigma)')
        ylabel('Accuracy (%)')
        title(sprintf('Accuracy of model with %s',strrep(Models{model_idx}, '_','\_')))
        savefig([FOLDER, sprintf('Matlab_Figures/Model=%s_Loss=[%.3f-%.2f].fig',Models{model_idx}, min(loss_dB), max(loss_dB))])
        saveas(gcf, [FOLDER, sprintf('Matlab_Figures/Model=%s_Loss=[%.3f-%.2f].png',Models{model_idx}, min(loss_dB), max(loss_dB))])
    else
        for nonlin_idx = 1:length(Nonlin)
            figure
        Model_acc = load([FOLDER, sprintf('acc_%s_loss=%.3f_uncert=%.3f_%dFeat_%s_set%d.txt', ...
            Models{model_idx}, loss_dB(1), phase_uncert(1), N, Nonlin{1}, DATASET_NUM)]);
            
            plot(phase_uncert, Model_acc, 'linewidth',2)
            
            legend_ = create_legend_single_model(Models{model_idx}, Nonlin{nonlin_idx}, loss_dB);
            
            title(sprintf('Accuracy of model with %s\n Nonlinearity: %s',strrep(Models{model_idx}, '_','\_'), strrep(Nonlin{nonlin_idx}, '_','\_')))
            legend(legend_);
            ylim([0, 100])
            xlabel('Phase Uncertainty (\sigma)')
            ylabel('Accuracy (%)')
            
            savefig([FOLDER, sprintf('Matlab_Figures/Model=%s_NonLinearity=%s_Loss=[%.3f-%.2f].fig',Models{model_idx}, ...
                Nonlin{nonlin_idx}, min(loss_dB), max(loss_dB))])
            saveas(gcf, [FOLDER, sprintf('Matlab_Figures/Model=%s_NonLinearity=%s_Loss=[%.3f-%.2f].png',Models{model_idx}, ...
                Nonlin{nonlin_idx}, min(loss_dB), max(loss_dB))])
        end
    end
end
