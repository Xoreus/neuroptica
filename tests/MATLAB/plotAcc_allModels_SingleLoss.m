% Function to take in data from a Neuroptica simulation (created in python
% with either ONN_Topologies_Analysis_Retrained.py or
% ONN_Topologies_Analysis.py
% Plots the accuracy for all models and a single loss with varying phase
% uncertainties
%
% Author: Simon Geoffroy-Gagnon
% Edit: 15.01.2020


function plotAcc_allModels_SingleLoss(FOLDER)

if ~exist([FOLDER, 'Matlab_Figures'], 'dir')
    mkdir([FOLDER, 'Matlab_Figures'])
end

[N, Models, Nonlin, phase_uncert, loss_dB, ~, DATASET_NUM] = load_ONN_data(FOLDER);

for l_idx = 1:length(loss_dB)
    figure
    for model_idx = 1:length(Models)
        legend_ = create_legend_single_loss(Models, Nonlin);
        if ~contains(Models{model_idx}, 'N')
        Model_acc = load([FOLDER, sprintf('acc_%s_loss=%.2f_uncert=%.2f_%dFeat_%s_set%d.txt', ...
            Models{model_idx}, loss_dB(1), phase_uncert(1), N, Nonlin{1}, DATASET_NUM)]);
            
            plot(phase_uncert, Model_acc(:, l_idx), 'linewidth', 2)
            hold on
        else
            for nonlin_idx = 1:length(Nonlin)
        Model_acc = load([FOLDER, sprintf('acc_%s_loss=%.2f_uncert=%.2f_%dFeat_%s_set%d.txt', ...
            Models{model_idx}, loss_dB(1), phase_uncert(1), N, Nonlin{1}, DATASET_NUM)]);
                
                plot(phase_uncert, Model_acc, 'linewidth',2)
                hold on
            end
        end
    end
    legend(legend_);
    ylim([0, 100])
    xlabel('Phase Uncertainty (\sigma)')
    ylabel('Accuracy (%)')
    title(sprintf('Accuracy of models with loss = %.3f dB/MZI', loss_dB(l_idx)))
    savefig([FOLDER, sprintf('Matlab_Figures/AllModels_Loss=%.3f.fig', loss_dB(l_idx))])
    saveas(gcf, [FOLDER, sprintf('Matlab_Figures/AllModels_Loss=%.3f.png', loss_dB(l_idx))])
end

end