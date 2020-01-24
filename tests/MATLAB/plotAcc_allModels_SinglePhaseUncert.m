% Function to take in data from a Neuroptica simulation (created in python
% with either ONN_Topologies_Analysis_Retrained.py or
% plotAcc_allModels_SinglePhaseUncert.py
% Plots the accuracy for all models and a single phase uncert with loss
%
% Author: Simon Geoffroy-Gagnon
% Edit: 15.01.2020


function plotAcc_allModels_SinglePhaseUncert(FOLDER)

if ~exist([FOLDER, 'Matlab_Figs'], 'dir')
    mkdir([FOLDER, 'Matlab_Figs'])
end
if ~exist([FOLDER, 'Matlab_Pngs'], 'dir')
    mkdir([FOLDER, 'Matlab_Pngs'])
end

[N, Models, Nonlin, phase_uncert, loss_dB, ~, DATASET_NUM] = load_ONN_data(FOLDER);

for p_idx = 1:length(phase_uncert)
    figure
    for model_idx = 1:length(Models)
        legend_ = create_legend_single_loss(Models, Nonlin);
        if ~contains(Models{model_idx}, 'N')
        Model_acc = load([FOLDER, sprintf('acc_%s_loss=%.3f_uncert=%.3f_%dFeat_%s_set%d.txt', ...
            Models{model_idx}, loss_dB(1), phase_uncert(1), N, Nonlin{1}, DATASET_NUM)]);
            plot(loss_dB, Model_acc(p_idx, :), 'linewidth', 2)
            hold on
        else
            for nonlin_idx = 1:length(Nonlin)
        Model_acc = load([FOLDER, sprintf('acc_%s_loss=%.3f_uncert=%.3f_%dFeat_%s_set%d.txt', ...
            Models{model_idx}, loss_dB(1), phase_uncert(1),  phase_uncert(end), N, Nonlin{1}, DATASET_NUM)]);
                 
                plot(phase_uncert, Model_acc, 'linewidth',2)
                hold on
            end
        end
    end
    legend(legend_);
    ylim([0, 100])
    xlabel('Loss (dB/MZI, Approximate)')
    ylabel('Accuracy (%)')
    title(sprintf('Accuracy of models with Phase Uncert = %.3f Rad', phase_uncert(p_idx)))
    savefig([FOLDER, sprintf('Matlab_Figs/AllModels_PhaseUncert=%.3f.fig', phase_uncert(p_idx))])
    saveas(gcf, [FOLDER, sprintf('Matlab_Pngs/AllModels_PhaseUncert=%.3f.png', phase_uncert(p_idx))])
end

end