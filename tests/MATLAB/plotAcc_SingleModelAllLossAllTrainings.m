% Function to take in data from a Neuroptica simulation (created in python
% with either ONN_Topologies_Analysis_Retrained.py or
% ONN_Topologies_Analysis.py
% Plots the accuracy for a single model trained at a specific loss/phase
% uncert, at all losses with varying phase uncert
%
% This will create a ridiculous amnt of data, so we'll save the fig in a
% separate dir than the pngs.
%
% Author: Simon Geoffroy-Gagnon
% Edit: 23.01.2020


function plotAcc_SingleModelAllLossAllTrainings(FOLDER)

if ~exist([FOLDER, 'Matlab_Figs'], 'dir')
    mkdir([FOLDER, 'Matlab_Figs'])
end
if ~exist([FOLDER, 'Matlab_Pngs'], 'dir')
    mkdir([FOLDER, 'Matlab_Pngs'])
end

[N, Models, Nonlin, phase_uncert_train, phase_uncert_test, loss_dB_train, loss_dB_test, ~, DATASET_NUM] = load_ONN_data_MultiTrain(FOLDER);

for l_idx = 1:length(loss_dB_train([1,end]))
    figure
    for model_idx = 1:length(Models)
        legend_ = create_legend_single_model(loss_dB_test);
        for p_idx = 1:length(phase_uncert_train([1,end]))
            
            Model_acc = load([FOLDER, sprintf('acc_%s_loss=%.3f_uncert=%.3f_%dFeat_%s_set%d.txt', ...
                Models{model_idx}, loss_dB_train(l_idx), phase_uncert_train(p_idx), N, Nonlin{1}, DATASET_NUM)]);
            
            plot(phase_uncert_test, Model_acc(:,:), 'linewidth', 2)
            legend(legend_);
            ylim([0, 100])
            xlabel('Phase Uncertainty (\sigma)')
            ylabel('Accuracy (%)')
            title(sprintf('Accuracy of model %s \ntrained at loss = %.3f dB/MZI, Phase Uncert = %.3f', ...
                strrep(Models{model_idx}, '_','\_'), loss_dB_test(l_idx), phase_uncert_train(p_idx)))
            savefig([FOLDER, sprintf('Matlab_Figs/%s_trainedAtLoss=%.3f_phaseUncert=%.3f.fig',...
                Models{model_idx}, loss_dB_train(l_idx), phase_uncert_train(p_idx))])
            saveas(gcf, [FOLDER, sprintf('Matlab_Pngs/%s_trainedAtLoss=%.3f_phaseUncert=%.3f.png',...
                Models{model_idx}, loss_dB_train(l_idx), phase_uncert_train(p_idx))])
        end
    end
end
end