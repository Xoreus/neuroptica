% Function to take in data from a Neuroptica simulation (created in python
% with either ONN_Topologies_Analysis_Retrained.py or
% ONN_Topologies_Analysis.py
% Plots the accuracy for all models and a single loss with varying phase
% uncertainties
%
% Author: Simon Geoffroy-Gagnon
% Edit: 23.01.2020


function plotAcc_AllModelsAllTrainings(FOLDER)
linestyles = {'--','-.',':', '-'};
linestyle_idx = 0;

if ~exist([FOLDER, 'Matlab_Figs'], 'dir')
    mkdir([FOLDER, 'Matlab_Figs'])
end
if ~exist([FOLDER, 'Matlab_Pngs'], 'dir')
    mkdir([FOLDER, 'Matlab_Pngs'])
end

[N, Models, Nonlin, phase_uncert_train, phase_uncert_test, loss_dB_train, loss_dB_test, ~, DATASET_NUM] = load_ONN_data_MultiTrain(FOLDER);

for l_idx = 1 % :length(loss_dB_test)
    legend_ = {};
    figure
    for model_idx = 1:length(Models)
        legend_ = [legend_, create_legend_trainingLoss(Models(model_idx), loss_dB_train)];
        linestyle_idx = linestyle_idx + 1;
        disp(linestyles(linestyle_idx) )
        for ii = 1:length(loss_dB_train)
            for jj = 1:length(phase_uncert_train)
                
                Model_acc = load([FOLDER, sprintf('acc_%s_loss=%.3f_uncert=%.3f_%dFeat_%s_set%d.txt', ...
                    Models{model_idx}, loss_dB_train(ii), phase_uncert_train(jj), N, Nonlin{1}, DATASET_NUM)]);
                
                plot(phase_uncert_test, Model_acc(:, l_idx), 'linewidth', 2, 'linestyle', linestyles{linestyle_idx})
                hold on
            end
        end
        legend(legend_);
        
        
        ylim([0, 100])
        xlabel('Phase Uncertainty (\sigma)')
        ylabel('Accuracy (%)')
        title(sprintf('Accuracy of models with loss = %.3f dB/MZI\nTrained at Loss=%.3f, phase uncertaity=%.3f',...
            loss_dB_test(l_idx), loss_dB_train(ii), phase_uncert_train(jj)))
        savefig([FOLDER, sprintf('Matlab_Figs/AllModels_trainedAtLoss=%.3f_PhaseUncert=%.3f_Loss=%.3f.fig',...
            loss_dB_train(ii), phase_uncert_train(jj), loss_dB_test(l_idx))])
        saveas(gcf, [FOLDER, sprintf('Matlab_Pngs/AllModels_trainedAtLoss=%.3f_PhaseUncert=%.3f_WithLoss=%.3f.png',...
            loss_dB_train(ii), phase_uncert_train(jj), loss_dB_test(l_idx))])
    end
end
end