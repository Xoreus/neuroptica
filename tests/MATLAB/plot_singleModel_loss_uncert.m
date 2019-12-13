clear; clc; close all;
fullPath = strsplit(mfilename('fullpath'), filesep);
cwd = strjoin(fullPath(1:end-1), filesep);
addpath(cwd)
cd(cwd)

% Matlab plotter for Acc vs Phase inacc and loss, based on Nonlinearities
% FOLDER = '../nonlinearity_MNIST_analysis_additional_tests2/';
% FOLDER = '../nonlinearity_MNIST_analysis_additional_tests5_1000valSamples/';
% FOLDER = '../nonlinearity_MNIST_analysis_AbsoluteValuesB4NonLin/';
% FOLDER = '../loss+uncert_sensitivity/';
% FOLDER = '../loss+uncert_sensitivity_GaussianBlobs/';
% FOLDER = '../old_loss_matrix_test/';
% FOLDER = '../new_loss_matrix_test/';
% FOLDER = '../new3_loss_matrix_test/';
% FOLDER = '../Diamond_tests/';
% FOLDER = '../Diamond_tests_Comparison_RIP/';
FOLDER = '../Diamond_tests_Comparison_RP/';
cd(FOLDER)

if ~exist([FOLDER, 'Figures'], 'dir')
    mkdir([FOLDER, 'Figures'])
end

DATASET_NUM = 0;
N = 4;

% Get nonlin, Models, Phase Uncert, loss, iterations
Models = textread([FOLDER, 'ONN_Setups.txt'], '%s', 'delimiter', '\n');
models = 'AllModels';

Nonlin = textread([FOLDER, 'Nonlinearities.txt'], '%s', 'delimiter', '\n');
phase_uncert = load([FOLDER, sprintf('PhaseUncert%dFeatures.txt',N)]);
loss_dB = load([FOLDER, sprintf('LossdB_%dFeatures.txt',N)]);
iterations = load([FOLDER, 'ITERATIONS.txt']); % How many times we test the same structure


for model_idx = 1:length(Models)
    if ~contains(Models{model_idx}, 'N')
        figure
        Model_acc = load([FOLDER, sprintf('accuracy_%s_%dFeatures_#%d_%s.txt', Models{model_idx}, N, DATASET_NUM, Nonlin{1})]);
        
        plot(phase_uncert, Model_acc, 'linewidth',2)
        
        legend_ = create_legend_single_model(Models{model_idx}, Nonlin{1}, loss_dB);
        legend(legend_);
        ylim([0, 100])
        xlabel('Phase Uncertainty (\sigma)')
        ylabel('Accuracy (%)')
        title(sprintf('Accuracy of model with %s',strrep(Models{model_idx}, '_','\_')))
        savefig([FOLDER, sprintf('Figures/Model=%s_Loss=[%.3f-%.2f].fig',Models{model_idx}, min(loss_dB), max(loss_dB))])
        saveas(gcf, [FOLDER, sprintf('Figures/Model=%s_Loss=[%.3f-%.2f].png',Models{model_idx}, min(loss_dB), max(loss_dB))])
        
    else
        for nonlin_idx = 1:length(Nonlin)
            figure
            Model_acc = load([FOLDER, sprintf('accuracy_%s_%dFeatures_#%d_%s.txt', Models{model_idx}, N, DATASET_NUM, Nonlin{nonlin_idx})]);
            plot(phase_uncert, Model_acc, 'linewidth',2)
            
            legend_ = create_legend_single_model(Models{model_idx}, Nonlin{nonlin_idx}, loss_dB);
            
            title(sprintf('Accuracy of model with %s\n Nonlinearity: %s',strrep(Models{model_idx}, '_','\_'), strrep(Nonlin{nonlin_idx}, '_','\_')))
            legend(legend_);
            ylim([0, 100])
            xlabel('Phase Uncertainty (\sigma)')
            ylabel('Accuracy (%)')
            
            savefig([FOLDER, sprintf('Figures/Model=%s_NonLinearity=%s_Loss=[%.3f-%.2f].fig',Models{model_idx}, ...
                Nonlin{nonlin_idx}, min(loss_dB), max(loss_dB))])
            saveas(gcf, [FOLDER, sprintf('Figures/Model=%s_NonLinearity=%s_Loss=[%.3f-%.2f].png',Models{model_idx}, ...
                Nonlin{nonlin_idx}, min(loss_dB), max(loss_dB))])
        end
    end
    
    
    %     closes all
end






function legend_ = create_legend_single_model(model, nonlin, loss_dB)
legend_ = cell(length(loss_dB),1);

for l_idx = 1:length(loss_dB)
    legend_{l_idx} = sprintf('L/MZI = %.2f dB', loss_dB(l_idx));
end

end




