clear; clc; close all;
fullPath = strsplit(mfilename('fullpath'), filesep);
cwd = strjoin(fullPath(1:end-1), filesep);
addpath(cwd)
cd(cwd)

% Matlab plotter for Acc vs Phase inacc and loss, based on Nonlinearities
% FOLDER = '../nonlinearity_MNIST_analysis_additional_tests2/';
% FOLDER = '../nonlinearity_MNIST_analysis_additional_tests5_1000valSamples/';
% FOLDER = '../nonlinearity_MNIST_analysis_AbsoluteValuesB4NonLin/';
FOLDER = '../loss+uncert_sensitivity/';
% FOLDER = '../loss+uncert_sensitivity_GaussianBlobs/';
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

loss_dB = loss_dB(1)

for model_idx = 1:length(Models)
    if contains(Models{model_idx}, 'N')
        for nonlin_idx = 1:length(Nonlin)
            figure
            Model_acc = load([FOLDER, sprintf('accuracy_%s_%dFeatures_#%d_%s.txt', Models{model_idx}, N, DATASET_NUM, Nonlin{nonlin_idx})]);
            plot(phase_uncert, Model_acc, 'linewidth',2)
            
            legend_ = create_legend_single_model(Nonlin);
            
            title(sprintf('Accuracy of model %s, Loss: %.2f dB',strrep(Models{model_idx}, '_','\_'), loss_dB))
            legend(legend_);
            ylim([0, 100])
            xlabel('Phase Uncertainty (\sigma)')
            ylabel('Accuracy (%)')
            
            savefig([FOLDER, sprintf('Figures/Model=%s_Loss=%.2f_NONLINEARITIES.fig',Models{model_idx}, loss_dB)])
            saveas(gcf, [FOLDER, sprintf('Figures/Model=%s_Loss=%.2f_NONLINEARITIES.png',Models{model_idx}, loss_dB)])
        end
    end
end


close all


function legend_ = create_legend_single_model(Nonlin)
legend_ = cell(length(Nonlin),1);

for nl_idx = 1:length(Nonlin)
    legend_{nl_idx} = sprintf('%s', strrep(Nonlin{nl_idx}, '_','\_'));
end

end




