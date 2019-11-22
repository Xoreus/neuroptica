clear; clc; close all;
fullPath = strsplit(mfilename('fullpath'), filesep);
cwd = strjoin(fullPath(1:end-1), filesep);
addpath(cwd)
cd(cwd)

% Matlab plotter for Acc vs Phase inacc and loss, based on Nonlinearities
FOLDER = '../nonlinearity_MNIST_analysis_additional_tests2/';
FOLDER = '../nonlinearity_MNIST_analysis_additional_tests5_1000valSamples/';
if ~exist([FOLDER, 'Figures'], 'dir')
       mkdir([FOLDER, 'Figures'])
end
cd(FOLDER)

DATASET_NUM = 0;
N = 4;

% Get nonlin, Models, Phase Uncert, loss, iterations
Models = textread([FOLDER, 'ONN_Setups.txt'], '%s', 'delimiter', '\n');
models = 'AllModels';

Nonlin = textread([FOLDER, 'Nonlinearities.txt'], '%s', 'delimiter', '\n');
phase_uncert = load([FOLDER, sprintf('PhaseUncert%dFeatures.txt',N)]);
loss_dB = load([FOLDER, sprintf('LossdB_%dFeatures.txt',N)]);
iterations = load([FOLDER, 'ITERATIONS.txt']); % How many times we test the same structure

% Get only interesting Models
% Models = Models([2,6,10,7]);
% models = 'InterestingModels';
% Nonlin = Nonlin(4);
loss_dB = loss_dB(1);

noNonLin_Models = {Models{~contains(Models, 'N')}};
nonLin_Models = {Models{contains(Models, 'N')}};

for l_idx = 1:length(loss_dB)
    % create legend elements
    legend_ = cell(length(loss_dB),1);
    for ijk = 1:length(noNonLin_Models)
        legend_{(ijk)} = sprintf('%s, Loss/MZI = %.2f dB',  strrep(noNonLin_Models{ijk}, '_','\_'), loss_dB(l_idx));
    end
    for ijk = 1:length(nonLin_Models)
        legend_{end+1} = sprintf('%s, Loss/MZI = %.2f dB', strrep(nonLin_Models{ijk}, '_','\_'), loss_dB(l_idx));
    end
    
    for ii = 0:DATASET_NUM
        for kk = 1:length(Nonlin)
            figure
            hold on
            for jj = 1:length(noNonLin_Models)
                Model_acc = load([FOLDER, sprintf('accuracy_%s_%dFeatures_#%d_%s.txt', noNonLin_Models{jj}, N, ii, Nonlin{1})]);
                plot(phase_uncert, Model_acc(:, 1), 'linewidth',2)
            end
            
            for jj = 1:length(nonLin_Models)
                Model_acc = load([FOLDER, sprintf('accuracy_%s_%dFeatures_#%d_%s.txt', nonLin_Models{jj}, N, ii, Nonlin{kk})]);
                plot(phase_uncert, Model_acc(:, 1), 'linewidth',2)
                xlabel('Phase Uncertainty (\sigma)')
                ylabel('Accuracy (%)')
                title(sprintf('Accuracy of model with %s Nonlinearity',strrep(Nonlin{kk}, '_','\_')))
                legend(legend_);
                ylim([0, 100])
            end
            savefig(sprintf('Figures/NonLinearity=%s_Loss=%.3f_%s_PCA_MNIST_[1,3,6,7].fig',Nonlin{kk}, loss_dB, models))
            saveas(gcf, sprintf('Figures/NonLinearity=%s_Loss=%.3f_%s_PCA_MNIST_[1,3,6,7].png',Nonlin{kk}, loss_dB, models))
        end
    end
end
