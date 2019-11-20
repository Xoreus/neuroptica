clear; clc; close all;

% Matlab plotter for Acc vs Phase inacc and loss, based on Nonlinearities
FOLDER = '../nonlinearity_analysis/';
FOLDER = '../nonlinearity_MNIST_analysis/';
FOLDER = '../nonlinearity_MNIST_analysis_additional_tests2/';
if ~exist([FOLDER, 'Figures'], 'dir')
       mkdir([FOLDER, 'Figures'])
end
    
DATASET_NUM = 0;
N = 4;

% Get nonlin, Models, Phase Uncert, loss, iterations
Models = textread([FOLDER, 'ONN_Setups.txt'], '%s', 'delimiter', '\n');
% Models = Models([3,6,11,7]);

Nonlin = textread([FOLDER, 'Nonlinearities.txt'], '%s', 'delimiter', '\n');
phase_uncert = load([FOLDER, sprintf('PhaseUncert%dFeatures.txt',N)]);
loss_dB = load([FOLDER, sprintf('LossdB_%dFeatures.txt',N)]);
iterations = load([FOLDER, 'ITERATIONS.txt']); % How many times we test the same structure

noNonLin_Models = {Models{~contains(Models, 'N')}};
nonLin_Models = {Models{contains(Models, 'N')}};

% Get only interesting Models
% Models = Models([3, 1]);
% Nonlin = Nonlin(4);
loss_dB = loss_dB(1);

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
                xlabel('Phase Uncertainty (\sigma)')
                ylabel('Accuracy (%)')
                title(sprintf('Accuracy of model with %s Nonlinearity (if used)\n %s\n %d iterations per Phase Uncertainty\n Dataset #%d',...
                    Nonlin{kk}, strrep(noNonLin_Models{jj}, '_','\_'), iterations, ii))
                legend(legend_);
            end
            
            for jj = 1:length(nonLin_Models)
                Model_acc = load([FOLDER, sprintf('accuracy_%s_%dFeatures_#%d_%s.txt', nonLin_Models{jj}, N, ii, Nonlin{kk})]);
                plot(phase_uncert, Model_acc(:, 1), 'linewidth',2)
                xlabel('Phase Uncertainty (\sigma)')
                ylabel('Accuracy (%)')
                title(sprintf('Accuracy of model with %s Nonlinearity (if used)\n %s\n %d iterations per Phase Uncertainty\n Dataset #%d',...
                    Nonlin{kk}, strrep(nonLin_Models{jj}, '_','\_'), iterations, ii))
                legend(legend_);
            end
            savefig(sprintf('Figures/NonLinearity:%s_Loss:%.3f_InterestingModels_PCA_MNIST_[1,3,6,7].fig',Nonlin{kk},loss_dB))
        end
    end
end
