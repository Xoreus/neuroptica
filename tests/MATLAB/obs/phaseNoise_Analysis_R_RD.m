clear; clc; close all;
Round_Flag = false;

% Matlab plotter for Acc vs Phase inacc and loss, based on Nonlinearities
FOLDER = '../nonlinearity_analysis/';
N = 4;

% Get nonlin
Nonlin = textread([FOLDER, 'Nonlinearities.txt'], '%s', 'delimiter', '\n');

DATASET_NUM = 19;

for ii = 0:DATASET_NUM
    phase_uncert = load([FOLDER, sprintf('PhaseUncert4Features%d.txt', ii)]);
    loss_dB = load([FOLDER, sprintf('LossdB_4Features%d.txt', ii)]);
    Reck = load([FOLDER, sprintf('accuracy_Reck4Features%d.txt', ii)]);
    
    Reck_DMM = load([FOLDER, sprintf('accuracy_Reck+DMM4Features%d.txt', ii)]);
    
    iterations = 200; % How many times we test the same structure
    % create legend elements
    
    legend_ = cell(length(loss_dB),1);
    
    for jj = 1:length(loss_dB)
        legend_{round(jj)} = sprintf('R, Loss = %.2f dB', loss_dB(jj));
    end
    for jj = 1:length(loss_dB)
        legend_{end+1} = sprintf('RD, Loss = %.2f dB', loss_dB(jj));
    end
    
    % Plot Reck Acc
    figure
    plot(phase_uncert, Reck(:, 1:end), 'linewidth',2)
    xlabel('Phase Uncertainty (\sigma)')
    ylabel('Accuracy (%)')
    title(sprintf('Accuracy VS Phase Uncertainty\n R and RD \n %d iterations per Phase Uncertainty\n Dataset #%d', iterations, ii))
    hold on
    % Plot Reck+DMM Acc
    plot(phase_uncert, Reck_DMM(:, 1:end), 'linewidth',2)
    legend(legend_);
end

