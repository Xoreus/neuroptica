clear; clc; close all;

% Matlab plotter for Acc vs Phase inacc and loss
FOLDER = 'Accuracy_vs_phaseUncert_300iter/';
phase_uncert = load([FOLDER, 'PhaseUncert.txt']);
loss_dB = load([FOLDER, 'LossdB.txt']);

Reck = load([FOLDER, 'accuracy_Reck.txt']);
invReck = load([FOLDER, 'accuracy_invReck.txt']);
Reck_invReck = load([FOLDER, 'accuracy_Reck+invReck.txt']);
Reck_DMM_invReck = load([FOLDER, 'accuracy_Reck+DMM+invReck.txt']);

% create legend elements
legend_ = cell(length(loss_dB),1);
for ii = 1:length(loss_dB)
    legend_{ii} = sprintf('Loss = %.2f dB', loss_dB(ii));
end

% Plot Reck Acc
plot(phase_uncert, Reck, 'linewidth',2)
legend(legend_);
xlabel('Phase Uncertainty (\sigma)')
ylabel('Accuracy (%)')
title(sprintf('Accuracy VS Phase Uncertainty\n Reck layer\n 10 iterations'))

% Plot invReck Acc
figure
plot(phase_uncert, invReck, 'linewidth',2)
legend(legend_);
xlabel('Phase Uncertainty (\sigma)')
ylabel('Accuracy (%)')
title(sprintf('Accuracy VS Phase Uncertainty\n Inverted Reck layer\n 10 iterations'))

% Plot Reck+invReck Acc
figure
plot(phase_uncert, Reck_invReck, 'linewidth',2)
legend(legend_);
xlabel('Phase Uncertainty (\sigma)')
ylabel('Accuracy (%)')
title(sprintf('Accuracy VS Phase Uncertainty\n Reck + Inverted Reck layer\n 10 iterations'))

% Plot Reck+DMM+invReck Acc
figure
plot(phase_uncert, Reck_DMM_invReck, 'linewidth',2)
legend(legend_);
xlabel('Phase Uncertainty (\sigma)')
ylabel('Accuracy (%)')
title(sprintf('Accuracy VS Phase Uncertainty\n Reck + DMM + Inverted Reck layer\n 10 iterations'))


