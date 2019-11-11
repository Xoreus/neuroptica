clear; clc; close all;

% Matlab plotter for Acc vs Phase inacc and loss
FOLDER = 'Accuracy_vs_phaseUncert&loss_400iter_100%acc2_butOnlyforRiR/';
FOLDER = 'Accuracy_vs_phaseUncert&loss_500iter_100%acc/';
% FOLDER = 'Accuracy_vs_phaseUncert_loss_500iter_100%acc/';

phase_uncert = load([FOLDER, 'PhaseUncert.txt']);
loss_dB = load([FOLDER, 'LossdB.txt']);

Reck = load([FOLDER, 'accuracy_Reck.txt']);
invReck = load([FOLDER, 'accuracy_invReck.txt']);
Reck_invReck = load([FOLDER, 'accuracy_Reck+invReck.txt']);
Reck_DMM_invReck = load([FOLDER, 'accuracy_Reck+DMM+invReck.txt']);
Reck_DMM = load([FOLDER, 'accuracy_Reck+DMM.txt']);

% create legend elements
legend_ = cell(length(loss_dB),1);
legend_ = cell(round(length(loss_dB)/4),1);
for ii = 1:4:length(loss_dB)
    legend_{round(ii/4)+1} = sprintf('Loss = %.2f dB', loss_dB(ii));
end

if 1
    % Plot Reck Acc
    plot(phase_uncert, Reck(:, 1:4:end), 'linewidth',2)
    legend(legend_);
    xlabel('Phase Uncertainty (\sigma)')
    ylabel('Accuracy (%)')
    title(sprintf('Accuracy VS Phase Uncertainty\n Reck layer\n 500 iterations per Phase Uncertainty'))
    
    % Plot invReck Acc
    figure
    plot(phase_uncert, invReck(:, 1:4:end), 'linewidth',2)
    legend(legend_);
    xlabel('Phase Uncertainty (\sigma)')
    ylabel('Accuracy (%)')
    title(sprintf('Accuracy VS Phase Uncertainty\n Inverted Reck layer\n 500 iterations per Phase Uncertainty'))
    
    % Plot Reck+invReck Acc
    figure
    plot(phase_uncert, Reck_invReck(:, 1:4:end), 'linewidth',2)
    legend(legend_);
    xlabel('Phase Uncertainty (\sigma)')
    ylabel('Accuracy (%)')
    title(sprintf('Accuracy VS Phase Uncertainty\n Reck + Inverted Reck layer\n 500 iterations per Phase Uncertainty'))
    
    % Plot Reck+DMM+invReck Acc
    figure
    plot(phase_uncert, Reck_DMM_invReck(:, 1:4:end), 'linewidth',2)
    legend(legend_);
    xlabel('Phase Uncertainty (\sigma)')
    ylabel('Accuracy (%)')
    title(sprintf('Accuracy VS Phase Uncertainty\n Reck + DMM + Inverted Reck layer\n 500 iterations per Phase Uncertainty'))
    
    % Plot Reck+DMM Acc
    figure
    plot(phase_uncert, Reck_DMM(:, 1:4:end), 'linewidth',2)
    legend(legend_);
    xlabel('Phase Uncertainty (\sigma)')
    ylabel('Accuracy (%)')
    title(sprintf('Accuracy VS Phase Uncertainty\n Reck + DMM layer\n 500 iterations per Phase Uncertainty'))
end

% % plot no loss but phase uncertainty for all ONN Mesh types
% figure
% plot(phase_uncert, Reck(:,1),'linewidth',2)
% hold on
% plot(phase_uncert, invReck(:,1),'linewidth',2)
% plot(phase_uncert, Reck_DMM(:,1),'linewidth',2)
% plot(phase_uncert, Reck_invReck(:,1),'linewidth',2)
% plot(phase_uncert, Reck_DMM_invReck(:,1),'linewidth',2)
% legend({'Reck','Inverted Reck','Reck + DMM', 'Reck + Inverted Reck','Reck + DMM + Inverted Reck'})
% xlabel('Phase Uncertainty (\sigma)')
% ylabel('Accuracy (%)')
% title('0 Loss, increasing Phase Uncertainty')
% 
% % plot no loss but phase uncertainty for all ONN Mesh types
% figure
% plot(loss_dB, Reck(1,:),'linewidth',2)
% hold on
% plot(loss_dB, invReck(1,:),'linewidth',2)
% plot(loss_dB, Reck_DMM(1,:),'linewidth',2)
% plot(loss_dB, Reck_invReck(1,:),'linewidth',2)
% plot(loss_dB, Reck_DMM_invReck(1,:),'linewidth',2)
% legend({'Reck','Inverted Reck','Reck + DMM', 'Reck + Inverted Reck','Reck + DMM + Inverted Reck'})
% xlabel('Loss (dB)')
% ylabel('Accuracy (%)')
% title('0 Phase Uncertainty, increasing loss')