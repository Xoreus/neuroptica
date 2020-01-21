clear; clc; close all;
Round_Flag = false;

% Matlab plotter for Acc vs Phase inacc and loss
% % FOLDER = 'Accuracy_vs_phaseUncert&loss_400iter_100%acc2_butOnlyforRiR/';
% FOLDER = 'Accuracy_vs_phaseUncert&loss_500iter_100%acc/';
% FOLDER = 'Accuracy_vs_phaseUncert_loss_500iter_100%acc/';
% FOLDER = 'Accuracy_vs_phaseUncert_loss_500iter_100%acc_8Features/';
% FOLDER = 'Accuracy_vs_phaseUncert_loss_500iter/';
% FOLDER = 'Accuracy_vs_phaseUncert_loss_500iter_Round#2/';
FOLDER = '../training_at_loss=0dB_iris_N=4_forFigures/';

N = 4;

phase_uncert = load([FOLDER, sprintf('PhaseUncert%dFeatures.txt', N)]);
loss_dB = load([FOLDER, sprintf('LossdB%dFeatures.txt', N)]);
Reck = load([FOLDER, sprintf('accuracy_Reck%dFeatures.txt', N)]);
invReck = load([FOLDER, sprintf('accuracy_invReck%dFeatures.txt', N)]);
Reck_invReck = load([FOLDER, sprintf('accuracy_Reck+invReck%dFeatures.txt', N)]);
Reck_DMM_invReck = load([FOLDER, sprintf('accuracy_Reck+DMM+invReck%dFeatures.txt', N)]);
Reck_DMM = load([FOLDER, sprintf('accuracy_Reck+DMM%dFeatures.txt', N)]);

iterations = 200; % How many times we test the same structure

% create legend elements
if Round_Flag
    legend_ = cell(round(length(loss_dB)/4),1);
    for ii = 1:4:length(loss_dB)
        legend_{round(ii/4)+1} = sprintf('Loss = %.2f dB', loss_dB(ii));
    end
else
    legend_ = cell(length(loss_dB),1);
    
    for ii = 1:length(loss_dB)
        legend_{round(ii)} = sprintf('Loss = %.2f dB', loss_dB(ii));
    end
end

if Round_Flag % ROUND LIKE A BITCHASSSSS
    % Plot Reck Acc
    plot(phase_uncert, Reck(:, 1:4:end), 'linewidth',2)
    legend(legend_);
    xlabel('Phase Uncertainty (\sigma)')
    ylabel('Accuracy (%)')
    title(sprintf('Accuracy VS Phase Uncertainty\n Reck layer\n %d iterations per Phase Uncertainty', iterations))
    
    % Plot invReck Acc
    figure
    plot(phase_uncert, invReck(:, 1:4:end), 'linewidth',2)
    legend(legend_);
    xlabel('Phase Uncertainty (\sigma)')
    ylabel('Accuracy (%)')
    title(sprintf('Accuracy VS Phase Uncertainty\n Inverted Reck layer\n %d iterations per Phase Uncertainty', iterations))
    
    % Plot Reck+invReck Acc
    figure
    plot(phase_uncert, Reck_invReck(:, 1:4:end), 'linewidth',2)
    legend(legend_);
    xlabel('Phase Uncertainty (\sigma)')
    ylabel('Accuracy (%)')
    title(sprintf('Accuracy VS Phase Uncertainty\n Reck + Inverted Reck layer\n %d iterations per Phase Uncertainty', iterations))
    
    % Plot Reck+DMM+invReck Acc
    figure
    plot(phase_uncert, Reck_DMM_invReck(:, 1:4:end), 'linewidth',2)
    legend(legend_);
    xlabel('Phase Uncertainty (\sigma)')
    ylabel('Accuracy (%)')
    title(sprintf('Accuracy VS Phase Uncertainty\n Reck + DMM + Inverted Reck layer\n %d iterations per Phase Uncertainty', iterations))
    
    % Plot Reck+DMM Acc
    figure
    plot(phase_uncert, Reck_DMM(:, 1:4:end), 'linewidth',2)
    legend(legend_);
    xlabel('Phase Uncertainty (\sigma)')
    ylabel('Accuracy (%)')
    title(sprintf('Accuracy VS Phase Uncertainty\n Reck + DMM layer\n %d iterations per Phase Uncertainty', iterations))
else % DO ALL OF THEM AHHAHAHAHAHAHAHA
    % Plot Reck Acc
    plot(phase_uncert, Reck(:, 1:end), 'linewidth',2)
    legend(legend_);
    xlabel('Phase Uncertainty (\sigma)')
    ylabel('Accuracy (%)')
    title(sprintf('Accuracy VS Phase Uncertainty\n Reck layer\n %d iterations per Phase Uncertainty', iterations))
    
    % Plot invReck Acc
    figure
    plot(phase_uncert, invReck(:, 1:end), 'linewidth',2)
    legend(legend_);
    xlabel('Phase Uncertainty (\sigma)')
    ylabel('Accuracy (%)')
    title(sprintf('Accuracy VS Phase Uncertainty\n Inverted Reck layer\n %d iterations per Phase Uncertainty', iterations))
    
    % Plot Reck+invReck Acc
    figure
    plot(phase_uncert, Reck_invReck(:, 1:end), 'linewidth',2)
    legend(legend_);
    xlabel('Phase Uncertainty (\sigma)')
    ylabel('Accuracy (%)')
    title(sprintf('Accuracy VS Phase Uncertainty\n Reck + Inverted Reck layer\n %d iterations per Phase Uncertainty', iterations))
    
    % Plot Reck+DMM+invReck Acc
    figure
    plot(phase_uncert, Reck_DMM_invReck(:, 1:end), 'linewidth',2)
    legend(legend_);
    xlabel('Phase Uncertainty (\sigma)')
    ylabel('Accuracy (%)')
    title(sprintf('Accuracy VS Phase Uncertainty\n Reck + DMM + Inverted Reck layer\n %d iterations per Phase Uncertainty', iterations))
    
    % Plot Reck+DMM Acc
    figure
    plot(phase_uncert, Reck_DMM(:, 1:end), 'linewidth',2)
    legend(legend_);
    xlabel('Phase Uncertainty (\sigma)')
    ylabel('Accuracy (%)')
    title(sprintf('Accuracy VS Phase Uncertainty\n Reck + DMM layer\n %d iterations per Phase Uncertainty', iterations))
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