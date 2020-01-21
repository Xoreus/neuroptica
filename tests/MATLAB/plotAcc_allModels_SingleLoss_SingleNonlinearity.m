% Function to take in data from a Neuroptica simulation (created in python
% with either ONN_Topologies_Analysis_Retrained.py or
% ONN_Topologies_Analysis.py
% Plots the accuracy for all models and a single loss with varying phase
% uncertainties
%
% Author: Simon Geoffroy-Gagnon
% Edit: 15.01.2020


function plotAcc_allModels_SingleLoss_SingleNonlinearity(FOLDER)

if ~exist([FOLDER, 'Matlab_Figures'], 'dir')
    mkdir([FOLDER, 'Matlab_Figures'])
end

[N, Models, Nonlin, phase_uncert, loss_dB, ~, DATASET_NUM] = load_ONN_data(FOLDER);

noNonLin_Models = {Models{~contains(Models, 'N')}};
nonLin_Models = {Models{contains(Models, 'N')}};

for l_idx = 1:length(loss_dB)
    % create legend elements
    legend_ = cell(length(Models),1);
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
                Model_acc = load([FOLDER,  sprintf('acc_%s_loss=%.2f_uncert=%.2f_%dFeat_%s_set%d.txt', ...
                    noNonLin_Models{jj}, loss_dB(1), phase_uncert(1), N, Nonlin{1}, DATASET_NUM)]);
                
                plot(phase_uncert, Model_acc(:, 1), 'linewidth',2)
                
            end
            
            for jj = 1:length(nonLin_Models)
                Model_acc = load([FOLDER,  sprintf('acc_%s_loss=%.2f_uncert=%.2f_%dFeat_%s_set%d.txt', ...
                    nonLin_Models{jj}, loss_dB(1), phase_uncert(1), N, Nonlin{nonlin_idx}, DATASET_NUM)]);
                
                plot(phase_uncert, Model_acc(:, 1), 'linewidth',2)
            end
            xlabel('Phase Uncertainty (\sigma)')
            ylabel('Accuracy (%)')
            title(sprintf('Accuracy of all models with %s Nonlinearity at %.3fdB/MZI loss',strrep(Nonlin{kk}, '_','\_'), loss_dB(l_idx)))
            legend(legend_);
            ylim([0, 100])
            savefig([FOLDER, sprintf('Matlab_Figures/NonLinearity=%s_Loss=%.3f.fig',Nonlin{kk}, loss_dB(l_idx))])
            saveas(gcf, [FOLDER, sprintf('Matlab_Figures/NonLinearity=%s_Loss=%.3f.png',Nonlin{kk}, loss_dB(l_idx))])
        end
    end
end

end