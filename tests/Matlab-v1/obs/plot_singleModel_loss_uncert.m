clear; clc; close all;

fullPath = strsplit(mfilename('fullpath'), filesep);
cwd = strjoin(fullPath(1:end-1), filesep);
addpath(cwd)
cd(cwd)

% Matlab plotter for Acc vs Phase inacc and loss, based on Nonlinearities
% FOLDER = '../training_at_loss=0dB_iris_N=4_forFigures/';
% FOLDER = '../training_at_every_loss_iris_N=4_forFigures/';
cd(FOLDER)

if ~exist([FOLDER, 'Figures'], 'dir')
    mkdir([FOLDER, 'Figures'])
end

load_ONN_data

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


