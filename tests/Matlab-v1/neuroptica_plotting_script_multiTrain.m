% Script to run both plotting function, saving all figures and pngs in
% their respecable folders ([FOLDER + '/Matlab_Figs/'] and [FOLDER + '/Matlab_Pngs/']))
%
% Author: Simon Geoffroy-Gagnon
% Edit: 25.01.2020


clc; close all; clear;
fontsz = 28;
Folder = '/home/simon/Documents/neuroptica/tests/Analysis/multiLossAnalysis/lossDiff=0_GaussAlways_rng4_avgFig/';

losses_dB_train = linspace(0, .05, 3);
losses_dB_test = linspace(0, 3, 31);
phase_uncerts_train = linspace(0, .05, 3)  ;
phase_uncerts_test = linspace(0, 1.5, 21);


loss_dB = [0.000, 0.025, 0.050];
phase_uncert = [0.000, 0.025, 0.050];
models = {'R_P', 'R_I_P', 'R_D_I_P', 'R_D_P', 'C_Q_P'};
avg_acc = cell(length(models), 1);
for ii = 1:length(models)
    avg_acc{ii} = zeros(length(phase_uncerts_test), length(losses_dB_test));
end
zeros(length(models), length(phase_uncerts_test),  length(losses_dB_test));

for ii = 1:length(loss_dB)
    for jj = 1:length(phase_uncert)
        for kk = 1:length(models)
            ActualFile = sprintf('acc_%s_loss=%.3f_uncert=%.3f_4Feat.txt', models{kk}, loss_dB(ii), phase_uncert(jj));
            FILE = [Folder ActualFile];
            a = load(FILE);
            avg_acc{kk} = avg_acc{kk} + a;
        end
    end
end

for ii = 1:length(models)
    avg_acc{ii} = avg_acc{ii}/(length(phase_uncert)*length(loss_dB));
    figure('Renderer', 'painters', 'Position', [400 400 1900 1400])
    imagesc(losses_dB_test, phase_uncerts_test, avg_acc{ii})
    
    set(gca,'YDir','normal')
    c = colorbar;
    c.Label.String = 'Accuracy (%)';
    caxis([20 80])
    
    a = get(gca,'XTickLabel');
    set(gca,'XTickLabel',a,'FontName','Times','fontsize',fontsz/1.2)
    
    a = get(gca,'YTickLabel');
    set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz/1.2)
    
    ylabel('Phase Uncertainty $(\sigma)$', 'fontsize', fontsz, 'interpreter','latex')
    xlabel('Loss (dB/MZI)', 'fontsize', fontsz, 'interpreter','latex')
    
    title(sprintf(['Average Accuracy of model %s for a training loss range of 0 dB to 0.05'...
        ' dB\n phase uncert range 0 to 0.05 Rad std dev'], strrep(models{ii}, '_','\_')))
    
end


% SimulationSettings = load_ONN_data(FOLDER);
% makeMatlabDirs(FOLDER)

% plotAcc_singleModel_AllLoss_Multi_A(FOLDER, SimulationSettings)
% plotAcc_singleModel_AllLoss_Multi(FOLDER, SimulationSettings)


% plotAcc_allModels_SingleLoss_Multi(FOLDER, SimulationSettings)
% close all;



% plotAcc_allModels_SingleLoss(FOLDER, SimulationSettings)