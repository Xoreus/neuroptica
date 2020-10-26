% Plot % difference in port pwer output for meshes of different sizes
%
% Author: Simon Geoffroy-Gagnon
% Edit: 2020.09.03

close all; clc; clear;

printMe = true;
linewid = 3;
fontsz = 64;

for lossy = 0
    Ns = [4,8,16,32,64];
    datasets_used = zeros(length(Ns),1);
    Ns_D = [];
    for N = Ns
        Ns_D(end+1) = N;
    end
    perc_diff = [];
    for jj = 1:length(Ns)
        outPow_per_port = zeros(4,Ns(jj));
        outPow_per_port_nl = zeros(4,Ns(jj));
        in_pwer = zeros(1,Ns(jj));
        
        if lossy == 1
            F = sprintf('../Analysis/outPorts_mean_pow_0.5/N=%d/N=%d_*', Ns(jj), Ns(jj));
        else
            F = sprintf('../Analysis/outPorts_mean_pow/N=%d/N=%d_*', Ns(jj),Ns(jj));
        end
        sims = dir(F);
        sim_folders = {sims.folder};
        
        for simulation = {sims.name}
            
            dirFiles = dir([sims(1).folder, '/', simulation{1}, '/Topologies/']);
            if length(dirFiles) == 6
                datasets_used(jj) = datasets_used(jj) + 1;
                dirNames = {dirFiles(3:end).name};
                for name = 1:length(dirNames)
                    sim = load([sims(1).folder, '/', simulation{1}, '/Topologies/' dirNames{name}]);
                    out_pwer = sim.(dirNames{name}(1:end-4)).out_pwer;
                    outPow_per_port(name, :) = outPow_per_port(name, :) + mean(out_pwer);
                end
                
                in_pwer = in_pwer + mean(sim.(dirNames{name}(1:end-4)).in_pwer);
            end
        end
        inPow_per_port = in_pwer/datasets_used(jj);
        outPow_per_port = outPow_per_port/(datasets_used(jj));
        outPow_per_port_nl = outPow_per_port_nl/(datasets_used(jj));
        
        perc_diff(jj, :) = mean(outPow_per_port');
        perc_diff_in(jj) = mean(inPow_per_port);
        perc_diff_wrt_inPow(jj, :) = perc_diff(jj, :);
    end
    
    lgd = {};
    
    for name = 1:length(dirNames)
        F = '../Analysis/outPorts_mean_pow/N=4/N=4_0';
        dirFiles = dir([sims(1).folder, '/', simulation{1}, '/Topologies/']);
        
        dirNames = {dirFiles(3:end).name};
        
        sim = load([F '/Topologies/' dirNames{name}]);
        lgd{end+1} = sim.(dirNames{name}(1:end-4)).topology;
    end
    lgd{3} = 'Reck/$\overline{\mathrm{Reck}}$';
    lgd(3) = []; % Delete lgd(3), reck + inv. reck, if not used
    
    figure('Renderer', 'painters', 'Position', [200 200 800 700]*2)
    
    if lossy
        plot(Ns, 10*log10(perc_diff(:, 4)), 'linewidth', linewid, 'marker', 'V', 'markersize', 20, 'MarkerFaceColor', '#c3c3c3')
        hold on
        plot(Ns, 10*log10(perc_diff(:, 2)), 'linewidth', linewid, 'marker', 'S', 'markersize', 20, 'MarkerFaceColor', '#c3c3c3')
        plot(Ns, 10*log10(perc_diff(:, 1)), 'linewidth', linewid, 'marker', 'D', 'markersize', 20, 'MarkerFaceColor', '#c3c3c3')
        %         plot(Ns, 10*log10(perc_diff(:, 3)), 'linewidth', linewid,'marker', 'O', 'markersize', 20, 'MarkerFaceColor', '#c3c3c3')
        plot(Ns, 10*log10(perc_diff_in),  'k', 'linewidth', 2)
        lgd(end+1) = {'Input Power'};
        %         l = legend(lgd, 'fontsize',  fontsz*0.8, 'interpreter','latex', 'location', 'southwest');
        l = legend({lgd{3}, lgd{2}, lgd{1}, lgd{4}}, 'fontsize',  fontsz*0.8, 'interpreter','latex', 'location', 'southwest');
        
    else
        plot(Ns, 10*log10(perc_diff(:, 4)), 'linewidth', linewid, 'marker', 'V', 'markersize', 25, 'MarkerFaceColor', '#c3c3c3')
        hold on
        plot(Ns, 10*log10(perc_diff(:, 2)), 'linewidth', linewid, 'marker', 'S', 'markersize', 15, 'MarkerFaceColor', '#c3c3c3')
        plot(Ns, 10*log10(perc_diff(:, 1)), 'linewidth', linewid, 'marker', 'D', 'markersize', 20, 'MarkerFaceColor', '#c3c3c3')
        %         plot(Ns, 10*log10(perc_diff(:, 3)), 'linewidth', linewid,'marker', 'O', 'markersize', 20, 'MarkerFaceColor', '#c3c3c3')
        plot(Ns, 10*log10(perc_diff_in),  'k', 'linewidth', 2)
        lgd(end+1) = {'Input Power'};
%         l = legend(lgd, 'fontsize',  fontsz*0.8, 'interpreter','latex', 'location', 'east');
        l = legend({lgd{3}, lgd{2}, lgd{1}, lgd{4}}, 'fontsize',  fontsz*0.8, 'interpreter','latex', 'location', 'southwest');

    end
    a = get(gca,'YTickLabel');
    set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz)
    a = get(gca,'XTickLabel');
    set(gca,'XTickLabel',a,'FontName','Times','fontsize',fontsz)
    
    a = get(gca,'YTickLabel');
    set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz*0.7)
    xlabel('Structure Size ($N$)', 'fontsize', fontsz*0.8, 'interpreter','latex', 'color', 'k')
    ylabel('Mean Output Power (a.u.)', 'fontsize', fontsz*0.8, 'interpreter','latex')
    
    h = gca;
    set(h, 'YTickLabelMode','auto')
    set(h, 'XTickLabelMode','auto')
    xticks([4, 8, 16, 32, 64])
    yticks([])
    %     xlim([0, 70])
    %     set(gca, 'YScale', 'log')
    %     if lossy
    %         yticklabels([-1, 0, 1])
    %     else
    %         labels = get(gca,'YTickLabel');    %# Get the current labels
    %         set(gca,'YLimMode','manual',...    %# Freeze the current limits
    %             'YTickMode','manual',...   %# Freeze the current tick values
    %             'YTickLabel',strcat(labels,{' dB'}));  %# Change the labels
    %     end
    
    if printMe
        if lossy
            pMe_lineplot(sprintf('../Crop_Me/meanPwer_05dB.pdf'))
        else
            pMe_lineplot(sprintf('../Crop_Me/meanPwer_0dB.pdf'))
        end
    end
end