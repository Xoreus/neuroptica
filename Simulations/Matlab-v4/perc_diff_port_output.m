% Plot % difference in port pwer output for meshes of different sizes
%
% Author: Simon Geoffroy-Gagnon
% Edit: 2020.03.20
% clc; clear; close all;
close all; clc; clear;

printMe = true;

linewid = 3;
fontsz = 64;
newPaper = false;
sigma = 0.5;

for lossy = [0]

    
    Ns = [4,6,8,10,12,16,32,64];
    Ns_D = [];
    for N = Ns
        Ns_D(end+1) = N;
    end
    perc_diff = [];
    for jj = 1:length(Ns)
        avg_outPow_per_port = zeros(4,Ns(jj));
        outPow_per_port_nl = zeros(4,Ns(jj));
        in_pwer = zeros(1,Ns(jj));
        
        F = sprintf('/home/edwar/Documents/Github_Projects/neuroptica/tests/Analysis/%.2f_sigma%.2f_outPorts_mean_pow', lossy, sigma);
        
        folders = dir([F sprintf('/N=%d_*', Ns(jj))]);
        foldNames = {folders.name};
        for n = 1:length(foldNames)
            dirFiles = dir([F '/' foldNames{n} '/Topologies/']);
            dirNames = {dirFiles(3:end).name};
            if length(dirNames) == 6
                dirNames = dirNames([1,2,5,6]);
            end
            for name = 1:length(dirNames)
                sim = load([F '/' foldNames{n} '/Topologies/' dirNames{name}]);
                out_pwer = sim.(dirNames{name}(1:end-4)).out_pwer;
                avg_outPow_per_port(name, :) = avg_outPow_per_port(name, :) + mean(out_pwer);
            end
            
            if Ns(jj) == 3
                mpwer = mean(sim.(dirNames{name}(1:end-4)).in_pwer);
                in_pwer = in_pwer + mpwer(2:end);
            else
                in_pwer = in_pwer + mean(sim.(dirNames{name}(1:end-4)).in_pwer);
            end
        end
        
        lgd = {};
        
        for name = 1:length(dirNames)
            sim = load([F '/' foldNames{n} '/Topologies/' dirNames{name}]);
            lgd{end+1} = sim.(dirNames{name}(1:end-4)).topology;
        end
        
        inPow_per_port = in_pwer/length(foldNames);
        avg_outPow_per_port = avg_outPow_per_port/length(foldNames);
        perc_diff(jj, :) = abs(max(avg_outPow_per_port') - 2*mean(avg_outPow_per_port') + min(avg_outPow_per_port'))./mean(avg_outPow_per_port')*100;
        
%         perc_diff(jj, :) = (max(avg_outPow_per_port') - min(avg_outPow_per_port'))./mean(avg_outPow_per_port')*100;

        
        perc_diff_in(jj) = (max(inPow_per_port') - min(inPow_per_port'))./mean(inPow_per_port');

        perc_diff_wrt_inPow(jj, :) = perc_diff(jj, :);
    end
    
    
    figure('Renderer', 'painters', 'Position', [400 400 1900 1400])
%     set(gca, 'YScale','log')
    if newPaper && 1
        plot(Ns(1:end), perc_diff_wrt_inPow(1:end, end), 'r-v', 'markersize', 20, 'MarkerFaceColor', '#c3c3c3','linewidth', 3)
        hold on
        plot(Ns_D(1:end), perc_diff_wrt_inPow(1:end, 1), 'b-d', 'markersize', 20, 'MarkerFaceColor', 'k', 'linewidth', 3)
        %         plot(Ns_D(1:end), perc_diff_in(1:end, 1), 'ko', 'markersize', 20, 'MarkerFaceColor', 'k', 'linewidth', 3)
        
        l = legend(lgd([end,1]), 'fontsize',  fontsz*0.8, 'interpreter','latex', 'location', 'northwest');
        a = get(gca,'YTickLabel');
        set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz)
        a = get(gca,'XTickLabel');
        set(gca,'XTickLabel',a,'FontName','Times','fontsize',fontsz)
        
        a = get(gca,'YTickLabel');
        set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz*0.7)
        xlabel('Structure Size ($N$)', 'fontsize', fontsz*0.8, 'interpreter','latex', 'color', 'k')
        ylabel('Output Ports Power Deviation (\%)', 'fontsize', fontsz*0.8, 'interpreter','latex')
        grid('on')
    else
        lgd = [lgd(4), lgd(3), lgd(2), lgd(1)];
        plot_idx = [4,3,2,1];
        markers = {'-v','-o','-s','-d'};
        
        for tt = 1:length(lgd)
            plot(Ns_D(1:end), perc_diff(1:end, plot_idx(tt)), markers{tt}, 'markersize', 20, 'MarkerFaceColor', '#c3c3c3', ...
                'linewidth', 3)
            hold on
        end
        plot(Ns, perc_diff_in, 'linewidth', linewid)
        lgd{end+1} = 'Input Power';
        lgd{2} = 'Reck + Inv. Reck';
        l = legend(lgd, 'fontsize',  fontsz*0.8, 'interpreter','latex', 'location', 'northwest');
        a = get(gca,'YTickLabel');
        set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz)
        a = get(gca,'XTickLabel');
        set(gca,'XTickLabel',a,'FontName','Times','fontsize',fontsz)
        
        a = get(gca,'YTickLabel');
        set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz*0.7)
        xlabel('Structure Size ($N$)', 'fontsize', fontsz*0.8, 'interpreter','latex', 'color', 'k')
        ylabel('Port Power Error', 'fontsize', fontsz*0.8, 'interpreter','latex')
    end
    
    h = gca;
    set(h, 'YTickLabelMode','auto')
    set(h, 'XTickLabelMode','auto')
    xticks([4, 8, 12, 16, 32, 64])
    
    %     ylim([0, 15])
    
    %     axis squasre
    if printMe
        pMe_lineplot(sprintf('../Crop_Me/perc_diff_loss_%.2fdB.pdf', lossy))
    end
end