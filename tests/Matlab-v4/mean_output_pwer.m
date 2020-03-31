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

for lossy = [0]
    Ns = [4,6,8,10,12,16,32,64];
    Ns_D = [];
    for N = Ns
        Ns_D(end+1) = N;
    end
    perc_diff = [];
    for jj = 1:length(Ns)
        outPow_per_port = zeros(4,Ns(jj));
        outPow_per_port_nl = zeros(4,Ns(jj));
        in_pwer = zeros(1,Ns(jj));
        if Ns(jj) == 32
            datasets = [0,1];
        elseif Ns(jj) == 16
            datasets = [0, 1, 10:16];
        elseif Ns(jj) == 6
            datasets = 0:100;
        elseif Ns(jj) == 4
            datasets = 0:58;
        elseif Ns(jj) <= 8
            datasets = 0:19;
        elseif Ns(jj) == 64
            datasets = 0;
        elseif Ns(jj) == 3
            datasets = 0:99;
        else
            datasets = 0:9;
        end
        
        for ii = datasets
            if lossy == 1
                F = sprintf('/home/simon/Documents/neuroptica/tests/Analysis/outPorts_mean_pow_Lossy/N=%d_%d', Ns(jj), ii);
            elseif lossy == 0.5 && 1
                F = sprintf('/home/simon/Documents/neuroptica/tests/Analysis/outPorts_mean_pow_Lossy_half_dB_loss/N=%d_%d', Ns(jj), ii);
            else
                F = sprintf('/home/simon/Documents/neuroptica/tests/Analysis/outPorts_mean_pow/N=%d_%d', Ns(jj), ii);
            end
            F_nl = sprintf('/home/simon/Documents/neuroptica/tests/Analysis/outPorts_mean_pow/N=%d_%d', Ns(jj), ii);
            
            dirFiles = dir([F '/Topologies/']);
            dirNames = {dirFiles(3:end).name};
            if length(dirNames) == 6
                dirNames = dirNames([1,2,5,6]);
            end
            for name = 1:length(dirNames)
                sim = load([F '/Topologies/' dirNames{name}]);
                out_pwer = sim.(dirNames{name}(1:end-4)).out_pwer;
                outPow_per_port(name, :) = outPow_per_port(name, :) + mean(out_pwer);
            end
            
            dirFiles = dir([F_nl '/Topologies/']);
            dirNames = {dirFiles(3:end).name};
            if length(dirNames) == 6
                dirNames = dirNames([1,2,5,6]);
            end
            for name = 1:length(dirNames)
                sim = load([F_nl '/Topologies/' dirNames{name}]);
                out_pwer = sim.(dirNames{name}(1:end-4)).out_pwer;
                outPow_per_port_nl(name, :) = outPow_per_port_nl(name, :) + mean(out_pwer);
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
            sim = load([F '/Topologies/' dirNames{name}]);
            lgd{end+1} = sim.(dirNames{name}(1:end-4)).topology;
        end
        
        inPow_per_port = in_pwer/length(datasets);
        outPow_per_port = outPow_per_port/length(datasets);
        outPow_per_port_nl = outPow_per_port_nl/length(datasets);
        
        perc_diff(jj, :) = mean(outPow_per_port');
        perc_diff_in(jj) = mean(inPow_per_port);
        perc_diff_wrt_inPow(jj, :) = perc_diff(jj, :);
    end
    
    
    figure('Renderer', 'painters', 'Position', [400 400 1900 1400])
    
    if newPaper && 1
        plot(Ns(1:end), perc_diff_wrt_inPow(1:end, end), 'r-v', 'markersize', 20, 'MarkerFaceColor', '#c3c3c3','linewidth', 3)
        hold on
        plot(Ns_D(1:end), perc_diff_wrt_inPow(1:end, 1), 'b-d', 'markersize', 20, 'MarkerFaceColor', 'k', 'linewidth', 3)
        l = legend(lgd([end,1]), 'fontsize',  fontsz*0.8, 'interpreter','latex', 'location', 'northwest');
        a = get(gca,'YTickLabel');
        set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz)
        a = get(gca,'XTickLabel');
        set(gca,'XTickLabel',a,'FontName','Times','fontsize',fontsz)
        
        a = get(gca,'YTickLabel');
        set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz*0.7)
        xlabel('Structure Size ($N$)', 'fontsize', fontsz*0.8, 'interpreter','latex', 'color', 'k')
        ylabel('Output Ports Power Error', 'fontsize', fontsz*0.8, 'interpreter','latex')
        grid('on')
    else
        plot(Ns, perc_diff(:, 1), 'linewidth', linewid)
        hold on
        plot(Ns, perc_diff(:, 2), 'linewidth', linewid*4)
        plot(Ns, perc_diff(:, 3), 'linewidth', linewid*3)
        plot(Ns, perc_diff(:, 4), 'linewidth', linewid)
        plot(Ns, perc_diff_in,  'ko', 'markersize', 20, 'MarkerFaceColor', '#c3c3c3', 'linewidth', 1)
        lgd(end+1) = {'Input Power'};
        
        l = legend(lgd, 'fontsize',  fontsz*0.8, 'interpreter','latex', 'location', 'southwest');
        l = legend(lgd, 'fontsize',  fontsz*0.8, 'interpreter','latex', 'location', 'east');
        a = get(gca,'YTickLabel');
        set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz)
        a = get(gca,'XTickLabel');
        set(gca,'XTickLabel',a,'FontName','Times','fontsize',fontsz)
        
        a = get(gca,'YTickLabel');
        set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz*0.7)
        xlabel('Structure Size ($N$)', 'fontsize', fontsz*0.8, 'interpreter','latex', 'color', 'k')
        ylabel('Mean Total Output Power (W)', 'fontsize', fontsz*0.8, 'interpreter','latex')
    end
    
    h = gca;
    set(h, 'YTickLabelMode','auto')
    set(h, 'XTickLabelMode','auto')
    xticks([4, 8, 12, 16, 32, 64])
    xlim([0, 70])
    set(gca, 'YScale', 'log')
    
    if printMe
        if lossy == 1
            pMe_lineplot(sprintf('../Crop_Me/meanPwer_1dB.pdf'))
        elseif lossy == 0.5
            pMe_lineplot(sprintf('../Crop_Me/meanPwer_05dB.pdf'))
        else
            pMe_lineplot(sprintf('../Crop_Me/meanPwer_0dB.pdf'))
        end
    end
end