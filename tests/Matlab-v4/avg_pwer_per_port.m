% Plot avg pwer per port
%
% Author: Simon Geoffroy-Gagnon
% Edit: 2020.03.20
clc; clear; close all;

printMe = true;

linewid = 3;
fontsz = 64;
ii = 4;
lossy = true;
datasets = 0:8;
Ns = [4,6,8,10,12];
Ns = [12];
% Ns = [4];
for N = Ns
    outPow_per_port = zeros(6,N);
    in_pwer = zeros(1,N);
    
    for ii = datasets
        if lossy && 0
            F = sprintf('/home/simon/Documents/neuroptica/tests/Analysis/outPorts_mean_pow_Lossy/N=%d_%d', N, ii);
        elseif lossy && 1
            F = sprintf('/home/simon/Documents/neuroptica/tests/Analysis/outPorts_mean_pow_Lossy_half_dB_loss/N=%d_%d', N, ii);
        else
            F = sprintf('/home/simon/Documents/neuroptica/tests/Analysis/outPorts_mean_pow/N=%d_%d', N, ii);
        end
        
        dirFiles = dir([F '/Topologies/']);
        dirNames = {dirFiles(3:end).name};
        
        for name = 1:length(dirNames)
            sim = load([F '/Topologies/' dirNames{name}]);
            out_pwer = sim.(dirNames{name}(1:end-4)).out_pwer;
            outPow_per_port(name, :) = outPow_per_port(name, :) + mean(out_pwer);
        end
        in_pwer = in_pwer + mean(sim.(dirNames{name}(1:end-4)).in_pwer);
    end
    
    
    lgd = {};
    
    for name = 1:length(dirNames)
        sim = load([F '/Topologies/' dirNames{name}]);
        lgd{end+1} = sim.(dirNames{name}(1:end-4)).topology;
    end
    
    inPow_per_port = in_pwer/length(datasets);
    outPow_per_port = outPow_per_port/length(datasets);
    
    figure('Renderer', 'painters', 'Position', [400 400 1900 1400])
    for name = 1:length(dirNames)
        plot(1:N, outPow_per_port(name, :), 'displayName', lgd{name}, 'linewidth', linewid)
        hold on
    end
    
    plot(1:N, inPow_per_port, 'displayName', 'Input Power', 'linewidth', linewid*2)
    hold off
%     l = legend('-DynamicLegend', 'fontsize', fontsz*0.6, 'interpreter','latex', 'location', 'southoutside');    
    xticks(1:N)
    xlim([0,N+1])
    a = get(gca,'YTickLabel');
    
    set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz*0.7)
    a = get(gca,'XTickLabel');
    set(gca,'XTickLabel',a,'FontName','Times','fontsize',fontsz*0.7)
    
    xlabel('Port Number', 'fontsize', fontsz*0.8, 'interpreter','latex')
    ylabel('Mean Output Power (W)', 'fontsize', fontsz*0.8, 'interpreter','latex')
%     set(gca, 'YScale', 'log')
    axis square
    drawnow;
    if 1 && printMe || 1
        if lossy && 1
            pMe_lineplot(sprintf('../Crop_Me/lossy_in_out_port_pwer_%d_N=%d.pdf', ii, N))
        else
            pMe_lineplot(sprintf('../Crop_Me/in_out_port_pwer_%d_N=%d.pdf', ii, N))
        end
    end
%     pMe_lineplot(sprintf('../Crop_Me/LEGEND.pdf'))
end

