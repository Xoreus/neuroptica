% Plot avg pwer per port
%
% Author: Simon Geoffroy-Gagnon
% Edit: 2020.03.20
clc; clear; close all;

printMe = false;

linewid = 3;
fontsz = 64;
ii = 4;
lossy = 0.5; % 0.5;
datasets = 0:9;
Ns = 12;
for N = Ns
    outPow_per_port = zeros(6,N);
    outPow_per_port_noloss = zeros(6,N);
    in_pwer = zeros(1,N);
    
    for ii = datasets
        if lossy == 1
            F = sprintf('../Analysis/outPorts_mean_pow_Lossy/N=%d_%d', N, ii);
        elseif lossy == 0.5 && 1
            F = sprintf('/home/edwar/Documents/Github_Projects/neuroptica/tests/Analysis/outPorts_mean_pow_Lossy_half_dB_loss/N=%d_%d', N, ii);
        else
            F = sprintf('/home/edwar/Documents/Github_Projects/neuroptica/tests/Analysis/outPorts_mean_pow/N=%d_%d', N, ii);
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
    for name = [1, 6] % 1:length(dirNames)
        plot(1:N, outPow_per_port(name, :), 'displayName', lgd{name}, 'linewidth', linewid)
        disp(lgd{name})
        disp((max(outPow_per_port(name, :)) - min(outPow_per_port(name, :)))/mean(outPow_per_port(name, :)))
        hold on
    end
 
    plot(1:N, inPow_per_port, 'displayName', 'Input Power', 'linewidth', linewid*2)
    hold off
    l = legend('-DynamicLegend', 'fontsize', fontsz, 'interpreter','latex', 'location', 'west');
    %     xticks(1:N)
    %     xlim([0,N+1])
    %     ylim([0, 0.35])
    a = get(gca,'YTickLabel');
    
    set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz)
    a = get(gca,'XTickLabel');
    set(gca,'XTickLabel',a,'FontName','Times','fontsize',fontsz)
    
    xlabel('Port Number', 'fontsize', fontsz, 'interpreter','latex')
    ylabel('Mean Output Power (W)', 'fontsize', fontsz, 'interpreter','latex')
    h = gca;
    set(h, 'YTickLabelMode','auto')
    set(h, 'XTickLabelMode','auto')
    axis square
    drawnow;
    if 1 && printMe
        if lossy == 1
            pMe(sprintf('../Crop_Me/lossy_in_out_port_pwer_%d_N=%d_1dB.pdf', ii, N))
        elseif lossy == 0.5
            pMe_lineplot(sprintf('../Crop_Me/lossy_in_out_port_pwer_%d_N=%d_half_dB.pdf', ii, N))
        else
            pMe_lineplot(sprintf('../Crop_Me/in_out_port_pwer_%d_N=%d.pdf', ii, N))
        end
    end
    %     pMe_lineplot(sprintf('../Crop_Me/LEGEND.pdf'))
end

