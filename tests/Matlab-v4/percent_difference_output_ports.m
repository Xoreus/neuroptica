% Plot % difference in port pwer output for meshes of different sizes
%
% Author: Simon Geoffroy-Gagnon
% Edit: 2020.03.20
clc; clear; close all;

printMe = true;

linewid = 3;
fontsz = 64;

lossy = 0.5; % 0.5;
datasets = 0:8;
Ns = [4,6,8,10,12,16,32];
perc_diff = [];
for jj = 1:length(Ns)
    outPow_per_port = zeros(6,Ns(jj));
    in_pwer = zeros(1,Ns(jj));
    if Ns(jj) == 16 || Ns(jj) == 32
        datasets = [0,1];
    end
        
    for ii = datasets
        if lossy == 1
            F = sprintf('/home/simon/Documents/neuroptica/tests/Analysis/outPorts_mean_pow_Lossy/N=%d_%d', Ns(jj), ii);
        elseif lossy == 0.5 && 1
            F = sprintf('/home/simon/Documents/neuroptica/tests/Analysis/outPorts_mean_pow_Lossy_half_dB_loss/N=%d_%d', Ns(jj), ii);
        else
            F = sprintf('/home/simon/Documents/neuroptica/tests/Analysis/outPorts_mean_pow/N=%d_%d', Ns(jj), ii);
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
    
    perc_diff(jj, :) = (max(outPow_per_port') - min(outPow_per_port'))./mean(outPow_per_port');
    
    
end


figure('Renderer', 'painters', 'Position', [400 400 1900 1400])
plot(Ns, perc_diff(:, 1), 'b-d', 'markersize', 20, 'MarkerFaceColor', 'k', 'linewidth', 3)
hold on
plot(Ns,   perc_diff(:, 6), 'r-v', 'markersize', 20, 'MarkerFaceColor', '#c3c3c3','linewidth', 3)


% plot(Ns, perc_diff(:, [1,6]), 'linewidth', linewid)
l = legend(lgd([1,6]), 'fontsize', fontsz*0.8, 'interpreter','latex', 'location', 'southeast');

a = get(gca,'YTickLabel');

set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz*0.7)
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',fontsz*0.7)

h = gca;
set(h, 'YTickLabelMode','auto')
set(h, 'XTickLabelMode','auto')

xlabel('Mesh Size (N)', 'fontsize', fontsz, 'interpreter','latex')
ylabel('Output Power Error', 'fontsize', fontsz, 'interpreter','latex')
% set(gca, 'YScale', 'log')

axis square
if 1 && printMe || 1
    if lossy == 1
        pMe(sprintf('../Crop_Me/lossy_output_power_percentDiff_1dB.pdf'))
    elseif lossy == 0.5
        pMe_lineplot(sprintf('../Crop_Me/lossy_output_power_percentDiff_half_dB.pdf'))
    else
        pMe_lineplot(sprintf('../Crop_Me/output_power_percentDiff.pdf'))
    end
end

