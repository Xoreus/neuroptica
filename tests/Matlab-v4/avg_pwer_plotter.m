% Plot AVG out pwer
%
% Author: Simon Geoffroy-Gagnon
% Edit: 2020.03.20
clc; clear; close all;

printMe = true;

linewid = 3;
fontsz = 64;
for ii = [0,1]
    if ii ~= 0
        lgd_loc = 'southwest';
    else
        lgd_loc = 'east';
    end
    F = sprintf('/home/simon/Documents/neuroptica/tests/Analysis/pwer_out/N=32_%d', ii);
    load([F '/Topologies/' 'E_P.mat'])
    topos = fieldnames(E_P.out_pwer);
    topos = topos([2,3,5]);
    Ns = [4,6,8,10,12,14,16,20,24,28,32];
    topologies = {'Reck','Diamond','Clements', 'Input Power'};
    
    figure('Renderer', 'painters', 'Position', [400 400 1900 1400])
    for tt = 1:length(topos)
        if ii == 0
            if strcmp(topos{tt}, 'R_P')
                linewid = 7;
            elseif strcmp(topos{tt}, 'E_P')
                linewid = 4;
            else
                linewid = 3;
            end
        end
        total_mean_pwer_out = E_P.out_pwer.(topos{tt});
        if length(Ns) == length(total_mean_pwer_out)
            plot(Ns, total_mean_pwer_out, 'displayName', topologies{tt}, 'linewidth', linewid)
            hold on
        end
    end
    
    in_pwer = [];
    for N = Ns
        F = sprintf('/home/simon/Documents/neuroptica/tests/Analysis/pwer_out/N=%d_%d', N, ii);
        X = load([F '/Datasets/X.txt']);
        in_pwer(end+1) = mean(sum(X'.^2));
    end
    
    if ii == 0
        plot(Ns, in_pwer,  'ko', 'markersize', 20, 'MarkerFaceColor', '#c3c3c3', 'displayName', topologies{end}, 'linewidth', 1)
    else
        plot(Ns, in_pwer,  'k-o', 'markersize', 20, 'MarkerFaceColor', '#c3c3c3', 'displayName', topologies{end}, 'linewidth', 1)
    end
    a = get(gca,'YTickLabel');
    set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz*0.7)
    h = gca;
    set(h, 'XTickLabelMode','auto')
    xticks(Ns)
    xlabel('Structure Size ($N$)', 'fontsize', fontsz*0.8, 'interpreter','latex')
    ylabel('Mean Total Output Power (W)', 'fontsize', fontsz*0.8, 'interpreter','latex')
    legend('-DynamicLegend', 'fontsize', fontsz*0.6, 'interpreter','latex', 'location', lgd_loc);
    
    if ii == 1 || ii == 2 || ii == 0
        set(gca, 'YScale', 'log')
        set(h, 'YTickLabelMode','auto')
    end
    
    axis square
    
    if 1 && printMe
        pMe_lineplot(sprintf('../Crop_Me/in_out_pwer-log-%d.pdf', ii))
    end
end