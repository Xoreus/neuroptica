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
sigma = 0;

topo = {'R_P', 'C_Q_P', 'E_P', 'R_I_P'};
Ns = [4,6,8,10,12,16,32,64];

for name_idx = 1:length(topo)
    num.(topo{name_idx}) = zeros(length(Ns), 1);
end

for lossy = [0]
    
    outPwer = zeros(4, length(Ns));
    inPwer = zeros(1, length(Ns));
    for jj = 1:length(Ns)
        
        avg_outPow = zeros(4,Ns(jj));
        outPow_per_port_nl = zeros(4,Ns(jj));
        in_pwer = zeros(1,Ns(jj));
        
        F = sprintf('../Analysis/%.2f_sigma%.2f_outPorts_mean_pow', lossy, sigma);
        
        folders = dir([F sprintf('/N=%d_*', Ns(jj))]);
        foldNames = {folders.name};
        for n = 1:length(foldNames)
            dirFiles = dir([F '/' foldNames{n} '/Topologies/']);
            dirNames = {dirFiles(3:end).name};
            if length(dirNames) == 6
                dirNames = dirNames([1,2,5,6]);
            end
            if length(dirNames) ~= 4
                disp(length(dirNames))
                disp(foldNames{n})
                asdfasdf
            end
            for name = 1:length(dirNames)
                sim = load([F '/' foldNames{n} '/Topologies/' dirNames{name}]);
                out_pwer = sim.(dirNames{name}(1:end-4)).out_pwer;
                avg_outPow(name, :) = avg_outPow(name, :) + mean(out_pwer);
                num.(dirNames{name}(1:end-4))(jj) = num.(dirNames{name}(1:end-4))(jj) + 1;
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
        
        inPwer(jj) = mean(in_pwer/length(foldNames));
        
        for name = 1:length(dirNames)
            outPwer(name, jj) = mean(avg_outPow(name, :)/num.(dirNames{name}(1:end-4))(jj));
        end
    end
    
    
    figure('Renderer', 'painters', 'Position', [400 400 1900 1400])
    
    
    %     lgd = [lgd(4), lgd(3), lgd(2), lgd(1)];
    plot_idx = [4,3,2,1];
    markers = {'-d', '-s','-o','-v'};
    markersz = [20, 10, 20, 30];
    lgd{end+1} = 'Input Power';
    lgd{3} = 'Reck + Inv. Reck';
    for tt = [4,3,2,1]
        plot(Ns, outPwer(tt, :), markers{tt}, 'markersize', markersz(tt), 'MarkerFaceColor', '#c3c3c3', ...
            'linewidth', 3, 'DisplayName', lgd{tt})
        hold on
    end
    plot(Ns, inPwer, 'linewidth', linewid, 'DisplayName', lgd{end})

    [~, objh] = legend('-DynamicLegend', 'fontsize',  fontsz*0.8, 'interpreter','latex', 'location', 'east');
    %// set font size as desired
    objhl = findobj(objh, 'type', 'line'); %// objects of legend of type line
    set(objhl, 'Markersize', 20); %// set marker size as desired

    a = get(gca,'YTickLabel');
    set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz)
    a = get(gca,'XTickLabel');
    set(gca,'XTickLabel',a,'FontName','Times','fontsize',fontsz)
    
    a = get(gca,'YTickLabel');
    set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz*0.7)
    xlabel('Structure Size ($N$)', 'fontsize', fontsz*0.8, 'interpreter','latex', 'color', 'k')
    ylabel('Mean Output Power ($W$)', 'fontsize', fontsz*0.8, 'interpreter','latex')
    
    
    h = gca;
    set(h, 'YTickLabelMode','auto')
    set(h, 'XTickLabelMode','auto')
    xticks([4, 8, 12, 16, 32, 64])
    set(gca, 'YScale', 'log')
    if printMe
        pMe_lineplot(sprintf('../Crop_Me/output_pwer_loss=%.2fdB.pdf', lossy))
    end
end