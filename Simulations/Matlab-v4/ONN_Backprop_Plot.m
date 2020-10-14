% Script to plot the Loss/Training Accuracuy/Validation Accurracy or an ONN
% model
%
% Author: Simon Geoffroy-Gagnon
% Edit: 15.02.2020


function ONN_Backprop_Plot(~, sim, topo, printMe)
fontsz = 64;

for t = 1:length(topo)
    simulation = sim.(topo{t});
    
    figure('Renderer', 'painters', 'Position', [400 400 1900 1400])
    
    a = get(gca,'XTickLabel');
    set(gca,'XTickLabel',a,'FontName','Times','fontsize',fontsz*0.8)
    a = get(gca,'YTickLabel');
    set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz*0.8)
    
    yyaxis left
    plot(simulation.losses(:, 1:end), 'linewidth', 3)
    ylabel('$\mathcal{L}_{\mathrm{(MSE)}}$', 'fontsize', fontsz*1, 'interpreter','latex')
%     ylim([0, 2])
%     ytickformat('%.1f')
%     yticks(0:0.2:2)
    
    yyaxis('right');
    plot(simulation.trn_accuracy(:, 1:end), '-k', 'linewidth', 3)
    hold on
    plot(simulation.val_accuracy(:, 1:end), '-', 'linewidth', 3)
    ylabel('Accuracy (\%)', 'fontsize', fontsz*1, 'interpreter','latex')
    xlabel('Epoch', 'fontsize', fontsz*1, 'interpreter','latex')
    
    
    title(sprintf('%s Backpropagation', simulation.topology), 'fontsize', fontsz*1, 'interpreter','latex')
    h = legend({'Loss Function','Training Accuracy','Validation Accuracy'}, 'fontsize', fontsz*0.9, ...
        'interpreter','latex', 'location', 'east');
    set(h, 'position', get(h, 'position') + [-0.125 -.015 0 0])
    
    ylim([0 100])
    
    xlim([0, length(simulation.val_accuracy)])
    h = gca;
    set(h, 'YTickLabelMode', 'auto')
    set(h, 'XTickLabelMode','auto')
    
    if printMe
        pMe_higher(['../Crop_Me/backprop-plot-', simulation.topology, sprintf('-N=%d_%.3f.pdf', simulation.N, max(simulation.val_accuracy))])  % get the accuracy to  output   different fileeees based on the ddiff acc of   trainings..
    end
    fprintf('Min MSE Loss = %.5f\n', min(simulation.losses))
    fprintf('Max Val Acc = %.2f%%\n', max(simulation.val_accuracy))
end