 % Script to plot the Loss/Training Accuracuy/Validation Accurracy or an ONN
% model
%
% Author: Simon Geoffroy-Gagnon
% Edit: 15.02.2020


function ONN_Backprop_Plot(FOLDER, sim, topo, printMe)
fontsz = 62;
fontsz_title = 64;

for t = 1:length(topo)
    simulation = sim.(topo{t});
    
    figure('Renderer', 'painters', 'Position', [400 400 1900 1400])
    model = topo{t};
    
    yyaxis left
    plot(simulation.losses(:, 1:end), 'linewidth', 3)
    ylabel('Loss Function (MSE)', 'fontsize', fontsz, 'interpreter','latex')
    
    yyaxis('right');
    plot(simulation.trn_accuracy(:, 1:end), '--', 'linewidth', 3)
    hold on
    plot(simulation.val_accuracy(:, 1:end), '-', 'linewidth', 3)
    ylabel('Accuracy (\%)', 'fontsize', fontsz, 'interpreter','latex')
    xlabel('Epoch', 'fontsize', fontsz, 'interpreter','latex')
    
    
    ylim([0 100])
    a = get(gca,'XTickLabel');
    set(gca,'XTickLabel',a,'FontName','Times','fontsize',fontsz*0.7)
    a = get(gca,'YTickLabel');
    set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz*0.7)
    h = gca;
    set(h, 'YTickLabelMode','auto')
    set(h, 'XTickLabelMode','auto')
    
    legend({'Loss Function','Training Accuracy','Validation Accuracy'}, 'fontsize', fontsz*0.8, 'interpreter','latex', 'location', 'east');
    
    title(sprintf('%s Backpropagation', simulation.topology), 'fontsize', fontsz_title, 'interpreter','latex')
    savefig([FOLDER, sprintf('Matlab_Figs/%s_loss+acc-plot.fig', simulation.topology)])
    saveas(gcf, [FOLDER, sprintf('Matlab_Pngs/%s_loss+acc-plot.png',  simulation.topology)])
    
    if printMe
        pMe([FOLDER, simulation.topology, '-backprop-plot.pdf'])
    end
end