% Figure out trend of figure of merits

clc; close all; clear;

Folder = '/home/simon/Documents/neuroptica/tests/Analysis/linsep/';
fig_of_merit_value = 0.75;
fontsz = 44;
N = 4:14;
vom = zeros(2,length(N));
legend_ = {};

for ii = 1:length(N)
    ActualFolder = ['N=' num2str(N(ii))];%, '-newSimValues'];
    FOLDER = [Folder, ActualFolder, '/'];
    
    [acc, sim, topo] = load_ONN_data(FOLDER, N(ii));
    makeMatlabDirs(FOLDER)
    warning( 'off', 'MATLAB:table:ModifiedAndSavedVarnames')
    
    for t = 1:length(topo)
        accuracy = acc.(topo{t}).accuracy;
        simulation = sim.(topo{t});
        legend_{end+1} = simulation.topology;
        a_of_m = zeros(1, size(accuracy, 3));
        accuracy = acc.(topo{t}).accuracy;
        simulation = sim.(topo{t});
        for loss_idx = 1:size(accuracy, 3)
            curr_acc = squeeze(accuracy(:,:,loss_idx));
            
            area_of_merit = sum(sum(curr_acc >= acc.max_accuracy*fig_of_merit_value)) * (simulation.phase_uncert_phi(2) - ...
                simulation.phase_uncert_phi(1)) * (simulation.phase_uncert_theta(2) - simulation.phase_uncert_theta(1));
            
            a_of_m(loss_idx) = area_of_merit;
        end
        vom(t, ii) = sum(a_of_m);
    end
end
figure('Renderer', 'painters', 'Position', [400 400 1800 1300])

plot(N, vom', 'linewidth', 3)

legend(legend_, 'fontsize', fontsz, 'interpreter','latex', 'location', 'best');
axis tight

xlabel('Matrix Size (N)', 'fontsize', fontsz, 'interpreter','latex')
ylabel('Figure of Merit $(Rad^2)$', 'fontsize', fontsz, 'interpreter','latex')

title(sprintf('Figure of Merit'), 'fontsize', fontsz, 'interpreter','latex')
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',fontsz*0.9)
a = get(gca,'YTickLabel');
set(gca,'YTickLabel',a,'FontName','Times','fontsize',fontsz*0.8)