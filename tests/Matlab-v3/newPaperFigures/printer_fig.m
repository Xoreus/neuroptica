xlbl_pt = '$\sigma_\theta$ (Rad)';
ylbl_pt = '$$\sigma_\phi$ (Rad)';
ylbl_power = '$\sigma_\phi = \sigma_\theta$ (Rad)';
xlbl_power = 'Loss/MZI (dB)';

uiopen('/home/simon/Documents/neuroptica/tests/MatlabV2/newPaperFigures/diamond_phiTheta.fig',1)
title('Classification Accuracy of Diamond Topology')
xlabel(xlbl_pt)
ylabel(ylbl_pt)
set(gcf,'Units','inches');
screenposition = [12.4271  4.7917   18.8542   12.9583];
set(gcf, 'PaperPosition',[0 0 screenposition(3:4)/1.25], 'PaperSize',[screenposition(3:4)/1.25]);

print -dpdf -painters diamond_phiTheta.pdf

uiopen('/home/simon/Documents/neuroptica/tests/MatlabV2/newPaperFigures/diamond_Power-PhaseUncert.fig',1)
title('Classification Accuracy of Diamond Topology')
xlabel(xlbl_power)
ylabel(ylbl_power)
set(gcf,'Units','inches');
screenposition = [12.4271  4.7917   18.8542   12.9583];
set(gcf, 'PaperPosition',[0 0 screenposition(3:4)/1.25], 'PaperSize',[screenposition(3:4)/1.25]);

print -dpdf -painters diamond_power-phaseUncert.pdf

uiopen('/home/simon/Documents/neuroptica/tests/MatlabV2/newPaperFigures/reck_phiTheta.fig',1)
title('Classification Accuracy of Reck Topology')
xlabel(xlbl_pt)
ylabel(ylbl_pt)
set(gcf,'Units','inches');
screenposition = [12.4271  4.7917   18.8542   12.9583];
set(gcf, 'PaperPosition',[0 0 screenposition(3:4)/1.25], 'PaperSize',[screenposition(3:4)/1.25]);

print -dpdf -painters reck_phiTheta.pdf

uiopen('/home/simon/Documents/neuroptica/tests/MatlabV2/newPaperFigures/reck_Power-PhaseUncert.fig',1)
title('Classification Accuracy of Reck Topology')
xlabel(xlbl_power)
ylabel(ylbl_power)
set(gcf,'Units','inches');
screenposition = [12.4271  4.7917   18.8542   12.9583];
set(gcf, 'PaperPosition',[0 0 screenposition(3:4)/1.25], 'PaperSize',[screenposition(3:4)/1.25]);
print -dpdf -painters reck_power-phaseUncert.pdf




% 
% xlabel()
% ylabel()
% 
% screenposition = get(gcf,'Position');
% set(gcf,'Units','inches');
% screenposition = [12.4271  4.7917   18.8542   12.9583];
% 
% set(gcf, 'PaperPosition',[0 0 screenposition(3:4)/1.25], 'PaperSize',[screenposition(3:4)/1.25]);

% print -dpdf -painters diamond_lossVphaseUncert.pdf
% 
% print -dpdf -painters reck_lossVphaseUncert.pdf

% print -dpdf -painters diamond_phiTheta.pdf
% print -dpdf -painters reck_phiTheta.pdf
