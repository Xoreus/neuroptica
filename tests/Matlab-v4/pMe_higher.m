% Function to print whatever file to pdf but dooble columns so higher
%
% Author: Simon Geoffroy-Gagnon
% Edit: 25.02.2020

function pMe_higher(fname)
set(gcf,'Units','inches');

screenposition = [4 4 15 14.5];
% screenposition = get(gcf, 'position');
set(gcf, 'PaperPosition',[0 0 screenposition(3:4)], 'PaperSize',[screenposition(3:4)]);
print('-dpdf','-painters', fname)
end