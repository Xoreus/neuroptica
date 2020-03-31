% Function to print whatever file to pdf for figures, since this paper size
% is too large for colormaps and kills my computer
%
% Author: Simon Geoffroy-Gagnon
% Edit: 25.02.2020

function pMe_lineplot(fname)
set(gcf,'Units','inches');

screenposition = [4 4 18 13];
set(gcf, 'PaperPosition',[0 0 screenposition(3:4)], 'PaperSize',[screenposition(3:4)]);
print('-dpdf','-painters', fname)
end