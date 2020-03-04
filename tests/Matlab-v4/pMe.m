% Function to print whatever file to pdf
%
% Author: Simon Geoffroy-Gagnon
% Edit: 25.02.2020

function pMe(fname)
set(gcf,'Units','inches');
axis square
screenposition = [4 4 18 13];
% screenposition = get(gcf, 'position');
set(gcf, 'PaperPosition',[0 0 screenposition(3:4)], 'PaperSize',[screenposition(3:4)]);
print('-dpdf','-painters', fname)
end