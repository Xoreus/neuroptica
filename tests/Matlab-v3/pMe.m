% Function to print whatever file to pdf
%
% Author: Simon Geoffroy-Gagnon
% Edit: 25.02.2020

function pMe(fname)
set(gcf,'Units','inches');
screenposition = [12  5   16  13.5];
set(gcf, 'PaperPosition',[0 0 screenposition(3:4)], 'PaperSize',[screenposition(3:4)]);
print('-dpdf','-painters', fname)
end