% Makes Matlab dirs in FOLDER for saving figs and pngs
%
% Author: Simon Geoffroy-Gagnon
% Edit: 20.01.2020


function makeMatlabDirs(FOLDER)

if ~exist([FOLDER, '/Matlab_Figs'], 'dir')
    mkdir([FOLDER, '/Matlab_Figs'])
end
if ~exist([FOLDER, '/Matlab_Pngs'], 'dir')
    mkdir([FOLDER, '/Matlab_Pngs'])
end

end
