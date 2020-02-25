% Delta Delta Visualiser
%
% Author: Simon Geoffroy-Gagnon
% Edit: 25.02.2020

f = '/home/simon/Documents/neuroptica/tests/Analysis/Good_Plots/new-paper-dataset/Datasets';
fid = fopen([f, '/Gaussian_X_4Features_4Classes_Samples=560_Dataset.txt']);
X = textscan(fid, '%f,%f,%f,%f');
X = [X{1} X{2} X{3} X{4}];

fclose(fid);
fid = fopen([f, '/Gaussian_y_4Features_4Classes_Samples=560_Dataset.txt']);
y = textscan(fid, '%f,%f,%f,%f');
y = [y{1} y{2} y{3} y{4}];
fclose(fid);

cqp = [-0.1090+0.9876j, -0.0055+0.0030j, -0.0100-0.0065j, -0.0159-0.0102j, 0.0000+0.0000j, 0.0000+0.0000j; 
-0.0028-0.0047j, 0.3234+0.8517j, 0.0080-0.0992j, -0.0828-0.1164j, 0.1317+0.0721j, 0.0000+0.0000j; 
0.0005+0.0132j, 0.0758+0.0696j, -0.0910-0.2331j, 0.6094+0.3640j, -0.0697-0.0071j, 0.2603+0.1722j; 
0.0186+0.0062j, 0.0695-0.1401j, 0.5585-0.4288j, 0.2321-0.3559j, 0.0611-0.0505j, -0.1703+0.1225j;
0.0000+0.0000j, 0.1349+0.0713j, 0.0225+0.0753j, -0.0146-0.0914j, -0.8616+0.1199j, -0.0807+0.1135j;
0.0000+0.0000j, 0.0000+0.0000j, 0.2904-0.0570j, -0.1832+0.0360j, -0.1133-0.0976j, 0.6948-0.3165j];

rp = [0.1369+0.1396j, -0.7961+0.0843j, -0.1572-0.4475j, -0.0346-0.0695j; 
-0.4001+0.6790j, -0.0346+0.2672j, 0.2024+0.2079j, -0.1484-0.1659j; 
0.0333-0.1740j, -0.3016+0.3055j, 0.6099+0.2152j, 0.1842+0.3481j; 
0.2835+0.1703j, -0.0759-0.0456j, -0.0832+0.3015j, 0.6908-0.3742j];

c_rp = accuracy(rp, X, y, 0)/length(X)*100
c_cqp = accuracy(cqp, [zeros(length(X),2) X], y, 0)/length(X)*100

function [c] = accuracy(D, x, y, zeta)

% Get output of NN
output = abs(D*x')';
% Sort output small to big
s_output = sort(output,2);
% Get predicted class
[~, pred] = max(output,[],2);
% Get ground truth class
[~, gt] = max(y,[],2);
% Get array of correct predictions
c_cls = pred == gt;
% Get array of max - second_max < zeta
thresh = (s_output(:, end) - s_output(:, end-1) >= zeta);
% Get total correct classifications
c = sum(c_cls.*thresh);

end


function [c] = accuracy_old(D, x, y, zeta)

% Get output of NN
output = abs(x*D).^2;
% Sort output small to big
s_output = sort(output,2);
% Get predicted class
[~, pred] = max(output,[],2);
% Get ground truth class
[~, gt] = max(y,[],2);
% Get array of correct predictions
c_cls = pred == gt;
% Get array of max - second_max < zeta
thresh = (s_output(:, end) - s_output(:, end-1) >= zeta);
% Get total correct classifications
c = sum(c_cls.*thresh);

end
