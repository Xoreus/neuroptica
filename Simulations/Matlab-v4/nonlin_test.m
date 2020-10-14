Theta=[
 2.311802
 1.041695
 1.011255
 2.491578
 4.742349
 0.553411];

Phi=[
 3.136030
 2.465208
 3.126446
 0.310615
 3.732331
 1.834289];

[U, D] = optical_matrix_builder(Theta, Phi);

Xt = [0.713, 1.600, 0.539, 0.497
1.139, 2.561, 0.391, 0.729
2.420, 2.201, 2.766, 3.162
0.941, 2.909, 0.541, 0.513
1.347, 0.621, 2.078, 2.167
0.488, 1.665, 0.491, 0.633
1.211, 2.081, 0.391, 0.497
1.866, 1.406, 2.554, 2.326
0.847, 1.821, 0.379, 0.535
0.909, 2.167, 0.515, 0.661
1.353, 3.162, 0.490, 0.729
0.940, 1.748, 0.502, 0.598
1.751, 1.573, 2.100, 2.019
0.784, 1.480, 0.440, 0.497
1.498, 1.482, 1.880, 1.893
2.416, 1.752, 2.703, 2.659
1.424, 1.119, 2.271, 2.467
1.851, 1.239, 2.519, 2.699
0.997, 1.960, 0.440, 0.497
0.786, 1.980, 0.496, 0.484
0.316, 1.424, 0.394, 0.435
1.135, 1.145, 1.778, 1.859
0.911, 2.346, 0.423, 0.499
1.439, 1.254, 2.215, 2.560
1.265, 2.215, 0.625, 0.516
1.122, 1.515, 1.988, 2.018
1.709, 1.239, 2.123, 2.351
1.139, 2.321, 0.490, 0.497
0.907, 1.084, 1.662, 1.539
0.696, 1.762, 0.458, 0.508
];

yt = [1, 0, 0
1, 0, 0
0, 0, 1
1, 0, 0
0, 0, 1
1, 0, 0
1, 0, 0
0, 0, 1
1, 0, 0
1, 0, 0
1, 0, 0
1, 0, 0
0, 1, 0
1, 0, 0
0, 1, 0
0, 0, 1
0, 0, 1
0, 0, 1
1, 0, 0
1, 0, 0
1, 0, 0
0, 1, 0
1, 0, 0
0, 0, 1
1, 0, 0
0, 1, 0
0, 0, 1
1, 0, 0
0, 1, 0
1, 0, 0];

yhat = abs(D*Xt').^2';
yhat = yhat(:, 1:3);
[~, idx] =  max(yhat');
[~,  gt] = max(yt');
accuracy = sum(gt == idx)/length(Xt);
fprintf('Accuracy = %d%%\n', accuracy*100)



