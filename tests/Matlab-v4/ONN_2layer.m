clc; clear;

Theta=[
1.507287483
1.752793582
1.304772931
5.125886141
2.732086518
2.056416028
1.821914828
2.282399544
1.310339926
1.008303134
1.738949103
1.833982112];

Phi=[
5.843894426
4.428772669
0.867642822
5.652938855
0.266078571
3.147036071
6.214548360
5.920605111
0.590662057
1.927144194
2.367647869
0.570578663];

[~, D1] = optical_matrix_builder(Theta(1:6), Phi(1:6));
[~, D2] = optical_matrix_builder(Theta(7:end), Phi(7:end));

Xt = [1.007, 1.889, 0.376, 0.316
1.957, 0.940, 2.233, 2.131
1.460, 1.466, 1.792, 1.713
0.512, 0.880, 0.400, 0.475
1.854, 1.221, 2.502, 2.708
1.865, 1.998, 2.029, 2.002
2.139, 1.652, 2.427, 2.439
2.357, 1.841, 3.002, 3.121
0.982, 2.518, 0.539, 0.740
1.315, 3.162, 0.439, 0.540
1.016, 1.914, 0.439, 0.540
1.723, 1.575, 2.307, 2.704
1.222, 2.602, 0.587, 0.621
2.240, 0.651, 2.427, 2.146
1.542, 1.958, 2.181, 2.788
2.291, 1.918, 2.188, 2.848
2.081, 1.457, 2.639, 2.689
1.430, 1.124, 2.084, 2.246
2.195, 1.466, 2.800, 2.846
0.973, 1.957, 0.326, 0.506
2.010, 1.662, 2.464, 2.622
1.834, 1.803, 2.166, 2.280
0.640, 1.425, 0.451, 0.430
1.396, 2.492, 0.535, 0.765
2.143, 1.992, 2.644, 2.716
1.514, 1.237, 2.341, 2.281
1.168, 2.111, 0.564, 0.686
1.360, 1.793, 1.761, 1.888
1.221, 2.386, 0.490, 0.737
2.376, 2.392, 2.626, 3.162
2.154, 1.637, 2.256, 2.817
1.273, 0.316, 1.992, 2.062
1.842, 1.516, 1.993, 1.844
2.235, 1.957, 2.312, 2.114
1.735, 1.221, 2.109, 2.275
1.536, 2.705, 0.457, 0.664
1.659, 1.536, 1.804, 1.807
1.903, 1.930, 2.613, 2.740
0.432, 1.359, 0.365, 0.578
1.079, 1.591, 0.532, 0.675
1.957, 1.803, 2.448, 2.885
0.963, 2.307, 0.316, 0.549
2.025, 1.068, 2.262, 1.755
2.154, 1.637, 2.158, 1.950
1.261, 2.369, 0.484, 0.572
1.351, 1.495, 1.922, 2.151
1.912, 1.020, 2.678, 2.620
1.590, 1.223, 1.637, 1.596
0.894, 1.573, 0.406, 0.528
2.127, 1.811, 2.041, 2.030
0.961, 1.927, 0.446, 0.620
1.568, 1.818, 1.665, 1.780
1.505, 1.666, 2.019, 2.555
2.226, 1.832, 2.436, 2.482
1.435, 1.360, 1.814, 1.733
2.025, 1.864, 2.544, 2.564
0.859, 1.824, 0.511, 0.505
1.442, 2.864, 0.534, 0.890
1.735, 2.053, 2.404, 2.817
2.239, 1.944, 2.650, 2.732
2.468, 1.647, 2.971, 2.612
1.016, 1.776, 0.341, 0.540
1.868, 1.240, 2.233, 1.951
2.339, 1.947, 2.574, 2.669
1.454, 1.060, 1.981, 2.253
1.291, 2.667, 0.505, 0.640
1.715, 3.134, 0.611, 0.620
2.482, 1.071, 2.937, 2.530
0.958, 2.076, 0.495, 0.528
2.274, 1.499, 2.649, 2.600
1.167, 2.280, 0.368, 0.644
1.012, 1.890, 0.544, 0.368
0.848, 1.697, 0.375, 0.508
2.034, 1.504, 2.749, 2.434
0.823, 1.192, 0.364, 0.686
1.784, 1.361, 2.064, 1.867
0.316, 0.574, 0.402, 0.374
2.660, 1.751, 3.060, 2.932
1.058, 2.224, 0.521, 0.633
1.988, 1.823, 2.594, 2.964
1.066, 1.853, 0.508, 0.396
1.135, 2.053, 0.439, 0.540
1.314, 1.528, 1.901, 2.608
1.675, 1.360, 2.060, 1.841
1.974, 1.499, 1.913, 1.841
2.051, 1.417, 2.760, 2.918
2.320, 1.278, 2.456, 2.455
1.615, 1.274, 2.353, 2.665
1.324, 1.018, 2.108, 2.159
1.714, 1.581, 1.981, 1.874
1.016, 2.192, 0.537, 0.974
1.967, 1.392, 2.062, 1.740
2.478, 2.231, 2.557, 2.949
3.162, 1.270, 3.162, 2.546
1.375, 1.221, 2.158, 2.492
1.735, 1.360, 1.863, 1.733
0.929, 2.243, 0.348, 0.480
1.135, 2.192, 0.488, 0.540
1.675, 1.499, 2.158, 2.275
0.656, 1.776, 0.390, 0.540
1.924, 1.317, 2.439, 2.249
1.437, 1.077, 1.708, 1.427
1.607, 0.434, 2.343, 2.261
1.648, 1.432, 2.201, 2.404
1.036, 1.550, 0.518, 0.764
1.488, 1.332, 1.694, 1.494
2.633, 0.944, 3.140, 2.817
1.065, 2.323, 0.615, 0.588
0.929, 1.749, 0.475, 0.532
1.164, 2.677, 0.634, 0.709
1.980, 1.284, 2.406, 2.452
1.594, 0.889, 1.629, 1.378
2.436, 1.528, 2.589, 2.302
2.278, 1.671, 2.852, 2.533
1.794, 0.805, 2.207, 2.383
2.290, 1.404, 2.332, 2.479
0.654, 0.995, 0.556, 0.366
0.610, 2.089, 0.534, 0.543
0.991, 1.913, 0.590, 0.535
2.254, 1.463, 2.846, 2.456
1.180, 2.253, 0.590, 0.535
1.338, 1.161, 2.221, 2.234
1.416, 1.336, 1.753, 1.695
0.792, 1.716, 0.468, 0.414
1.546, 1.227, 1.542, 1.458
0.808, 1.561, 0.416, 0.538
1.746, 1.270, 1.677, 1.615
1.910, 1.519, 1.990, 1.954
2.048, 1.725, 2.646, 2.544
1.654, 1.033, 2.364, 2.679
0.874, 1.697, 0.472, 0.680
2.067, 1.237, 1.879, 1.654
1.911, 1.077, 2.268, 3.162
1.433, 1.121, 1.784, 1.406
1.370, 1.460, 1.830, 1.624
0.978, 1.416, 0.527, 0.465
0.672, 2.001, 0.715, 0.524
0.904, 2.130, 0.530, 0.624
1.236, 2.388, 0.569, 0.749
1.639, 1.432, 1.972, 1.873
1.280, 2.188, 0.585, 0.673
2.633, 1.234, 2.977, 2.495
1.553, 1.386, 1.789, 1.795
1.749, 1.234, 2.243, 1.951
1.246, 1.334, 2.040, 2.654
0.900, 1.712, 0.510, 0.411
2.139, 1.613, 2.496, 2.862
2.137, 1.305, 2.801, 2.439
1.858, 1.482, 2.311, 2.664
1.172, 2.026, 0.502, 0.582
0.913, 2.000, 0.570, 0.637
0.738, 1.687, 0.636, 0.535
1.791, 1.625, 2.237, 2.546
2.102, 1.518, 2.639, 2.922
2.001, 0.894, 2.564, 2.277
2.442, 1.691, 2.485, 2.730
1.253, 2.673, 0.666, 0.643
1.658, 1.790, 1.977, 1.702
0.858, 1.718, 0.652, 0.685
1.717, 1.189, 1.828, 1.640
0.994, 2.017, 0.649, 0.593
1.042, 1.682, 0.542, 0.589
1.875, 1.460, 2.427, 2.277
2.441, 1.539, 2.544, 2.295
1.596, 1.032, 2.328, 2.522
2.148, 1.863, 2.396, 3.086
2.134, 1.654, 1.989, 1.832
0.975, 2.108, 0.614, 0.690
1.764, 1.412, 2.072, 1.756
1.785, 1.695, 2.403, 2.591
1.812, 1.573, 2.427, 2.277
1.250, 3.086, 0.552, 0.747
1.767, 1.744, 2.255, 2.231
1.926, 1.647, 2.451, 2.332
1.749, 1.913, 2.472, 2.930
1.930, 0.851, 2.419, 2.616
0.635, 1.818, 0.659, 0.517
1.021, 2.175, 0.750, 0.738
0.675, 1.771, 0.599, 0.574
0.838, 1.598, 0.633, 0.735
1.614, 1.068, 2.439, 2.174
2.074, 0.921, 2.531, 2.021
2.125, 1.809, 2.575, 3.010
2.094, 0.870, 2.651, 2.428
2.286, 1.221, 2.561, 2.149
0.661, 2.046, 0.549, 0.700
1.815, 1.680, 1.799, 1.860
1.833, 0.855, 2.436, 2.162
0.869, 2.177, 0.568, 0.763
0.618, 1.902, 0.552, 0.629
2.033, 1.696, 2.412, 2.345
1.807, 1.496, 2.224, 2.878
1.875, 1.460, 2.564, 2.713
0.843, 1.565, 0.616, 0.745
1.969, 1.827, 2.422, 2.845
0.901, 1.623, 0.607, 0.817
1.749, 1.347, 2.472, 2.277
1.455, 1.756, 2.224, 2.791
1.722, 1.409, 2.410, 2.440
2.005, 1.984, 2.563, 2.878
0.686, 1.555, 0.477, 0.471
2.036, 1.343, 2.458, 2.526
0.690, 1.675, 0.657, 0.566
1.837, 2.194, 2.505, 2.926
0.920, 1.860, 0.719, 0.603
0.478, 1.530, 0.541, 0.711
1.004, 2.807, 0.638, 0.550
1.063, 1.943, 0.593, 0.687
1.933, 0.469, 2.589, 1.762
2.291, 1.999, 2.385, 2.210
1.191, 2.097, 0.640, 0.421
1.228, 1.719, 2.089, 2.837
1.409, 1.363, 1.764, 1.720
1.408, 1.404, 1.769, 1.774
1.077, 1.807, 0.553, 0.658
2.094, 1.901, 2.197, 2.289
1.630, 1.542, 2.265, 2.182
0.999, 1.952, 0.661, 0.585
1.802, 1.550, 2.017, 1.890
1.015, 2.123, 0.659, 0.730
0.675, 2.139, 0.361, 0.535
1.762, 1.256, 1.927, 1.775
1.758, 1.354, 1.726, 1.695
0.801, 1.460, 0.544, 0.427
2.406, 1.355, 2.597, 2.173
0.916, 2.095, 0.532, 0.475
0.778, 1.843, 0.541, 0.523
1.812, 1.121, 2.335, 2.386
2.028, 1.632, 2.424, 2.294
2.564, 1.365, 2.876, 2.350
1.719, 0.977, 2.097, 2.473
0.801, 1.913, 0.774, 0.535
1.848, 0.967, 2.506, 1.885
1.202, 2.521, 0.629, 0.715
0.966, 1.798, 0.582, 0.553
1.604, 1.084, 2.248, 2.467
0.998, 2.010, 0.617, 0.575
0.835, 1.680, 0.451, 0.517
0.956, 1.925, 0.545, 0.730
1.559, 1.460, 2.105, 2.277
0.667, 1.819, 0.354, 0.356
1.749, 1.800, 2.656, 3.039
1.828, 1.751, 2.194, 2.578
1.433, 1.121, 2.243, 2.386
1.901, 1.419, 2.520, 2.759
0.927, 2.139, 0.544, 0.535
2.573, 1.507, 2.966, 2.919
1.016, 2.463, 0.641, 0.829
2.408, 1.982, 2.698, 3.022
1.343, 2.861, 0.597, 0.722
1.180, 2.479, 0.682, 0.753
1.975, 1.682, 2.605, 2.867
0.675, 1.687, 0.544, 0.535
2.379, 1.788, 2.835, 2.803
1.090, 1.928, 0.666, 0.769
1.768, 1.113, 1.824, 1.615
1.622, 1.007, 2.472, 1.842
1.257, 1.172, 1.978, 2.101
0.364, 1.353, 0.526, 0.416
0.943, 1.572, 0.538, 0.750
1.062, 1.922, 0.606, 0.316
0.683, 1.160, 0.612, 0.549
1.014, 2.191, 0.585, 0.454
1.833, 1.287, 2.242, 2.204
2.001, 1.800, 2.518, 3.039
1.644, 1.244, 2.232, 2.707
0.734, 1.317, 0.316, 0.498
0.710, 1.518, 0.575, 0.422
2.144, 1.722, 2.377, 2.534
0.745, 1.670, 0.623, 0.578
0.976, 2.164, 0.465, 0.594
1.798, 1.115, 2.213, 2.254
0.759, 1.428, 0.683, 0.580
1.082, 1.860, 0.589, 0.723
2.435, 1.670, 2.843, 2.519
2.001, 1.460, 2.289, 2.821
2.443, 1.234, 2.702, 2.386
0.991, 1.800, 0.682, 0.862
2.380, 1.347, 2.794, 2.277
0.864, 1.573, 0.590, 0.535
0.934, 1.890, 0.578, 0.606
2.200, 1.601, 2.655, 2.707
2.122, 1.091, 2.620, 2.595
1.433, 2.592, 0.453, 0.535
2.113, 2.177, 2.443, 2.379
0.449, 1.407, 0.502, 0.478
0.991, 2.366, 0.636, 0.535
1.749, 1.121, 2.151, 2.277
0.789, 1.494, 0.493, 0.505
1.095, 0.706, 1.996, 2.469
0.864, 2.139, 0.544, 0.427
2.064, 1.234, 2.105, 1.842
2.128, 1.573, 2.381, 2.604
0.991, 2.253, 0.590, 0.753
1.875, 1.687, 2.243, 2.495
2.373, 1.545, 2.567, 2.317
2.064, 1.460, 2.427, 2.604
0.987, 2.068, 0.640, 0.737
0.801, 1.913, 0.636, 0.535
0.810, 1.717, 0.460, 0.540
0.925, 2.117, 0.663, 0.675
1.682, 0.748, 2.279, 2.715
0.958, 0.858, 2.052, 2.009
1.433, 1.121, 2.243, 2.386
2.379, 1.740, 2.642, 2.689
1.071, 2.242, 0.562, 0.615
1.785, 1.177, 1.996, 1.649
0.549, 1.347, 0.544, 0.535
2.001, 1.573, 2.059, 1.951
0.642, 1.552, 0.456, 0.426
1.288, 0.336, 1.874, 1.979
0.991, 2.366, 0.590, 0.644
1.306, 1.460, 1.784, 1.733
1.954, 1.212, 2.430, 2.624
1.129, 1.938, 0.604, 0.680
0.864, 1.573, 0.590, 0.427
1.938, 1.347, 2.013, 1.733
1.397, 1.308, 2.359, 2.781
1.908, 1.478, 2.136, 1.904
1.130, 2.463, 0.589, 0.680
1.783, 1.112, 2.408, 2.258
1.074, 2.427, 0.568, 0.578
1.032, 1.170, 1.929, 2.549
1.495, 1.596, 1.799, 1.795
1.888, 1.993, 2.394, 2.936
0.717, 1.974, 0.570, 0.838
1.224, 0.966, 2.126, 2.681
2.051, 1.730, 2.444, 2.968
2.723, 1.605, 2.973, 2.675
1.145, 1.095, 1.837, 2.341
2.532, 1.958, 2.789, 2.799
1.370, 0.894, 2.197, 2.495
0.316, 1.325, 0.561, 0.473
1.196, 1.065, 1.418, 1.337
1.858, 1.810, 2.476, 2.939
1.654, 1.818, 2.100, 1.906
1.559, 0.555, 2.197, 1.951
0.835, 1.707, 0.556, 0.631
0.991, 2.026, 0.544, 0.535
1.656, 1.308, 2.011, 2.354
1.389, 1.037, 2.076, 2.424
0.977, 2.276, 0.529, 0.537
0.736, 1.743, 0.544, 0.696
0.738, 1.687, 0.499, 0.535
0.668, 1.412, 0.493, 0.477
1.396, 1.407, 2.191, 2.867
0.949, 1.932, 0.467, 0.643
0.490, 1.131, 0.557, 0.570
1.559, 1.913, 1.968, 2.059
1.622, 1.460, 2.013, 1.842
0.354, 1.275, 0.595, 0.625
1.848, 1.831, 2.485, 3.075
1.812, 0.798, 2.596, 2.453
1.180, 1.913, 0.682, 0.535
1.444, 1.063, 2.244, 2.656
1.612, 1.338, 2.058, 2.659
1.496, 1.460, 2.243, 2.277
0.845, 2.000, 0.484, 0.448
2.173, 2.190, 2.753, 2.774
1.798, 1.499, 1.709, 1.593
0.662, 1.453, 0.496, 0.441
0.675, 1.573, 0.590, 0.535
1.404, 2.682, 0.690, 0.701
1.893, 1.540, 2.121, 2.066
2.001, 1.573, 2.472, 2.930
1.604, 0.955, 1.599, 1.394
1.253, 2.901, 0.533, 0.527
0.931, 2.072, 0.511, 0.552
1.703, 1.608, 1.942, 1.803
1.796, 1.434, 2.368, 3.069
1.259, 1.259, 1.915, 2.142
0.954, 2.017, 0.581, 0.474
1.803, 0.997, 2.504, 2.156
1.970, 1.260, 2.445, 2.418
1.126, 2.213, 0.667, 0.595
2.041, 1.852, 2.553, 2.958
0.795, 1.864, 0.439, 0.595
1.109, 1.839, 0.656, 0.569
0.927, 1.913, 0.590, 0.535
1.717, 1.640, 2.034, 1.790
1.180, 2.479, 0.499, 0.753
0.845, 1.393, 0.508, 0.466
1.069, 2.394, 0.538, 0.373
0.927, 1.913, 0.636, 0.753
1.348, 1.506, 2.094, 2.876
0.598, 1.424, 0.492, 0.694
0.923, 1.444, 0.657, 0.493
2.191, 1.687, 2.059, 1.842
1.370, 3.045, 0.590, 0.753
1.114, 2.365, 0.758, 0.594
0.714, 1.704, 0.509, 0.390
2.067, 1.195, 2.140, 1.838
1.812, 1.687, 2.335, 2.821
2.128, 1.687, 2.518, 2.821
1.596, 1.393, 2.148, 2.729
1.049, 2.163, 0.669, 0.610
2.129, 1.384, 2.610, 2.839
0.771, 1.973, 0.413, 0.478
1.335, 2.135, 0.741, 0.693
0.868, 1.882, 0.612, 0.643
1.332, 1.131, 2.045, 2.544
2.066, 1.130, 2.515, 2.456
2.078, 1.868, 2.766, 2.915
0.802, 1.565, 0.659, 0.528
2.465, 1.836, 2.744, 2.986
0.602, 1.635, 0.592, 0.663
0.833, 1.821, 0.558, 0.686
1.686, 1.413, 2.468, 2.763
1.953, 2.039, 2.521, 2.696
0.991, 2.366, 0.774, 0.753
0.772, 1.392, 0.446, 0.407
2.210, 1.613, 2.730, 2.682
1.291, 2.152, 0.716, 0.553
0.967, 2.200, 0.655, 0.780
0.868, 1.708, 0.654, 0.659
1.937, 1.339, 2.202, 1.625
1.682, 1.485, 1.848, 1.796
2.088, 1.874, 2.667, 2.751
2.408, 1.704, 2.735, 2.973
0.947, 1.997, 0.588, 0.578
1.063, 2.313, 0.555, 0.689
0.879, 1.996, 0.510, 0.770
1.196, 2.290, 0.506, 0.504
0.927, 1.460, 0.636, 0.535
1.243, 2.026, 0.499, 0.535
0.934, 2.044, 0.511, 0.577
1.845, 1.774, 2.667, 2.990
1.893, 1.480, 1.953, 1.851
1.433, 1.234, 2.243, 2.930
1.812, 1.347, 1.876, 1.733
1.030, 2.139, 0.608, 0.602
1.573, 1.785, 2.011, 1.889
1.233, 2.323, 0.654, 0.595
1.812, 1.687, 1.968, 1.951
0.970, 2.051, 0.553, 0.622
2.570, 1.460, 2.931, 2.604
1.219, 2.441, 0.569, 0.742
0.808, 1.547, 0.456, 0.560
1.204, 2.329, 0.568, 0.506
0.876, 2.327, 0.578, 0.346
0.675, 1.913, 0.544, 0.644
1.044, 2.020, 0.632, 0.837
1.024, 2.072, 0.504, 0.436
1.740, 1.356, 1.999, 1.698
1.686, 1.302, 2.390, 2.384
2.267, 1.713, 2.234, 2.062
1.180, 1.644, 1.687, 1.638
0.755, 1.456, 0.658, 0.704
2.231, 1.695, 2.468, 2.730
1.890, 1.665, 2.445, 2.829
1.048, 2.596, 0.666, 0.748
2.606, 2.042, 2.719, 2.785
1.424, 1.560, 2.096, 2.483
1.025, 2.141, 0.743, 0.733
2.265, 1.496, 2.436, 2.104
0.954, 1.961, 0.523, 0.746
1.584, 2.039, 0.738, 0.737
0.778, 1.848, 0.462, 0.513
2.324, 1.284, 2.700, 2.386
1.180, 1.913, 0.590, 0.753
0.693, 2.258, 0.502, 0.501
1.858, 1.062, 2.510, 2.648
1.201, 2.055, 0.614, 0.597
2.522, 1.584, 2.741, 2.853
1.749, 1.800, 2.059, 2.059
2.345, 1.864, 2.738, 2.642
1.775, 1.084, 2.475, 2.253
1.812, 1.234, 2.472, 2.604
0.549, 1.460, 0.499, 0.535
1.302, 2.371, 0.659, 0.817
1.757, 1.683, 2.168, 2.419
0.503, 1.755, 0.513, 0.524
1.043, 1.880, 0.692, 0.701
2.487, 1.764, 2.602, 2.687
2.186, 1.642, 2.523, 2.832
0.485, 1.460, 0.407, 0.427
0.801, 1.573, 0.636, 0.535
1.370, 2.366, 0.682, 0.644
1.571, 2.073, 2.020, 2.132
0.953, 0.647, 1.924, 2.687
1.065, 1.185, 2.111, 2.986
1.641, 0.892, 2.251, 2.351
0.732, 2.390, 0.493, 0.430
1.622, 1.234, 2.059, 1.624
0.801, 1.460, 0.544, 0.644
2.071, 1.461, 2.517, 2.577
0.873, 1.601, 0.516, 0.653
0.934, 1.990, 0.440, 0.718
1.100, 3.162, 0.712, 0.476
0.920, 1.781, 0.487, 0.571
1.382, 1.455, 1.655, 1.785
0.927, 2.026, 0.499, 0.644
0.929, 2.215, 0.729, 0.593
2.001, 1.573, 1.922, 1.842
0.786, 1.726, 0.561, 0.546
1.712, 1.470, 2.150, 2.635
2.420, 1.691, 2.777, 2.594
1.859, 1.570, 2.029, 1.979
2.001, 1.800, 2.518, 2.604
1.177, 2.099, 0.631, 0.793
1.789, 1.718, 2.322, 2.639
1.876, 0.946, 2.555, 2.933
1.158, 2.395, 0.613, 0.626
1.000, 1.988, 0.510, 0.634
2.164, 1.822, 2.842, 2.576
2.694, 1.515, 2.763, 2.566
1.066, 1.942, 0.597, 0.587
0.991, 2.026, 0.544, 0.644
0.608, 1.741, 0.485, 0.620
0.706, 1.144, 0.461, 0.502
0.803, 1.561, 0.479, 0.510
1.845, 1.537, 2.033, 1.759
0.683, 1.806, 0.651, 0.651
2.085, 1.201, 2.497, 2.431
0.838, 1.464, 0.568, 0.662
1.090, 1.796, 0.689, 0.610
1.261, 2.328, 0.580, 0.552
2.405, 1.536, 2.818, 2.570
1.388, 1.256, 1.772, 1.508
2.514, 1.306, 2.800, 2.685
0.861, 1.718, 0.491, 0.551
2.225, 1.401, 2.724, 2.163
1.869, 1.789, 2.492, 2.382
1.315, 1.498, 2.187, 3.107
1.644, 1.448, 1.963, 1.853
1.754, 1.410, 2.074, 1.842
0.981, 2.161, 0.526, 0.574
1.210, 2.587, 0.561, 0.601
0.459, 1.368, 0.702, 0.430
2.489, 2.042, 2.847, 3.109
1.164, 1.220, 1.963, 2.019
1.790, 1.476, 2.202, 2.774
0.864, 0.894, 1.968, 2.168
1.752, 1.791, 1.885, 1.718
1.054, 2.705, 0.590, 0.427
0.573, 1.016, 0.534, 0.457
1.306, 1.764, 1.962, 1.985
1.875, 1.460, 2.289, 2.495
1.554, 1.365, 1.749, 1.746
0.708, 1.925, 0.555, 0.632
2.185, 1.949, 2.670, 2.927
1.507, 1.740, 1.779, 1.758
0.455, 1.333, 0.437, 0.530
2.633, 1.460, 2.702, 2.821
0.856, 1.886, 0.467, 0.682
1.117, 2.253, 0.590, 0.535
0.864, 1.460, 0.544, 0.535
2.105, 1.693, 2.450, 3.057
0.897, 1.207, 0.619, 0.585
0.574, 1.584, 0.522, 0.364
0.645, 1.652, 0.511, 0.592
0.693, 1.681, 0.541, 0.447
1.080, 1.840, 0.679, 0.575
0.981, 2.371, 0.593, 0.608
1.906, 0.895, 2.331, 2.214
2.064, 1.687, 2.610, 2.821
1.003, 1.713, 0.601, 0.630
1.370, 1.007, 1.508, 1.406
2.023, 1.533, 2.352, 2.454
0.692, 1.927, 0.445, 0.520
1.928, 1.750, 2.573, 2.780
1.252, 2.279, 0.496, 0.646
1.841, 1.923, 2.175, 2.272
2.643, 1.706, 2.904, 2.690
1.532, 1.144, 2.028, 2.406
2.317, 2.139, 2.702, 3.039
0.694, 1.837, 0.528, 0.596
1.527, 1.105, 2.195, 2.057
1.372, 0.738, 1.352, 1.162];

Yt = [1, 0, 0
0, 0, 1
0, 1, 0
1, 0, 0
0, 0, 1
0, 1, 0
0, 0, 1
0, 0, 1
1, 0, 0
1, 0, 0
1, 0, 0
0, 0, 1
1, 0, 0
0, 0, 1
0, 0, 1
0, 0, 1
0, 0, 1
0, 0, 1
0, 0, 1
1, 0, 0
0, 0, 1
0, 0, 1
1, 0, 0
1, 0, 0
0, 0, 1
0, 0, 1
1, 0, 0
0, 1, 0
1, 0, 0
0, 0, 1
0, 0, 1
0, 0, 1
0, 1, 0
0, 1, 0
0, 0, 1
1, 0, 0
0, 1, 0
0, 0, 1
1, 0, 0
1, 0, 0
0, 0, 1
1, 0, 0
0, 1, 0
0, 1, 0
1, 0, 0
0, 0, 1
0, 0, 1
0, 1, 0
1, 0, 0
0, 1, 0
1, 0, 0
0, 1, 0
0, 0, 1
0, 0, 1
0, 1, 0
0, 0, 1
1, 0, 0
1, 0, 0
0, 0, 1
0, 0, 1
0, 0, 1
1, 0, 0
0, 0, 1
0, 0, 1
0, 0, 1
1, 0, 0
1, 0, 0
0, 0, 1
1, 0, 0
0, 0, 1
1, 0, 0
1, 0, 0
1, 0, 0
0, 0, 1
1, 0, 0
0, 1, 0
1, 0, 0
0, 0, 1
1, 0, 0
0, 0, 1
1, 0, 0
1, 0, 0
0, 0, 1
0, 1, 0
0, 1, 0
0, 0, 1
0, 0, 1
0, 0, 1
0, 0, 1
0, 1, 0
1, 0, 0
0, 1, 0
0, 0, 1
0, 0, 1
0, 0, 1
0, 1, 0
1, 0, 0
1, 0, 0
0, 0, 1
1, 0, 0
0, 0, 1
0, 1, 0
0, 0, 1
0, 0, 1
1, 0, 0
0, 1, 0
0, 0, 1
1, 0, 0
1, 0, 0
1, 0, 0
0, 0, 1
0, 1, 0
0, 0, 1
0, 0, 1
0, 0, 1
0, 0, 1
1, 0, 0
1, 0, 0
1, 0, 0
0, 0, 1
1, 0, 0
0, 0, 1
0, 1, 0
1, 0, 0
0, 1, 0
1, 0, 0
0, 1, 0
0, 1, 0
0, 0, 1
0, 0, 1
1, 0, 0
0, 1, 0
0, 0, 1
0, 1, 0
0, 1, 0
1, 0, 0
1, 0, 0
1, 0, 0
1, 0, 0
0, 1, 0
1, 0, 0
0, 0, 1
0, 1, 0
0, 0, 1
0, 0, 1
1, 0, 0
0, 0, 1
0, 0, 1
0, 0, 1
1, 0, 0
1, 0, 0
1, 0, 0
0, 0, 1
0, 0, 1
0, 0, 1
0, 0, 1
1, 0, 0
0, 1, 0
1, 0, 0
0, 1, 0
1, 0, 0
1, 0, 0
0, 0, 1
0, 1, 0
0, 0, 1
0, 0, 1
0, 1, 0
1, 0, 0
0, 1, 0
0, 0, 1
0, 0, 1
1, 0, 0
0, 0, 1
0, 0, 1
0, 0, 1
0, 0, 1
1, 0, 0
1, 0, 0
1, 0, 0
1, 0, 0
0, 0, 1
0, 0, 1
0, 0, 1
0, 0, 1
0, 0, 1
1, 0, 0
0, 1, 0
0, 0, 1
1, 0, 0
1, 0, 0
0, 0, 1
0, 0, 1
0, 0, 1
1, 0, 0
0, 0, 1
1, 0, 0
0, 0, 1
0, 0, 1
0, 0, 1
0, 0, 1
1, 0, 0
0, 0, 1
1, 0, 0
0, 0, 1
1, 0, 0
1, 0, 0
1, 0, 0
1, 0, 0
0, 0, 1
0, 1, 0
1, 0, 0
0, 0, 1
0, 1, 0
0, 1, 0
1, 0, 0
0, 1, 0
0, 0, 1
1, 0, 0
0, 1, 0
1, 0, 0
1, 0, 0
0, 1, 0
0, 1, 0
1, 0, 0
0, 0, 1
1, 0, 0
1, 0, 0
0, 0, 1
0, 0, 1
0, 0, 1
0, 0, 1
1, 0, 0
0, 0, 1
1, 0, 0
1, 0, 0
0, 0, 1
1, 0, 0
1, 0, 0
1, 0, 0
0, 0, 1
1, 0, 0
0, 0, 1
0, 0, 1
0, 0, 1
0, 0, 1
1, 0, 0
0, 0, 1
1, 0, 0
0, 0, 1
1, 0, 0
1, 0, 0
0, 0, 1
1, 0, 0
0, 0, 1
1, 0, 0
0, 1, 0
0, 0, 1
0, 0, 1
1, 0, 0
1, 0, 0
1, 0, 0
1, 0, 0
1, 0, 0
0, 0, 1
0, 0, 1
0, 0, 1
1, 0, 0
1, 0, 0
0, 0, 1
1, 0, 0
1, 0, 0
0, 0, 1
1, 0, 0
1, 0, 0
0, 0, 1
0, 0, 1
0, 0, 1
1, 0, 0
0, 0, 1
1, 0, 0
1, 0, 0
0, 0, 1
0, 0, 1
1, 0, 0
0, 1, 0
1, 0, 0
1, 0, 0
0, 0, 1
1, 0, 0
0, 0, 1
1, 0, 0
0, 1, 0
0, 0, 1
1, 0, 0
0, 0, 1
0, 0, 1
0, 0, 1
1, 0, 0
1, 0, 0
1, 0, 0
1, 0, 0
0, 0, 1
0, 0, 1
0, 0, 1
0, 0, 1
1, 0, 0
0, 1, 0
1, 0, 0
0, 1, 0
1, 0, 0
0, 0, 1
1, 0, 0
0, 1, 0
0, 0, 1
1, 0, 0
1, 0, 0
0, 1, 0
0, 0, 1
0, 1, 0
1, 0, 0
0, 0, 1
1, 0, 0
0, 0, 1
0, 1, 0
0, 0, 1
1, 0, 0
0, 0, 1
0, 0, 1
0, 0, 1
0, 0, 1
0, 0, 1
0, 0, 1
1, 0, 0
0, 1, 0
0, 0, 1
0, 1, 0
0, 0, 1
1, 0, 0
1, 0, 0
0, 0, 1
0, 0, 1
1, 0, 0
1, 0, 0
1, 0, 0
1, 0, 0
0, 0, 1
1, 0, 0
1, 0, 0
0, 1, 0
0, 1, 0
1, 0, 0
0, 0, 1
0, 0, 1
1, 0, 0
0, 0, 1
0, 0, 1
0, 0, 1
1, 0, 0
0, 0, 1
0, 1, 0
1, 0, 0
1, 0, 0
1, 0, 0
0, 1, 0
0, 0, 1
0, 1, 0
1, 0, 0
1, 0, 0
0, 1, 0
0, 0, 1
0, 0, 1
1, 0, 0
0, 0, 1
0, 0, 1
1, 0, 0
0, 0, 1
1, 0, 0
1, 0, 0
1, 0, 0
0, 1, 0
1, 0, 0
1, 0, 0
1, 0, 0
1, 0, 0
0, 0, 1
1, 0, 0
1, 0, 0
0, 1, 0
1, 0, 0
1, 0, 0
1, 0, 0
0, 1, 0
0, 0, 1
0, 0, 1
0, 0, 1
1, 0, 0
0, 0, 1
1, 0, 0
1, 0, 0
1, 0, 0
0, 0, 1
0, 0, 1
0, 0, 1
1, 0, 0
0, 0, 1
1, 0, 0
1, 0, 0
0, 0, 1
0, 0, 1
1, 0, 0
1, 0, 0
0, 0, 1
1, 0, 0
1, 0, 0
1, 0, 0
0, 1, 0
0, 1, 0
0, 0, 1
0, 0, 1
1, 0, 0
1, 0, 0
1, 0, 0
1, 0, 0
1, 0, 0
1, 0, 0
1, 0, 0
0, 0, 1
0, 1, 0
0, 0, 1
0, 1, 0
1, 0, 0
0, 1, 0
1, 0, 0
0, 1, 0
1, 0, 0
0, 0, 1
1, 0, 0
1, 0, 0
1, 0, 0
1, 0, 0
1, 0, 0
1, 0, 0
1, 0, 0
0, 1, 0
0, 0, 1
0, 1, 0
0, 1, 0
1, 0, 0
0, 0, 1
0, 0, 1
1, 0, 0
0, 0, 1
0, 0, 1
1, 0, 0
0, 1, 0
1, 0, 0
1, 0, 0
1, 0, 0
0, 0, 1
1, 0, 0
1, 0, 0
0, 0, 1
1, 0, 0
0, 0, 1
0, 1, 0
0, 0, 1
0, 0, 1
0, 0, 1
1, 0, 0
1, 0, 0
0, 0, 1
1, 0, 0
1, 0, 0
0, 0, 1
0, 0, 1
1, 0, 0
1, 0, 0
1, 0, 0
0, 1, 0
0, 0, 1
0, 0, 1
0, 0, 1
1, 0, 0
0, 1, 0
1, 0, 0
0, 0, 1
1, 0, 0
1, 0, 0
1, 0, 0
1, 0, 0
0, 1, 0
1, 0, 0
1, 0, 0
0, 1, 0
1, 0, 0
0, 0, 1
0, 0, 1
0, 1, 0
0, 0, 1
1, 0, 0
0, 0, 1
0, 0, 1
1, 0, 0
1, 0, 0
0, 0, 1
0, 0, 1
1, 0, 0
1, 0, 0
1, 0, 0
1, 0, 0
1, 0, 0
0, 1, 0
1, 0, 0
0, 0, 1
1, 0, 0
1, 0, 0
1, 0, 0
0, 0, 1
0, 1, 0
0, 0, 1
1, 0, 0
0, 0, 1
0, 0, 1
0, 0, 1
0, 1, 0
0, 1, 0
1, 0, 0
1, 0, 0
1, 0, 0
0, 0, 1
0, 0, 1
0, 0, 1
0, 0, 1
0, 1, 0
1, 0, 0
1, 0, 0
0, 1, 0
0, 0, 1
0, 1, 0
1, 0, 0
0, 0, 1
0, 1, 0
1, 0, 0
0, 0, 1
1, 0, 0
1, 0, 0
1, 0, 0
0, 0, 1
1, 0, 0
1, 0, 0
1, 0, 0
1, 0, 0
1, 0, 0
1, 0, 0
0, 0, 1
0, 0, 1
1, 0, 0
0, 1, 0
0, 0, 1
1, 0, 0
0, 0, 1
1, 0, 0
0, 1, 0
0, 0, 1
0, 0, 1
0, 0, 1
1, 0, 0
0, 0, 1
0, 1, 0
];

zhat = (D1*Xt').';
yhat = cReLU(zhat);
yhat = (D2*yhat.').';
Yhat = abs(yhat).^2;
Yhat = Yhat(:, 1:3);
[~, idx] = max(Yhat');
[~,  gt] = max(Yt');
accuracy = sum(gt == idx)/length(Xt);
fprintf('Accuracy = %.2f%%\n', accuracy*100)


function yhat = cReLU(zhat)
re = real(zhat);
im = imag(zhat);

re(re < 0) = 0;
im(im < 0) = 0;

yhat = re + 1j*im;
end

