''' phase_uncert_thetar simulating Optical Neural Network

Using Neuroptica and linearly separable datasets or MNIST

Author: Simon Geoffroy-Gagnon
Edit: 2020.09.04

Edit: 2022.10.13 by Bokun Zhao (bokun.zhao@mail.mcgill.ca)
'''
#%%
from math import floor
import numpy as np
from sklearn.preprocessing import MinMaxScaler as mms
import ONN_Simulation_Class as ONN_Cls
from plot_scatter_matrix import plot_scatter_matrix
import ONN_Setups
import training_onn as train
import test_trained_onns as test
import create_datasets
from sklearn import preprocessing
import sys
sys.path.append('../')
import neuroptica as neu

def see_each_mzi(p_model):
    '''
    Helper Function to visualize each MZI's (sigma_theta, sigma_phi, loss)

    sigma is the standard deviation of the normal distribution from which the phase error is drawn
    see the method "get_transfer_matrix()" in components.py
    '''
    print("\n------------------------------------------------------------------------------------------------")
    MZImesh = [layer for layer in p_model.layers if isinstance(layer, neu.OpticalMeshNetworkLayer)]
    print(f"There are {len(MZImesh)} MZImesh(es) in the model.")
    min_theta = np.inf
    max_theta = -np.inf
    min_phi = np.inf
    max_phi = -np.inf
    for i in range(len(MZImesh)):
        mzi_layers = MZImesh[i].mesh.layers # list of objects <MZILayer>
        print(f"\t In MZImesh {i}, there are {len(mzi_layers)} MZILayers (columns)")
        for col_idx, eachMZILayer in enumerate(mzi_layers):
            for row_idx, eachMZI in enumerate(eachMZILayer.mzis):
                # you can also print other information related to each MZI here, such as the theta/phi phases...
                mzi_info = f" [{eachMZI.m}____{eachMZI.n}] "
                # mzi_info = f" [{eachMZI.theta}____{eachMZI.phi}] "
                stagger = " "*(len(mzi_info)//2) # to better visualize the zig-zag alignment between each MZI column
                if row_idx==0: # at the start of each MZI column
                    print(stagger*eachMZI.m,end="") # amount of visual offset to print
                else:
                    print(stagger*(eachMZI.m-previousMZI_n-1),end="")
                print(mzi_info, end="")
                if eachMZI.theta < min_theta: min_theta = eachMZI.theta
                if eachMZI.phi < min_phi: min_phi = eachMZI.phi
                if eachMZI.theta > max_theta: max_theta = eachMZI.theta
                if eachMZI.phi > max_phi: max_phi = eachMZI.phi
                previousMZI_n = eachMZI.n
            print("\n")
    print(f"theta range: ({min_theta},{max_theta})")
    print(f"phi range: ({min_phi},{max_phi})")
    print("------------------------------------------------------------------------------------------------")

def sigma_adjust(p_model):
    '''
    if you want to individually tune certain MZIs' sigma values,
    you can use this method.
    '''
    for layer in p_model.layers:
        if isinstance(layer, neu.OpticalMeshNetworkLayer):
            # adjust sigma for individual mzis in a mesh
            # e.g. set the sigma_theta of the MZI at column 0, row 1 to be 0.5 rad
            layer.mesh.layers[0].mzis[1].phase_uncert_theta = 0.5
            # e.g. halve the sigma_theta of the MZI at column 3, row 0
            layer.mesh.layers[3].mzis[0].phase_uncert_theta /= 2
            # e.g. set the sigma_phi of the MZI at column 8, row 2 to be 2/3 of the old value
            layer.mesh.layers[8].mzis[2].phase_uncert_phi *= (2/3)

topo_index = 7

def init_onn_settings():
    ''' Initialize onn settings for training, testing and simulation '''
    onn = ONN_Cls.ONN_Simulation() # Required for containing training/simulation information

    onn.BATCH_SIZE = 50 # # of input samples per batch
    onn.EPOCHS = 50 # Epochs for ONN training
    onn.STEP_SIZE= 0.005 # Learning Rate
    onn.SAMPLES = 6000 # # of samples per class

    onn.ITERATIONS = 20 # number of times to retry same loss/PhaseUncert
    onn.rng = 50 # starting RNG value
    onn.max_number_of_tests = 1 # Max number of retries for a single model's training (keeps maximum accuracy model)
    onn.max_accuracy_req = 99.9 # Will stop retrying after accuracy above this is reached

    onn.features = 10 # How many features? max for MNIST = 784 
    onn.classes = 2 # How many classes? max for MNIST = 10
    onn.N = onn.features # number of ports in device

    onn.zeta = 0.75 * 100 # Min diff between max (correct) sample and second sample
    onn.beta = 1.0 # factor to penalize false negative: 1.0, 1.2, 1.4, 1.6, 1.8. 2.0

    # TO SCALE THE FIELD SUCH THAT POWER IS WITHIN A RANGE OF dB #
    # it is important to note that the ONN takes in FIELD, not POWER #
    # As such, we scale it to the sqrt() of the dB power #
    onn.MinMaxScaling = (0.5623, 1.7783) # For power = [-5 dB, +5 dB]
    onn.MinMaxScaling = (np.sqrt(0.1), np.sqrt(10)) # For power = [-10 dB, +10 dB]
    onn.range_linear = 10/onn.N # 10 mW total input power distributed to N input ports

    onn.topo = f'Dataset_Topology' # Name of the model, "{Dataset}_{Topology}"

    return onn

def dataset(onn, dataset='MNIST', half_square_length=2):
    ''' Create a dataset for training/testing the ONN
        Choices: Gauss, for a multivariate gaussian dataset
             MNIST, for a PCE'ed MNIST dataset
             FFT MNIST, for the central square of the FFT'ed MNIST
             FFT+PCA,  for the FFT'ed MNIST with PCA'''
    if dataset == 'Gauss':
        onn, onn.rng = train.get_dataset(onn, onn.rng, SAMPLES=onn.SAMPLES, EPOCHS=60, linear_sep_acc_limit=95) # EPOCHS here refers to the number of epochs for digital NN to see if linearly separable
    elif dataset == 'MNIST':
        onn.X, onn.y, onn.Xt, onn.yt, onn.X_test, onn.y_test = create_datasets.MNIST_dataset(classes=onn.classes, features=onn.features, nsamples=onn.SAMPLES) # this gives real valued vectors as input samples 
    elif dataset == 'FFT_MNIST':
        onn.X, onn.y, onn.Xt, onn.yt = create_datasets.FFT_MNIST(half_square_length=half_square_length, nsamples=onn.SAMPLES) # this gives complex valued vectors
        onn.features = (2*half_square_length)**2
        onn.N = (2*half_square_length)**2
    elif dataset == 'FFT_PCA':
        onn.X, onn.y, onn.Xt, onn.yt = create_datasets.FFT_MNIST_PCA(classes=onn.classes, features=onn.features, nsamples=onn.SAMPLES) # this gives real valued vectors as input samples
    elif dataset == 'Iris': # Gives real valued vectors, 4 features 3 classes
        onn.X, onn.y, onn.Xt, onn.yt = create_datasets.iris_dataset(nsamples=onn.SAMPLES)
        onn.classes = 3
        onn.features = 4
        onn.N = 4
    elif dataset == 'Iris_augment':# Gives real valued vectors, 4 features 4 classes
        onn.X, onn.y, onn.Xt, onn.yt = create_datasets.iris_dataset_augment(divide_mean=1.25, nsamples=onn.SAMPLES)
        onn.classes = 4
        onn.features = 4
        onn.N = 4
    elif dataset == 'WIDER':
        onn.X, onn.y, onn.Xt, onn.yt, onn.X_test, onn.y_test = create_datasets.WIDER_FACE(classes=onn.classes, features=onn.features, nsamples=onn.SAMPLES) # this gives real valued vectors as input samples 
    elif dataset == 'CIFAR-10':
        onn.X, onn.y, onn.Xt, onn.yt, onn.X_test, onn.y_test = create_datasets.CIFAR_10(classes=onn.classes, features=onn.features, nsamples=onn.SAMPLES) # this gives real valued vectors as input samples 
    else: 
        print("\nDataset not understood. Use 'Gauss', 'MNIST', 'FFT_MNIST', 'FFT_PCA', 'Iris', or 'Iris_augment'.\n")
    return onn

def normalize_dataset(onn, normalization='MinMaxScaling', experimental=False):
    ''' Constant_Power: Sends extra power to extra channel
        Normalize_Power: Normalize input power --> [0, onn.range_linear]
        MinMaxScaling: Normalize input power to [Min, Max]
    '''
    print(f'Original Dataset range: [{np.min(onn.X):.3f}, {np.max(onn.X):.3f}], [{np.min(onn.Xt):.3f}, {np.max(onn.Xt):.3f}], [{np.min(onn.X_test):.3f}, {np.max(onn.X_test):.3f}]')
    if normalization == 'Absolute':
        onn.X = np.abs(onn.X)
        onn.Xt = np.abs(onn.Xt)

    if normalization == 'MinMaxScaling':
        # print(f"Min Max Scaling with range: [{onn.MinMaxScaling[0]:.3f}, {onn.MinMaxScaling[1]:.3f}].")

        # Testing purposes
        # onn.MinMaxScaling = (np.min(onn.X), np.max(onn.X))

        scaler = mms(feature_range=onn.MinMaxScaling)
        onn.X = scaler.fit_transform(onn.X)
        onn.Xt = scaler.fit_transform(onn.Xt)

    if normalization == 'Constant_Power':
        print("Going with 1 extra channel and normalizing power")
        # add an extra channel (+1 ports) and normalize power #
        onn.N += 1

        onn.features += 1
        onn.X = normalize_inputs(onn.X, onn.N)
        # print(onn.X)
        onn.Xt = normalize_inputs(onn.Xt, onn.N)
        print(f"Number of channels: {onn.features}")
    elif normalization == 'Normalized':              
        # To Simply Normalize input power from [0-1]*onn.range_linear #
        if not experimental:
            X = (onn.X - np.min(onn.X))/(np.max(onn.X) - np.min(onn.X))*onn.range_linear
            Xt = (onn.Xt - np.min(onn.Xt))/(np.max(onn.Xt) - np.min(onn.Xt))*onn.range_linear
            X_test = (onn.X_test - np.min(onn.X_test))/(np.max(onn.X_test) - np.min(onn.X_test))*onn.range_linear
            # the above three lines are power, convert to amplitude by square root
            # (should be beneficial, as VOA attenuates less power)
            onn.X = np.sqrt(X)
            onn.Xt = np.sqrt(Xt)
            onn.X_test = np.sqrt(X_test)

        # To shift them to useable experimental samples...
        onn.range_dB = 10
        onn.dB_shift = 0.6
        if experimental:
            # change it to dBs
            onn.X = 10*np.log10(np.abs(onn.X)**2+sorted(set(np.abs(onn.X).reshape(-1)))[1])
            # Then normalize it ([0, 1] - shift)*range_dB
            onn.X = ((onn.X - np.min(onn.X))/(np.max(onn.X) - np.min(onn.X)) - onn.dB_shift)*onn.range_dB
            # Then reconvert it back to linear values
            onn.X = 10**(onn.X/10)
            onn.Xt = 10*np.log10(np.abs(onn.Xt)**2+sorted(set(np.abs(onn.Xt).reshape(-1)))[1])
            onn.Xt = ((onn.Xt - np.min(onn.Xt))/(np.max(onn.Xt) - np.min(onn.Xt)) - onn.dB_shift)*onn.range_dB
            onn.Xt = 10**(onn.Xt/10)
    elif normalization == 'Center':
        onn.X = (onn.X - np.min(onn.X))/(np.max(onn.X) - np.min(onn.X)) - 0.5
        onn.Xt = (onn.Xt - np.min(onn.Xt))/(np.max(onn.Xt) - np.min(onn.Xt)) - 0.5
    elif normalization == '10mW_total':
        # ensures all data sample encoding has a total of 10mW, but not realizable with attenuator-modulated input 
        onn.X = (onn.X - np.min(onn.X))/(np.max(onn.X) - np.min(onn.X))*onn.range_linear
        onn.Xt = (onn.Xt - np.min(onn.Xt))/(np.max(onn.Xt) - np.min(onn.Xt))*onn.range_linear
        prescale_vector_sums = np.sum(onn.X, axis=1)
        postscale_vector = onn.X / prescale_vector_sums[:, np.newaxis] * 10
        onn.X = np.sqrt(postscale_vector)
        prescale_vector_sums = np.sum(onn.Xt, axis=1)
        postscale_vector = onn.Xt / prescale_vector_sums[:, np.newaxis] * 10
        onn.Xt = np.sqrt(postscale_vector)
    print(f'Using {normalization} scaling, Dataset range: [{np.min(onn.X):.3f}, {np.max(onn.X):.3f}], [{np.min(onn.Xt):.3f}, {np.max(onn.Xt):.3f}], [{np.min(onn.X_test):.3f}, {np.max(onn.X_test):.3f}]')
    print(f'Power range: [{10*np.log10(np.min(onn.X)**2):.5f}, {10*np.log10(np.max(onn.X)**2):.5f}] dB, [{10*np.log10(np.min(onn.Xt)**2):.5f}, {10*np.log10(np.max(onn.Xt)**2):.5f}] dB, [{10*np.log10(np.min(onn.X_test)**2):.5f}, {10*np.log10(np.max(onn.X_test)**2):.5f}] dB')
    return onn

def normalize_inputs(data, num_inputs, P0=10):
    ''' Reshapes the inputs to fit into the specified mesh size and normalizes input data to
    have the same total power input by injecting extra power to an "unused" input port.
    :param X: the input data
    :param num_inputs: the size of the network (number of waveguides)
    :param P0: the total power to inject with each data input
    '''
    _, input_size = data.shape
    injection_port = input_size
    # data = (data - np.min(data))/(np.max(data) - np.min(data))*10
    data = [x**2 for x in data]
    P0 = max(np.sum([x for x in data], axis=1))
    # print(f"Power Required: {P0:.3f}, or {10*np.log10(P0):.3f} dB")
    data_normalized = np.array(np.pad(data, ((0, 0), (0, num_inputs - input_size)), mode="constant"))
    for i, x in enumerate(data_normalized):
        data_normalized[i][injection_port] = np.sqrt(P0 - np.sum(x))
    return data_normalized

def create_model(features, classes):
    ''' create ONN model based on neuroptica layer '''
    eo_settings = {'alpha': 0.1, 'g':0.5 * np.pi, 'phi_b': -1 * np.pi} # If Electro-Optic Nonlinear Activation is used

    # Some nonlinearities, to be used withing neu.Activation()
    eo_activation = neu.ElectroOpticActivation(features, **eo_settings)
    cReLU = neu.cReLU(features)
    zReLU = neu.zReLU(features)
    bpReLU = neu.bpReLU(features, cutoff=0.11, alpha=0.01)
    modReLU = neu.modReLU(features, cutoff=1)
    sigmoid = neu.Sigmoid(features)
    leakycReLU = neu.LeakycReLU(features, leaky_gradient=0.01)
    leakyReLU = neu.LeakyReLU(features, thres = 0.1, leaky_gradient=0.01)
    custom_sigmoid = neu.CustomSigmoid(features, k = 69.6) # 69.6

    nlaf = eo_activation # Pick the Non Linear Activation Function


    # If you want multi-layer BOTTOM Diamond Topology
    # model = neu.Sequential([
        # neu.AddMaskDiamond(features),
        # neu.DiamondLayer(features),
        # neu.DropMask(2*features - 2, keep_ports=range(features - 2, 2*features - 2)), # Bottom Diamond Topology
        # neu.Activation(nlaf), # first layer ends here
        # neu.AddMaskDiamond(features),
        # neu.DiamondLayer(features),
        # neu.DropMask(2*features - 2, keep_ports=range(features - 2, 2*features - 2)), # Bottom Diamond Topology
        # neu.Activation(neu.AbsSquared(features)), # photodetector measurement
        # neu.DropMask(features, keep_ports=range(classes)), # needs a drop mask at the output
    # ])

    # If you want multi-layer TOP Diamond Topology
    # model = neu.Sequential([
    #     # neu.AddMaskDiamond(features),
    #     # neu.DiamondLayer(features),
    #     # neu.DropMask(2*features - 2, keep_ports=range(0, features)), # Top Diamond Topology
    #     # neu.Activation(nlaf), # first layer ends here
    #     neu.AddMaskDiamond(features),
    #     neu.DiamondLayer(features),
    #     neu.DropMask(2*features - 2, keep_ports=range(0, features)), # Top Diamond Topology
    #     neu.Activation(neu.AbsSquared(features)), # photodetector measurement
    #     neu.DropMask(features, keep_ports=range(classes)),
    # ])

    # If you want multi-layer Bokun Topology
    # model = neu.Sequential([
    #     neu.AddMaskDiamond(features),
    #     neu.DiamondLayer(features),
    #     neu.DropMask(2*features - 2, keep_ports=range(features//2-1, floor(features*1.5)-1)), # Middle Diamond Topology
    #     neu.Activation(nlaf), # first layer ends here
    #     neu.AddMaskDiamond(features),
    #     neu.DiamondLayer(features),
    #     neu.DropMask(2*features - 2, keep_ports=range(features//2-1, floor(features*1.5)-1)), # Middle Diamond Topology
    #     neu.Activation(neu.AbsSquared(features)), # photodetector measurement
    #     neu.DropMask(features, keep_ports=range(classes)),
    # ])

    # If you want regular Clements (multi-layer) topology
    # model = neu.Sequential([
    #     neu.ClementsLayer(features),
    #     neu.Activation(nlaf),
    #     neu.ClementsLayer(features),
    #     neu.Activation(neu.AbsSquared(features)), # photodetector measurement
    #     neu.DropMask(features, keep_ports=range(classes))
    # ])

    # If you want regular Reck (single-layer) topology
    # model = neu.Sequential([
    #     # neu.ReckLayer(features),
    #     # neu.Activation(nlaf), # photodetector measurement
    #     neu.ReckLayer(features),
    #     # neu.Activation(neu.AbsSquared(features)), # photodetector measurement
    #     # neu.DropMask(features, keep_ports=range(classes)) # Drops the unwanted ports
    # ])

    # if using custom ONN layers, specify the mesh profile: a 2D matrix indicate MZI locations
    # Representation of profile: 2D lists with waveguide indices as entries
    # each pair of numbers in a row denote the MZI location
    # This representation is the most convenient to parse by MZILayer.from_waveguide_indices() method
    clement_4x4_profile = [[0, 1, 2, 3],
                           [   1, 2,  ],
                           [0, 1, 2, 3],
                           [   1, 2,  ]]
    reck_4x4_profile =    [[0, 1      ],
                           [   1, 2,  ],
                           [0, 1, 2, 3],
                           [   1, 2   ],
                           [0, 1      ]]
    Bokun_4x4_profile =   [[      2, 3      ],
                           [   1, 2, 3, 4   ],
                           [0, 1, 2, 3, 4, 5],
                           [   1, 2, 3, 4   ],
                           [      2, 3      ]]
    reck_10x10_profile =    [[0,1                ],
                             [  1,2              ],
                             [0,1,2,3            ],
                             [  1,2,3,4          ],
                             [0,1,2,3,4,5        ],
                             [  1,2,3,4,5,6      ],
                             [0,1,2,3,4,5,6,7    ],
                             [  1,2,3,4,5,6,7,8  ],
                             [0,1,2,3,4,5,6,7,8,9],
                             [  1,2,3,4,5,6,7,8  ],
                             [0,1,2,3,4,5,6,7    ],
                             [  1,2,3,4,5,6      ],
                             [0,1,2,3,4,5        ],
                             [  1,2,3,4          ],
                             [0,1,2,3            ],
                             [  1,2              ],
                             [0,1                ],]
    
    clement_10x10_profile = [[0,1,2,3,4,5,6,7,8,9],
                             [  1,2,3,4,5,6,7,8  ],
                             [0,1,2,3,4,5,6,7,8,9],
                             [  1,2,3,4,5,6,7,8  ],
                             [0,1,2,3,4,5,6,7,8,9],
                             [  1,2,3,4,5,6,7,8  ],
                             [0,1,2,3,4,5,6,7,8,9],
                             [  1,2,3,4,5,6,7,8  ],
                             [0,1,2,3,4,5,6,7,8,9],
                             [  1,2,3,4,5,6,7,8  ]]
    
    clement_10x10_withHole = [[0,1,2,3,4,5,6,7,8,9],
                             [  1,2,3,4,5,6,7,8  ],
                             [0,1,2,3,4,5,6,7,8,9],
                             [  1,2,3,4,5,6,7,8  ],
                             [0,1,2,3,4,5,6,7,8,9],
                             [  1,2,3,4,    7,8  ],
                             [0,1,2,3,4,5,6,7,8,9],
                             [  1,2,3,4,5,6,7,8  ],
                             [0,1,2,3,4,5,6,7,8,9],
                             [  1,2,3,4,5,6,7,8  ]]
    
    custom_10x10_profile  = [[0,1,2,3,4,5,6,7,8,9],
                             [  1,2,3,4,5,6,7,8  ],
                             [0,1,2,3,4,5,6,7,8,9],
                             [  1,2,3,4,5,6,7,8  ],
                             [0,1,2,3,4,5,6,7    ],
                             [  1,2,3,4,5,6      ],
                             [0,1,2,3,4,5        ],
                             [  1,2,3,4          ],
                             [0,1,2,3            ],
                             [  1,2              ]]
 
    drill6_10x10_profile =  [[0,1,2,3,4,5,6,7,8,9],
                             [  1,2,3,4,5,6,7,8  ],
                             [0,1,2,3,4,5,6,7,8,9],
                             [  1,2,3,4,5,6,7,8  ],
                             [0,1,2,3,4,5,6,7,8,9],
                             [  1,2,3,4,5,6,7,8  ],
                             [0,1,2,3,4,5,6,7,8,9],
                             [  1,2,3,4,5,6,7,8  ],
                             [    2,3,4,5,6,7    ],
                             [      3,4,5,6      ]]
    drill5_10x10_profile =  [[  1,2,3,4,5,6,7,8  ],
                             [0,1,2,3,4,5,6,7,8,9],
                             [  1,2,3,4,5,6,7,8  ],
                             [0,1,2,3,4,5,6,7,8,9],
                             [  1,2,3,4,5,6,7,8  ],
                             [0,1,2,3,4,5,6,7,8,9],
                             [  1,2,3,4,5,6,7,8  ],
                             [    2,3,4,5,6,7    ],
                             [      3,4,5,6      ]]
    drill4_10x10_profile =  [[0,1,2,3,4,5,6,7,8,9],
                             [  1,2,3,4,5,6,7,8  ],
                             [0,1,2,3,4,5,6,7,8,9],
                             [  1,2,3,4,5,6,7,8  ],
                             [0,1,2,3,4,5,6,7,8,9],
                             [  1,2,3,4,5,6,7,8  ],
                             [    2,3,4,5,6,7    ],
                             [      3,4,5,6      ]]
    drill3_10x10_profile =  [[  1,2,3,4,5,6,7,8  ],
                             [0,1,2,3,4,5,6,7,8,9],
                             [  1,2,3,4,5,6,7,8  ],
                             [0,1,2,3,4,5,6,7,8,9],
                             [  1,2,3,4,5,6,7,8  ],
                             [    2,3,4,5,6,7    ],
                             [      3,4,5,6      ]]
    drill2_10x10_profile =  [[0,1,2,3,4,5,6,7,8,9],
                             [  1,2,3,4,5,6,7,8  ],
                             [0,1,2,3,4,5,6,7,8,9],
                             [  1,2,3,4,5,6,7,8  ],
                             [    2,3,4,5,6,7    ],
                             [      3,4,5,6      ]]
    drill_10x10_profile =   [[  1,2,3,4,5,6,7,8  ],
                             [0,1,2,3,4,5,6,7,8,9],
                             [  1,2,3,4,5,6,7,8  ],
                             [    2,3,4,5,6,7    ],
                             [      3,4,5,6      ]]
    triangle_10x10_profile =[[0,1,2,3,4,5,6,7,8,9],
                             [  1,2,3,4,5,6,7,8  ],
                             [    2,3,4,5,6,7    ],
                             [      3,4,5,6      ]]
    miniBokun_10x10_profile=[[    2,3,4,5,6,7    ],
                             [  1,2,3,4,5,6,7,8  ],
                             [0,1,2,3,4,5,6,7,8,9],
                             [  1,2,3,4,5,6,7,8  ],
                             [    2,3,4,5,6,7    ],
                             [      3,4,5,6      ]]
    topology_to_test = [drill6_10x10_profile,
                        drill5_10x10_profile,
                        drill4_10x10_profile,
                        drill3_10x10_profile,
                        drill2_10x10_profile,
                        drill_10x10_profile,
                        triangle_10x10_profile,
                        miniBokun_10x10_profile]
    
    miniBokun_8x8_profile=[[    2,3,4,5    ],
                           [  1,2,3,4,5,6  ],
                           [0,1,2,3,4,5,6,7],
                           [  1,2,3,4,5,6  ],
                           [    2,3,4,5    ]]
    miniBokun_16x16_profile=[[    2,3,4,5,6,7,8,9,10,11,12,13      ],
                             [  1,2,3,4,5,6,7,8,9,10,11,12,13,14   ],
                             [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
                             [  1,2,3,4,5,6,7,8,9,10,11,12,13,14   ],
                             [    2,3,4,5,6,7,8,9,10,11,12,13      ],
                             [      3,4,5,6,7,8,9,10,11,12         ],
                             [        4,5,6,7,8,9,10,11            ],
                             [          5,6,7,8,9,10               ],
                             [            6,7,8,9                  ]]
    miniBokun_32x32_profile=[list(range(2,30)),
                             list(range(1,31)),
                             list(range(0,32)),
                             list(range(1,31)),
                             list(range(2,30)),
                             list(range(3,29)),
                             list(range(4,28)),
                             list(range(5,27)),
                             list(range(6,26)),
                             list(range(7,25)),
                             list(range(8,24)),
                             list(range(9,23)),
                             list(range(10,22)),
                             list(range(11,21)),
                             list(range(12,20)),
                             list(range(13,19)),
                             list(range(14,18))]
    miniBokun_64x64_profile = [list(range(i, features-i)) for i in range(features//2 - 1)]
    miniBokun_64x64_profile.insert(0, list(range(1, features-1)))
    miniBokun_64x64_profile.insert(0, list(range(2, features-2)))
# [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
    diamond_10x10_profile  = [[                8,9                        ],
                              [              7,8,9,10                     ],
                              [            6,7,8,9,10,11                  ],
                              [          5,6,7,8,9,10,11,12               ],
                              [        4,5,6,7,8,9,10,11,12,13            ],
                              [      3,4,5,6,7,8,9,10,11,12,13,14         ],
                              [    2,3,4,5,6,7,8,9,10,11,12,13,14,15      ],
                              [  1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16   ],
                              [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],
                              [  1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16   ],
                              [    2,3,4,5,6,7,8,9,10,11,12,13,14,15      ],
                              [      3,4,5,6,7,8,9,10,11,12,13,14         ],
                              [        4,5,6,7,8,9,10,11,12,13            ],
                              [          5,6,7,8,9,10,11,12               ],
                              [            6,7,8,9,10,11                  ],
                              [              7,8,9,10                     ],
                              [                8,9                        ]]

    bokun_10x10_profile    = [[        4,5,6,7,8,9,10,11,12,13            ],
                              [      3,4,5,6,7,8,9,10,11,12,13,14         ],
                              [    2,3,4,5,6,7,8,9,10,11,12,13,14,15      ],
                              [  1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16   ],
                              [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],
                              [  1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16   ],
                              [    2,3,4,5,6,7,8,9,10,11,12,13,14,15      ],
                              [      3,4,5,6,7,8,9,10,11,12,13,14         ],
                              [        4,5,6,7,8,9,10,11,12,13            ],
                              [          5,6,7,8,9,10,11,12               ]]

    bokun_pr_10x10_profile = [[        4,5,6,7,8,9,10,11,12,13            ],
                              [      3,4,5,6,7,8,9,10,11,12,13,14         ],
                              [    2,3,4,5,6,7,8,9,10,11,12,13            ],
                              [  1,2,3,4,5,6,7,8,9,10,11,12               ],
                              [0,1,2,3,4,5,6,7,8,9,10,11                  ],
                              [  1,2,3,4,5,6,7,8,9,10                     ],
                              [    2,3,4,5,6,7,8,9                        ],
                              [      3,4,5,6,7,8                          ],
                              [        4,5,6,7                            ],
                              [          5,6                              ]]


    model = neu.Sequential([
        # neu.AddMaskDiamond(features),
        neu.CustomLayer(features, topology_to_test[topo_index]),
        # neu.Activation(neu.AbsSquared(features)), # photodetector measurement
        # neu.Activation(nlaf), # non-linear activation layer
        # neu.CustomLayer(features, miniBokun_8x8_profile),
        # neu.ClementsLayer(features),
        # neu.DropMask(features, keep_ports=[4, 5]), # Drops the unwanted ports
        # neu.Activation(nlaf), # non-linear activation layer
        # neu.PadZeros(features),
        # neu.CustomLayer(features, clement_10x10_profile),
        # neu.CustomLayer(features, miniBokun_8x8_profile),
        # neu.ClementsLayer(features),
        neu.Activation(neu.AbsSquared(features)), # photodetector measurement
        # neu.DropMask(features, keep_ports=range(classes)) # Drops the unwanted ports
        neu.DropMask(features, keep_ports=[4, 5]) # Drops the unwanted ports
    ])

    # model = neu.Sequential([
    #     neu.AddMaskDiamond(features),
    #     neu.DiamondLayer(features),
    #     neu.DropMask(2*features - 2, keep_ports=range(features//2-1, floor(features*1.5)-1)), # Middle Diamond Topology
    #     neu.Activation(nlaf), # first layer ends here
    #     neu.AddMaskDiamond(features),
    #     neu.DiamondLayer(features),
    #     neu.DropMask(2*features - 2, keep_ports=range(features//2-1, floor(features*1.5)-1)), # Middle Diamond Topology
    #     neu.Activation(neu.AbsSquared(features)), # photodetector measurement
    #     neu.DropMask(features, keep_ports=range(classes)),
    # ])

    return model

def save_onn(onn, model, lossDiff=0):
    num_tst_inst = 8000 # use a subset of the test set to speed things up (MNIST: 10000; CIFAR: 8000)
    onn.loss_diff = lossDiff # Set loss_diff
    # For simulation purposes, defines range of loss and phase uncert
    onn.loss_dB = np.linspace(0., 1, 40) # set loss/MZI range
    onn.phase_uncert_theta = np.linspace(0, 0.6153846154, 25) # set theta phase uncert range
    onn.phase_uncert_phi = np.linspace(0, 0.6153846154, 25) # set phi phase uncert range

    onn, model = test.test_PT(onn, onn.X_test[:num_tst_inst], onn.y_test[:num_tst_inst], model, show_progress=True) # test Phi Theta phase uncertainty accurracy
    onn, model = test.test_LPU(onn, onn.X_test[:num_tst_inst], onn.y_test[:num_tst_inst], model, show_progress=True) # test Loss/MZI + Phase uncert accuracy
    onn.saveAll(model, cmap='hsv') # Save best model information
    # onn.plotAll() # plot training and tests
    onn.plotBackprop(backprop_legend_location=0)
    ''' Backprop Legend Location Codes:
    'best' 	        0
    'upper right' 	1
    'upper left' 	2
    'lower left' 	3
    'lower right' 	4
    'right'        	5
    'center left' 	6
    'center right' 	7
    'lower center' 	8
    'upper center' 	9
    'center'    	10
    '''
    onn.pickle_save() # save pickled version of the onn class


def compute_F1_score(_pred, _gt):
    # Calculate True Positives, False Positives, and False Negatives
    TP = np.sum((_pred == 1) & (_gt == 1))
    FP = np.sum((_pred == 1) & (_gt == 0))
    TN = np.sum((_pred == 0) & (_gt == 0))
    FN = np.sum((_pred == 0) & (_gt == 1))
    # Precision and Recall
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    # F1 Score
    if (precision + recall) > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0
    P_trigger = (FP + TP)/(TN + FP + FN + TP)
    return f1_score, FN, P_trigger
#%%
def main():
    onn = init_onn_settings()
    onn.topo = sys.argv[1] # name of the model
    onn.rng = int(sys.argv[2])
    print(f"name: {onn.topo}", f"onn.rng: {onn.rng}")
    np.random.seed(onn.rng)

    # model = create_model(onn.features, onn.classes)
    # print("Phases when creating the model: D_mzi =")
    # print(f"phases({np.shape(model.get_all_phases())}):{model.get_all_phases()}")
    # print("\nTransformation matrix when creating the model: D_mzi =")
    # print(f"phases({np.shape(model.get_transformation_matrix())}):\n{model.get_transformation_matrix()}")

    # see_each_mzi(model)
    # current_phases = model.get_all_phases()
    # current_phases = [[(1.34, 5.72) for _ in layer] for layer in current_phases]
    # model.set_all_phases_uncerts_losses(current_phases, phase_uncert_theta=0.0, phase_uncert_phi=0.0, loss_dB=0.0, loss_diff=0.0)
    # # print("Phases after updating the model: D_mzi =")
    # # print(f"phases({np.shape(model.get_all_phases())}):{model.get_all_phases()}")
    # # print("\nTransformation matrix after updating the model: D_mzi =")
    # # print(f"phases({np.shape(model.get_transformation_matrix())}):\n{model.get_transformation_matrix()}")
    # see_each_mzi(model)
    data = 'MNIST' # Choose one: Gauss, MNIST, FFT_MNIST, Iris_augment, Iris, CIFAR-10
    onn = dataset(onn, dataset=data)

    onn = normalize_dataset(onn, normalization='Normalized') # dataset -> [Min, Max]
    # exit(0)

    #======RETRAINING: after deciding some hyperparameters (layer # and N), retrain with validation data as part of training data
    # onn.X = np.vstack((onn.X, onn.Xt))
    # onn.y = np.vstack((onn.y, onn.yt))
    # onn.Xt = onn.X_test # change validation set to be same as test set
    # onn.yt = onn.y_test
    #======================================================

    # Feature Permutation: place important features at the center
    indices = np.arange(0, onn.features, dtype=np.int32)
    new_order = np.concatenate((indices[onn.features-2::-2],indices[1::2]), axis=0)
    print(f"new_order_index: {new_order}")
    # new_order = [14, 12, 10, 8, 6, 4, 2, 0, 1, 3, 5, 7, 9, 11, 13, 15]
    onn.X = onn.X[:, new_order]
    onn.Xt = onn.Xt[:, new_order]
    onn.X_test = onn.X_test[:, new_order]

    # make the label length N by padding additional zero (to test use all ports for binary classification)
    # onn.y = np.hstack((np.zeros((onn.y.shape[0],(onn.features-onn.classes)//2)),
    #                    onn.y,
    #                    np.zeros((onn.y.shape[0],(onn.features-onn.classes)//2))))
    # onn.yt = np.hstack((np.zeros((onn.yt.shape[0],(onn.features-onn.classes)//2)),
    #                     onn.yt,
    #                     np.zeros((onn.yt.shape[0],(onn.features-onn.classes)//2))))
    # onn.y_test = np.hstack((np.zeros((onn.y_test.shape[0],(onn.features-onn.classes)//2)),
    #                     onn.y_test,
    #                     np.zeros((onn.y_test.shape[0],(onn.features-onn.classes)//2))))

    # OR gradient labeling: [0.01,0.02,0.03,0.04,0.05,0.75,0.04,0.03,0.02,0.01]
    # a = np.arange(1, (onn.features-onn.classes)//2+1) # top/bottom pad thickness
    # top_pad = np.tile(0.01*a, (8000, 1))
    # bottom_pad = np.flip(top_pad, axis=1)
    # onn.y = np.hstack((top_pad[:onn.y.shape[0], :],
    #                    onn.y*0.70+0.05,
    #                    bottom_pad[:onn.y.shape[0], :]))
    # onn.yt = np.hstack((top_pad[:onn.yt.shape[0], :],
    #                     onn.yt*0.70+0.05,
    #                     bottom_pad[:onn.yt.shape[0], :]))
    # onn.y_test = np.hstack((top_pad[:onn.y_test.shape[0], :],
    #                     onn.y_test*0.70+0.05,
    #                     bottom_pad[:onn.y_test.shape[0], :]))

    print(f"{data} dataset prepared:")
    print(f"X: {np.shape(onn.X)}, range={onn.X.max(), onn.X.min()}")
    print(f"Y: {np.shape(onn.y)}, range={onn.y.max(), onn.y.min()}")
    print(f"Xt: {np.shape(onn.Xt)}, range={onn.Xt.max(), onn.Xt.min()}")
    print(f"Yt: {np.shape(onn.yt)}, range={onn.yt.max(), onn.yt.min()}")
    print(f"X_test: {np.shape(onn.X_test)}, range={onn.X_test.max(), onn.X_test.min()}")
    print(f"y_test: {np.shape(onn.y_test)}, range={onn.y_test.max(), onn.y_test.min()}")
    # print(f"example(first 5) X:\n{onn.X[0:5]}\npower sum: {np.sum(onn.X[0:5]**2, axis=1)}")
    # print(f"example(first 5) y:\n{onn.y[0:5]}")
    # print(f"example(first 5) X_test:\n{onn.X_test[0:5]}\npower sum: {np.sum(onn.X_test[0:5]**2, axis=1)}")
    # print(f"example(first 5) y_test:\n{onn.y_test[0:5]}")
    print(f"training label ratio:\n{np.sum(onn.y, axis=0)}")
    print(f"validation label ratio:\n{np.sum(onn.yt, axis=0)}")
    print(f"testing label ratio:\n{np.sum(onn.y_test, axis=0)}")

    # exit(0)
    model = create_model(onn.features, onn.classes)
    # see_each_mzi(model)
    # print("Phases when creating the model:")
    # print(f"phases({np.shape(model.get_all_phases())}):{model.get_all_phases()}")
    # print("\nTransformation matrix when creating the model: D_mzi =")
    # print(f"phases({np.shape(model.get_transformation_matrix())}):{model.get_transformation_matrix()}")
    # exit(0)
    loss_diff = [0] # If loss_diff is used in insertion loss/MZI
    training_loss = [0] # loss used during training

    for lossDiff in loss_diff:
        for trainLoss in training_loss:
            onn.FOLDER = f'Analysis/iris_augment/{onn.features}x{onn.classes}_{onn.topo}' # Name the folder to be created
            onn.createFOLDER() # Creates folder to save this ONN training and simulation info
            onn.saveSimDataset() # save the simulation datasets

            max_acc = 0 # Reset maximum accuracy achieved
            onn.loss_diff = lossDiff
            onn.loss_dB = [trainLoss]
            onn.get_topology_name()
            for test_number in range(onn.max_number_of_tests):
                onn.phases = [] # Reset Saved Phases
                
                # Reset the phases to create new model
                current_phases = model.get_all_phases()
                current_phases = [[(None, None) for _ in layer] for layer in current_phases]
                model.set_all_phases_uncerts_losses(current_phases, phase_uncert_theta=0.0, phase_uncert_phi=0.0, loss_dB=0, loss_diff=0.0)
                
                onn, model = train.train_single_onn(onn, model, loss_function='ccew') # 'cce' for categorical, 'mse' for Gaussian, 'ccew' for cce with FN reduction (beta parameter)

                if test_number>0:
                    print("\nPhase of current best model")
                    # print(best_model.get_all_phases())
                # # Save best model
                if max(onn.val_accuracy) > max_acc:
                    best_model = model
                    onn.model = model
                    best_onn = onn
                    max_acc = max(onn.val_accuracy) 
                    onn.plotBackprop(backprop_legend_location=0)
                    onn.pickle_save() # save pickled version of the onn class
                    current_phases = best_model.get_all_phases()
                    best_model.set_all_phases_uncerts_losses(current_phases, phase_uncert_theta=0.0, phase_uncert_phi=0.0, loss_dB=0.0, loss_diff=0.0)
                    # print("\nNew Best Model!")
                    # print(best_model.get_all_phases())

                if (max(onn.val_accuracy) > onn.max_accuracy_req or
                        test_number == onn.max_number_of_tests-1):
                    print("\nThis is the best model")
                    # print(best_model.get_all_phases())
                    print(f'\nBest Validation Accuracy: {max_acc:.2f}%.')
                    best_model.set_all_phases_uncerts_losses(onn.phases, phase_uncert_theta=0.0, phase_uncert_phi=0.0, loss_dB=0.00, loss_diff=0.0)
                    yhat = best_model.forward_pass(onn.X_test.T)
                    cls = np.array([np.argmax(yhat) for yhat in yhat.T])
                    gt = np.array([np.argmax(tru) for tru in onn.y_test]) # y_test: one-hot encoded vectors; gt: actual scalar labels
                    test_accuracy = sum(gt == cls)/len(onn.X_test)*100
                    f1_score, fn, p_trigger = compute_F1_score(cls, gt)
                    print(f'Best Testing Accuracy: {test_accuracy:.2f}%. Using this model for simulations.')
                    print(f'F1 score: {(f1_score*100):.3f}%.')
                    print(f'False Nagatives: {fn}.')
                    print(f'P(trigger): {(p_trigger):.5f}.')
                    # save_onn(best_onn, best_model) # uncomment to simulate and produce PT & LPU graphs
                    best_onn.saveForwardPropagation(best_model)
                    current_phases = best_model.get_all_phases()
                    best_model.set_all_phases_uncerts_losses(current_phases, phase_uncert_theta=0.0, phase_uncert_phi=0.0, loss_dB=0.00, loss_diff=0.0)
                    best_onn.save_correct_classified_samples(best_model)
                    best_onn.save_correct_classified_samples(best_model, zeta=onn.zeta)
                    best_onn.save_correct_classified_samples(best_model, zeta=2*onn.zeta)

                    # To plot scattermatrix of dataset
                    # axes = plot_scatter_matrix(onn.X, onn.y,  figsize=(15, 15), label='X', start_at=0, fontsz=54)
                    # plt.savefig(onn.FOLDER + '/scatterplot.pdf')
                    break

if __name__ == '__main__':
     main()

# %%
