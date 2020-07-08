''' phase_uncert_thetar simulating Optical Neural Network

using Neuroptica and linearly separable datasets
Now goes over every topology types with N = 4-32

Author: Simon Geoffroy-Gagnon
Edit: 2020.07.07
'''
import numpy as np
import calculate_accuracy as calc_acc
import cmath
import ONN_Simulation_Class as ONN_Cls
import ONN_Setups
import acc_colormap
import training_onn as train
import test_trained_onns as test
import create_datasets
import sys
from sklearn import preprocessing
sys.path.append('../')
import neuroptica as neu
import cmath
import copy

def normalize_inputs(data, num_inputs, P0=10):
    '''
    Reshapes the inputs to fit into the specified mesh size and normalizes input data to
    have the same total power input by injecting extra power to an "unused" input port.
    :param X: the input data
    :param num_inputs: the size of the network (number of waveguides)
    :param P0: the total power to inject with each data input
    '''
    _, input_size = data.shape
    injection_port = input_size
    data = (data - np.min(data))/(np.max(data) - np.min(data))
    data_normalized = np.array(np.pad(data, ((0, 0), (0, num_inputs - input_size)), mode="constant"))
    for i, x in enumerate(data_normalized):
        data_normalized[i][injection_port] = np.sqrt(P0 - np.sum(x**2))
    return data_normalized

def L2_norm(data):
    polar_data = [[cmath.polar(feature) for feature in sample] for sample in data]
    r = np.array([[feature[0] for feature in sample] for sample in polar_data])
    phi = np.array([[feature[1] for feature in sample] for sample in polar_data])
    r_nor = preprocessing.normalize(r, norm='l2')
    data_nor = []
    for idx, _ in enumerate(r):
        data_nor += [list(zip(r_nor[idx], phi[idx]))]
    # print(data_nor[1])
    data_nor = [[cmath.rect(r, phi) for r, phi in sample] for sample in data_nor]

    return np.array(data_nor)

onn = ONN_Cls.ONN_Simulation() # Required for containing training/simulation information
onn.BATCH_SIZE = 2**6 
onn.EPOCHS = 100
onn.STEP_SIZE = 0.0005 # Learning Rate

onn.ITERATIONS = 2 # number of times to retry same loss/PhaseUncert
rng_og = 1 # starting RNG value
max_number_of_tests = 5 # Max number of retries for a single model's training (keeps maximum accuracy model)
max_accuracy_req = 50 # (%) Will stop retrying after accuracy above this is reached

features = 17 # How many features? max for MNIST = 784 # Add +1 if using normalize_input()
classes = 10 # How many classes? max for MNIST = 10

eo_settings = {'alpha': 0.1, 'g':0.5 * np.pi, 'phi_b': -1 * np.pi} # If Electro-Optic Nonlinear Activation is used

# If you want regular Clements (multi-layer) topology
model = neu.Sequential([
    neu.ClementsLayer(features),
    neu.Activation(neu.cReLU(features)),
    neu.ClementsLayer(features),
    neu.Activation(neu.AbsSquared(features)), # photodetector measurement
    neu.DropMask(features, keep_ports=range(classes))
])

# If you want multi-layer Diamond Topology
# model = neu.Sequential([
#     neu.AddMaskDiamond(features),
#     neu.DiamondLayer(features),
#     neu.DropMask(2*features - 2, keep_ports=range(features - 2, 2*features - 2)), # Bottom Diamond Topology
#     neu.Activation(neu.cReLU(features)),
#     neu.AddMaskDiamond(features),
#     neu.DiamondLayer(features),
#     neu.DropMask(2*features - 2, keep_ports=range(features - 2, 2*features - 2)), # Bottom Diamond Topology
#     neu.Activation(neu.AbsSquared(features)), # photodetector measurement
#     neu.DropMask(features, keep_ports=range(classes),
# ])

# dataset = 'Gauss'
dataset = 'MNIST'

for onn.N in [features]:
    loss_diff = [0] # If loss_diff is used in insertion loss/MZI
    loss_var = [0] # if loss_var is used in insertion loss/MZI

    for ld in loss_diff:
        for lt in loss_var:
            rng = rng_og
            np.random.seed(rng)
            if dataset == 'Gauss':
                onn, _ = train.get_dataset(onn, rng, SAMPLES=40, EPOCHS=60, extra_channels=1)
                onn.X = normalize_inputs(onn.X, onn.N)
                onn.Xt = normalize_inputs(onn.Xt, onn.N)
            elif dataset == 'MNIST':
                # onn.X, onn.y, onn.Xt, onn.yt = create_datasets.MNIST_dataset(classes=classes, features=features, nsamples=200) # this gives real valued vectors as input samples 
                onn.X, onn.y, onn.Xt, onn.yt = create_datasets.FFT_MNIST(half_square_length=2, nsamples=40) # this gives complex valued vectors
                # onn.X, onn.y, onn.Xt, onn.yt = create_datasets.FFT_MNIST_PCA(classes=classes, features=features, nsamples=40) # this gives real valued vectors as input samples

                # To Add an extra channel and normalize power #
                onn.X = normalize_inputs(onn.X, onn.N)
                onn.Xt = normalize_inputs(onn.Xt, onn.N)
                
                # To Simply Normalize input power from 0-1 #
                # onn.X = (onn.X - np.min(onn.X))/(np.max(onn.X) - np.min(onn.X))
                # onn.Xt = (onn.Xt - np.min(onn.Xt))/(np.max(onn.Xt) - np.min(onn.Xt))

            onn.FOLDER = f'Analysis/FFT_MNIST/bs={onn.BATCH_SIZE}/N={onn.N}' # Name the folder to be created
            onn.createFOLDER() # Creates folder to save this ONN training and simulation info
            onn.saveSimDataset() # save the simulation datasets

            max_acc = 0
            onn.loss_diff = ld
            onn.loss_dB = [lt]
            for onn.rng in range(max_number_of_tests):
                onn.phases = []
                
                # Reset the phases to create new model
                current_phases = model.get_all_phases()
                current_phases = [[(None, None) for _ in layer] for layer in current_phases]
                model.set_all_phases_uncerts_losses(current_phases)

                # model = ONN_Setups.ONN_creation(onn) # If ONN_Setups is used

                onn, model = train.train_single_onn(onn, model, loss_function='cce')

                if max(onn.val_accuracy) > max_acc:
                    best_model = model
                    max_acc = max(onn.val_accuracy) 

                if max(onn.val_accuracy) > max_accuracy_req or onn.rng == max_number_of_tests-1:
                    onn.loss_diff = ld # Set loss_diff
                    onn.loss_dB = np.linspace(0, 3, 3) # set loss/MZI range
                    onn.phase_uncert_theta = np.linspace(0., 0.75, 3) # set theta phase uncert range
                    onn.phase_uncert_phi = np.linspace(0., 0.75, 3) # set phi phase uncert range
                    test.test_PT(onn, best_model, show_progress=True) # test Phi Theta phase uncertainty accurracy
                    test.test_LPU(onn, best_model, show_progress=True) # test Loss/MZI + Phase uncert accuracy
                    onn.saveAll(best_model) # Save best model information
                    onn.plotAll() # plot training and tests
                    onn.pickle_save() # save pickled version of the onn class
                    break


