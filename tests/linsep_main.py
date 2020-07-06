''' phase_uncert_thetar simulating Optical Neural Network

using Neuroptica and linearly separable datasets
Now goes over every topology types with N = 4-32

Author: Simon Geoffroy-Gagnon
Edit: 2020.03.28
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

onn = ONN_Cls.ONN_Simulation()
onn.BATCH_SIZE = 2**0
onn.EPOCHS = 300
onn.STEP_SIZE = 0.0005

onn.ITERATIONS = 2 # number of times to retry same loss/PhaseUncert
rng_og = 16
number_of_tests = 5
onn_topo = ['C_Q_P']

features = 7
classes = 4

eo_settings = {'alpha': 0.1,
               'g':     0.5 * np.pi,
               'phi_b': -1 * np.pi }

# model = neu.Sequential([
#     neu.ClementsLayer(features),
#     neu.Activation(neu.cReLU(features)),
#     neu.ClementsLayer(features),
#     neu.Activation(neu.AbsSquared(features)), # photodetector measurement
#     neu.DropMask(features, keep_ports=range(classes))
# ])

model = neu.Sequential([
    neu.AddMaskDiamond(features),
    neu.DiamondLayer(features, include_phase_shifter_layer=False),
    neu.DropMask(2*features - 2, keep_ports=range(features - 2, 2*features - 2)), # Bottom Diamond Topology
    neu.Activation(neu.cReLU(features)),
    neu.AddMaskDiamond(features),
    neu.DiamondLayer(features, include_phase_shifter_layer=False),
    neu.DropMask(2*features - 2, keep_ports=range(features - 2, 2*features - 2)), # Bottom Diamond Topology
    neu.Activation(neu.AbsSquared(features)), # photodetector measurement
    neu.DropMask(features, keep_ports=range(classes))
])

# dataset = 'Gauss'
dataset = 'MNIST'

for onn.N in [features]:
    loss_diff = [0]
    loss_var = [0]

    for ld in loss_diff:
        for lt in loss_var:
            rng = rng_og
            np.random.seed(rng)
            if dataset == 'Gauss':
                onn, _ = train.get_dataset(onn, rng, SAMPLES=40, EPOCHS=60, extra_channels=1)
                onn.X = normalize_inputs(onn.X, onn.N)
                onn.Xt = normalize_inputs(onn.Xt, onn.N)
            elif dataset == 'MNIST':
                onn.X, onn.y, onn.Xt, onn.yt = create_datasets.MNIST_dataset(classes=classes, features=onn.N, nsamples=40)  

                # onn.X, onn.y, onn.Xt, onn.yt = create_datasets.FFT_MNIST(N=2, nsamples=300)
                # onn.X, onn.y, onn.Xt, onn.yt = create_datasets.FFT_MNIST_PCA(classes=classes, features=onn.N, nsamples=100)
                # onn.X = normalize_inputs(onn.X, onn.N)
                # onn.Xt = normalize_inputs(onn.Xt, onn.N)

            onn.FOLDER = f'Analysis/FFT_MNIST/bs={onn.BATCH_SIZE}/N={onn.N}'
            onn.createFOLDER()
            onn.saveSimDataset()

            for onn.topo in onn_topo:
                max_acc = 0
                onn.loss_diff = ld
                onn.loss_dB = [lt]
                onn.get_topology_name()
                for onn.rng in range(number_of_tests):
                    onn.phases = []
                    # model = ONN_Setups.ONN_creation(onn) # If ONN_Setups is used
                    onn, model = train.train_single_onn(onn, model, loss_function='cce')

                    if max(onn.val_accuracy) > max_acc:
                        best_model = model
                        max_acc = max(onn.val_accuracy) 

                    if max(onn.val_accuracy) > 0 or onn.rng == max_rng-1:
                        onn.loss_diff = ld
                        onn.loss_dB = np.linspace(0, 3, 3)
                        onn.phase_uncert_theta = np.linspace(0., 0.75, 3)
                        onn.phase_uncert_phi = np.linspace(0., 0.75, 3)
                        test.test_PT(onn, best_model, show_progress=True)
                        test.test_LPU(onn, best_model, show_progress=True)
                        onn.saveAll(best_model)
                        onn.plotAll()
                        onn.pickle_save()
                        break


