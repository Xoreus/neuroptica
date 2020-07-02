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

ONN = ONN_Cls.ONN_Simulation()
ONN.BATCH_SIZE = 2**6
ONN.EPOCHS = 500
ONN.STEP_SIZE = 0.0005

ONN.ITERATIONS = 2 # number of times to retry same loss/PhaseUncert
rng_og = 16
max_rng = 5
onn_topo = ['E_P']

features = 19

classes = 10
eo_settings = {'alpha': 0.1,
               'g':     0.5 * np.pi,
               'phi_b': -1 * np.pi }

model = neu.Sequential([
    neu.ClementsLayer(features),
    neu.Activation(neu.ElectroOpticActivation(features, **eo_settings)),
    neu.ClementsLayer(features),
    neu.Activation(neu.AbsSquared(features)), # photodetector measurement
    neu.DropMask(features, keep_ports=range(classes))
])

# dataset = 'Gauss'
dataset = 'MNIST'

for ONN.N in [features]:
    loss_diff = [0]
    loss_var = [0]

    for ld in loss_diff:
        for lt in loss_var:
            rng = rng_og
            np.random.seed(rng)
            if dataset == 'Gauss':
                ONN, _ = train.get_dataset(ONN, rng, SAMPLES=40, EPOCHS=60, extra_channels=1)
                # ONN.X = normalize_inputs(ONN.X, ONN.N)
                ONN.Xt = normalize_inputs(ONN.Xt, ONN.N)
            elif dataset == 'MNIST':
                # ONN.X, ONN.y, ONN.Xt, ONN.yt = create_datasets.MNIST_dataset(classes=classes, features=ONN.N-1, nsamples=40)  
                ONN.X, ONN.y, ONN.Xt, ONN.yt = create_datasets.FFT_MNIST(N=2, nsamples=100)
                # ONN.X, ONN.y, ONN.Xt, ONN.yt = create_datasets.FFT_MNIST_PCA(classes=classes, features=ONN.N-1, nsamples=100)
                ONN.X = normalize_inputs(ONN.X, ONN.N)
                ONN.Xt = normalize_inputs(ONN.Xt, ONN.N)

#                 ONN.X = L2_norm(ONN.X)
#                 ONN.Xt = L2_norm(ONN.Xt)

            # X_norm = np.sum(np.abs(ONN.X[:,:-1])**2,axis=-1)**(1./2)
            # print(ONN.X[:,:-1])
            # print(X_norm)

            ONN.FOLDER = f'Analysis/FFT_MNIST/bs={ONN.BATCH_SIZE}/N={ONN.N}'
            ONN.createFOLDER()
            ONN.saveSimDataset()

            for ONN.topo in onn_topo:
                max_acc = 0
                ONN.loss_diff = ld
                ONN.loss_dB = [lt]
                ONN.get_topology_name()
                for ONN.rng in range(max_rng):
                    ONN.phases = []

                    # model = ONN_Setups.ONN_creation(ONN)
                    ONN, model = train.train_single_onn(ONN, model, loss_function='cce')

                    if max(ONN.val_accuracy) > max_acc:
                        best_model = model
                        max_acc = max(ONN.val_accuracy) 

                    if max(ONN.val_accuracy) > 0 or ONN.rng == max_rng-1:
                        ONN.loss_diff = ld
                        ONN.loss_dB = np.linspace(0, 3, 3)
                        ONN.phase_uncert_theta = np.linspace(0., 0.75, 3)
                        ONN.phase_uncert_phi = np.linspace(0., 0.75, 3)
                        test.test_PT(ONN, best_model)
                        test.test_LPU(ONN, best_model)
                        ONN.saveAll(best_model)
                        ONN.plotAll()
                        ONN.pickle_save()
                        break


