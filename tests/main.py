''' Main function for simulating Optical Neural Network
using Neuroptica

Author: Simon Geoffroy-Gagnon
Edit: 18.02.2020
'''
import numpy as np
import ONN_Simulation_Class as ONN_Cls
import onnClassTraining
import calculate_accuracy as calc_acc

ONN = ONN_Cls.ONN_Simulation()
ONN.N = 64
ONN.BATCH_SIZE = 2**6
ONN.EPOCHS = 100
ONN.STEP_SIZE = 0.0005

ONN.SAMPLES = 6400
ONN.ITERATIONS = 20 # number of times to retry same loss/PhaseUncert
ONN.loss_diff = 0 # \sigma dB
ONN.loss_dB = np.linspace(0, 0, 1)
ONN.phase_uncert_theta = np.linspace(0., 2.5, 1)
ONN.phase_uncert_phi = np.linspace(0., 2.5, 1)
ONN.same_phase_uncert = False
ONN.rng = 2
ONN.zeta = 0

# ONN.dataset_name = 'MNIST'
ONN.dataset_name = 'Gaussian'
# ONN.dataset_name = 'Iris'

ONN_setup = np.array(['R_P'])
for setup in ONN_setup: 
    ONN.onn_topo = setup
    ONN.get_topology_name()
    model, ONN = onnClassTraining.train_single_onn(ONN)
    ONN.accuracy = calc_acc.get_accuracy(ONN, model, ONN.Xt, ONN.yt, loss_diff=ONN.loss_diff, zeta=ONN.zeta)
    onnClassTraining.create_folder(ONN)
    ONN.saveAll(model)
    # onnClassTraining.retrain_ONN(ONN, range(1))

'''
For MNIST PCA, [2, 4, 5, 6] w/ R_I_P = 88%
For MNIST PCA, [0, 1, 2, 9] w/ R_I_P = 87%
For MNIST PCA, [1, 3, 6, 7] w/ R_I_P = %
'''
