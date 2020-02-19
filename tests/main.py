''' Main function for simulating Optical Neural Network
using Neuroptica

Author: Simon Geoffroy-Gagnon
Edit: 18.02.2020
'''
import sys
import numpy as np
import numpy
import ONN_Setups
import create_datasets as cd 
import setupSimulation as setSim
import calculate_accuracy as calc_acc
import ONN_Simulation_Class as ONN_Cls
import saveSimulationData as sSD
import digital_NN_main
sys.path.append('/home/simon/Documents/neuroptica')
import neuroptica as neu
import onnClassTraining

ONN = ONN_Cls.ONN_Simulation()
ONN.N = 8
ONN.BATCH_SIZE = 2**6
ONN.EPOCHS = 10
ONN.STEP_SIZE = 0.0005

ONN.SAMPLES = 300
ONN.ITERATIONS = 20 # number of times to retry same loss/PhaseUncert
ONN.loss_diff = 0 # \sigma dB
ONN.loss_dB = np.linspace(0, 0, 1)
ONN.phase_uncert_theta = np.linspace(0., 2.5, 11)
ONN.phase_uncert_phi = np.linspace(0., 2.5, 11)
ONN.same_phase_uncert = False
ONN.rng = 2
ONN.zeta = 0

# ONN.dataset_name = 'MNIST'
ONN.dataset_name = 'Gaussian'
# ONN.dataset_name = 'Iris'

# ONN.ONN_setup = np.array(['R_D_P', 'R_D_I_P', 'R_I_P', 'C_Q_P', 'C_W_P', 'E_P', 'R_P', 'I_P'])
ONN.ONN_setup = np.array(['R_P', 'C_Q_P'])

# for d in [[2,4,5,6],[0,1,2,9],[1,3,6,7]]:
    # onnClassTraining.ONN_Training(ONN, digits=d)
onnClassTraining.ONN_Training(ONN)

onnClassTraining.retrain_ONN(ONN, range(1))

'''
For MNIST PCA, [2, 4, 5, 6] w/ R_I_P = 88%
For MNIST PCA, [0, 1, 2, 9] w/ R_I_P = 87%
For MNIST PCA, [1, 3, 6, 7] w/ R_I_P = %

'''
