''' Main function for simulating Optical Neural Network
using Neuroptica and linearly separable datasets

Author: Simon Geoffroy-Gagnon
Edit: 19.02.2020
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
ONN.EPOCHS = 2500
ONN.STEP_SIZE = 0.0005
ONN.SAMPLES = 3000
ONN.ITERATIONS = 20 # number of times to retry same loss/PhaseUncert
ONN.loss_diff = 0 # \sigma dB
ONN.loss_dB = np.linspace(0, 2, 3)
ONN.phase_uncert_theta = np.linspace(0., 2.5, 11)
ONN.phase_uncert_phi = np.linspace(0., 2.5, 11)
ONN.same_phase_uncert = False
ONN.rng = 2
ONN.zeta = 0
x = 32
reps = 0

# ONN.ONN_setup = np.array(['R_D_P', 'R_D_I_P', 'R_I_P', 'C_Q_P', 'C_W_P', 'E_P', 'R_P', 'I_P'])
# ONN.ONN_setup = np.array(['R_P'])
# ONN.ONN_setup = np.array(['C_Q_P'])
ONN.onn_topo = 'R_P'
ONN.onn_topo = 'C_Q_P'
ONN.onn_topo = 'R_I_P'

Ns = range(4, 21)
for N in Ns:
    folder = f'/home/simon/Documents/neuroptica/Linearly-Separable-Datasets/N={N}/'
    ONN.X = np.loadtxt(folder + 'X.txt', delimiter=',')
    ONN.y = np.loadtxt(folder + 'y.txt', delimiter=',')
    ONN.Xt = np.loadtxt(folder + 'Xt.txt', delimiter=',')
    ONN.yt = np.loadtxt(folder + 'yt.txt', delimiter=',')
    ONN.N = N

    for ONN.rng in range(20):
        ONN.phases = []
        ONN.topo = ONN.ONN_setup[0]
        ONN.get_topology_name()
        model, *_ =  onnClassTraining.train_single_onn(ONN)
        if ONN.val_accuracy[-1] > 85:
            ONN.accuracy = calc_acc.get_accuracy(ONN, model, ONN.Xt, ONN.yt, loss_diff=ONN.loss_diff, zeta=ONN.zeta)
            ONN.FOLDER = r'Analysis/linsep/' + f'{ONN.dataset_name}_N={ONN.N}_loss-diff={ONN.loss_diff}_rng{ONN.rng}'
            setSim.createFOLDER(ONN.FOLDER)
            ONN.saveAll(model)
            break

