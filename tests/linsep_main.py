''' Main function for simulating Optical Neural Network
using Neuroptica and linearly separable datasets

Author: Simon Geoffroy-Gagnon
Edit: 19.02.2020
'''
import numpy as np
import calculate_accuracy as calc_acc
import ONN_Simulation_Class as ONN_Cls
import onnClassTraining

ONN = ONN_Cls.ONN_Simulation()
ONN.N = 8
ONN.BATCH_SIZE = 2**6
ONN.EPOCHS = 1300
ONN.STEP_SIZE = 0.001
ONN.ITERATIONS = 20 # number of times to retry same loss/PhaseUncert
ONN.loss_diff = 0 # \sigma dB
ONN.loss_dB = np.linspace(0, 2, 21)
ONN.phase_uncert_theta = np.linspace(0., 1.5, 31)
ONN.phase_uncert_phi = np.linspace(0., 1.5, 31)
ONN.same_phase_uncert = True
ONN.rng = 2
ONN.zeta = 0

onn_topo = ['R_P', 'C_Q_P', 'E_P']
for ONN.onn_topo in onn_topo:
    ONN.get_topology_name()
    for N in [4]:
        folder = f'/home/simon/Documents/neuroptica/linsep-datasets/N={N}/'
        folder = f'/home/simon/Documents/neuroptica/tests/Analysis/test/Datasets/'
        folder = f'/home/simon/Documents/neuroptica/linsep-datasets/N=4-NotLinSep/'
        ONN.X = np.loadtxt(folder + 'X.txt', delimiter=',')
        ONN.y = np.loadtxt(folder + 'y.txt', delimiter=',')
        ONN.Xt = np.loadtxt(folder + 'Xt.txt', delimiter=',')
        ONN.yt = np.loadtxt(folder + 'yt.txt', delimiter=',')
        ONN.N = N

        for ONN.rng in range(1):
            ONN.phases = []
            model, *_ =  onnClassTraining.train_single_onn(ONN)
            ONN.accuracy = calc_acc.get_accuracy(ONN, model, ONN.Xt, ONN.yt, loss_diff=ONN.loss_diff)
            ONN.FOLDER = f'Analysis/test/N={N}'
            ONN.createFOLDER()
            ONN.saveAll(model)
            np.savetxt(f'{ONN.FOLDER}/all_topologies.txt', onn_topo, fmt='%s')
            with open(f'{ONN.FOLDER}/all_topologies.txt', 'a') as f:
                f.write(ONN.onn_topo)
            break

