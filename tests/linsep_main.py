''' Main function for simulating Optical Neural Network
using Neuroptica and linearly separable datasets

Author: Simon Geoffroy-Gagnon
Edit: 19.02.2020
'''
import numpy as np
import setupSimulation as setSim
import calculate_accuracy as calc_acc
import ONN_Simulation_Class as ONN_Cls
import onnClassTraining

ONN = ONN_Cls.ONN_Simulation()
ONN.N = 8
ONN.BATCH_SIZE = 2**6
ONN.EPOCHS = 2500
ONN.STEP_SIZE = 0.001
ONN.ITERATIONS = 20 # number of times to retry same loss/PhaseUncert
ONN.loss_diff = 0 # \sigma dB
ONN.loss_dB = np.linspace(0, 2, 11)
ONN.phase_uncert_theta = np.linspace(0., 2.5, 16)
ONN.phase_uncert_phi = np.linspace(0., 2.5, 16)
ONN.same_phase_uncert = False
ONN.rng = 2
ONN.zeta = 0

onn_topo = ['R_P','C_Q_P']
for ONN.onn_topo in onn_topo:
    ONN.get_topology_name()
    for N in range(5, 10):
        folder = f'/home/simon/Documents/neuroptica/linsep-datasets/N={N}/'
        ONN.X = np.loadtxt(folder + 'X.txt', delimiter=',')
        ONN.y = np.loadtxt(folder + 'y.txt', delimiter=',')
        ONN.Xt = np.loadtxt(folder + 'Xt.txt', delimiter=',')
        ONN.yt = np.loadtxt(folder + 'yt.txt', delimiter=',')
        ONN.N = N

        for ONN.rng in range(20):
            ONN.phases = []
            model, *_ =  onnClassTraining.train_single_onn(ONN)
            if max(ONN.val_accuracy) > 95:
                ONN.accuracy = calc_acc.get_accuracy(ONN, model, ONN.Xt, ONN.yt, loss_diff=ONN.loss_diff, zeta=ONN.zeta)
                ONN.FOLDER = f'Analysis/linsep/N={N}'
                setSim.createFOLDER(ONN.FOLDER)
                ONN.saveAll(model)
                np.savetxt(f'{ONN.FOLDER}/all_topologies.txt', onn_topo, fmt='%s')
                break

