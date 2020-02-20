''' changes the loss and phase uncert limits for testing different topology sizes
using Neuroptica and linearly separable datasets

Author: Simon Geoffroy-Gagnon
Edit: 20.02.2020
'''
import numpy as np
import setupSimulation as setSim
import calculate_accuracy as calc_acc
import ONN_Simulation_Class as ONN_Cls
import onnClassTraining
import ONN_Setups 


ONN = ONN_Cls.ONN_Simulation()
ONN.N = 8
ONN.BATCH_SIZE = 2**6
ONN.EPOCHS = 2500
ONN.STEP_SIZE = 0.0005
ONN.ITERATIONS = 10 # number of times to retry same loss/PhaseUncert
ONN.loss_diff = 0 # \sigma dB
ONN.loss_dB = np.linspace(0, 1, 11)
ONN.phase_uncert_theta = np.linspace(0., 2, 21)
ONN.phase_uncert_phi = np.linspace(0., 2, 21)
ONN.same_phase_uncert = False
ONN.rng = 2
ONN.zeta = 0
topos = ['R_P', 'C_Q_P']

for ONN.onn_topo in topos:
    ONN.get_topology_name()
    for N in range(4, 5):
        folder = f'/home/simon/Documents/neuroptica/linsep-datasets/N={N}/'
        FOLDER = f'/home/simon/Documents/neuroptica/tests/Analysis/linsep/N={N}'
        ONN.X = np.loadtxt(folder + 'X.txt', delimiter=',')
        ONN.y = np.loadtxt(folder + 'y.txt', delimiter=',')
        ONN.Xt = np.loadtxt(folder + 'Xt.txt', delimiter=',')
        ONN.yt = np.loadtxt(folder + 'yt.txt', delimiter=',')
        ONN.N = N
        phases = np.loadtxt(f'{FOLDER}/Phases/Phases_Best_{ONN.onn_topo}_loss=0.000dB_uncert=0.000Rad_{N}Features.txt', skiprows=1, usecols=(1,2), delimiter=',')
        ONN.phases = [[(t, p) for t, p in phases]]
        model = ONN_Setups.ONN_creation(ONN)
        ONN.accuracy = calc_acc.get_accuracy(ONN, model, ONN.Xt, ONN.yt, loss_diff=ONN.loss_diff, zeta=ONN.zeta)

        # break
        ONN.FOLDER = f'Analysis/linsep/N={N}-newSimValues'
        ONN.createFOLDER()
        ONN.saveSelf()
        ONN.saveSimDataset()
        ONN.saveAccuracyData()
        np.savetxt(f'{ONN.FOLDER}/all_topologies.txt', topos, fmt='%s')

