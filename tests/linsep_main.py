''' phase_uncert_thetar simulating Optical Neural Network
using Neuroptica and linearly separable datasets
Now goes over every topology types with N = 4-32

Author: Simon Geoffroy-Gagnon
Edit: 2020.03.09
'''
import numpy as np
import calculate_accuracy as calc_acc
import ONN_Simulation_Class as ONN_Cls
import onnClassTraining
import acc_colormap
import pickle

ONN = ONN_Cls.ONN_Simulation()
ONN.BATCH_SIZE = 2**6
ONN.EPOCHS = 600
ONN.STEP_SIZE = 0.005
ONN.ITERATIONS = 5 # number of times to retry same loss/PhaseUncert
ONN.loss_diff = 0 # \sigma dB

ONN.loss_dB = np.linspace(0, .5, 41)
ONN.phase_uncert_theta = np.linspace(0., .5, 41)
ONN.phase_uncert_phi = np.linspace(0., .5, 41)

ONN.rng = 2

# onn_topo = ['R_D_P']
onn_topo = ['R_P', 'C_Q_P', 'E_P', 'R_I_P', 'R_D_I_P', 'R_D_P']
# onn_topo = ['C_Q_P']
for ii in [0]:
    for N in [4]:
        for ONN.topo in onn_topo:
            ONN.get_topology_name()
            folder = f'/home/simon/Documents/neuroptica/linsep-datasets/LinSepFinal/N={N}/'

            ONN.X = np.loadtxt(folder + f'X.txt', delimiter=',')
            ONN.y = np.loadtxt(folder + f'y.txt', delimiter=',')
            ONN.Xt = np.loadtxt(folder + f'Xt.txt', delimiter=',')
            ONN.yt = np.loadtxt(folder + f'yt.txt', delimiter=',')
            ONN.N = N
            for ONN.rng in range(30, 40):
                ONN.phases = []
                model, *_ =  onnClassTraining.train_single_onn(ONN)
                if max(ONN.val_accuracy) > 97:
                    ONN.FOLDER = f'Analysis/linsep_final/N={N}_{ii}'
                    ONN.createFOLDER()

                    ONN.same_phase_uncert = False
                    print('Different Phase Uncert')
                    ONN.accuracy_PT = calc_acc.get_accuracy(ONN, model, ONN.Xt, ONN.yt,
                            loss_diff=ONN.loss_diff)

                    ONN.same_phase_uncert = True
                    print('Same Phase Uncert')
                    ONN.accuracy_LPU = calc_acc.get_accuracy(ONN, model, ONN.Xt, ONN.yt,
                            loss_diff=ONN.loss_diff)

                    acc_colormap.colormap_me(ONN)
                    ONN.saveAll(model)
                    np.savetxt(f'{ONN.FOLDER}/all_topologies.txt', onn_topo, fmt='%s')
                    break
