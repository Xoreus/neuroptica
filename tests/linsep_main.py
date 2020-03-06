''' Main function for simulating Optical Neural Network
using Neuroptica and linearly separable datasets

Author: Simon Geoffroy-Gagnon
Edit: 19.02.2020
'''
import numpy as np
import calculate_accuracy as calc_acc
import ONN_Simulation_Class as ONN_Cls
import onnClassTraining
import acc_colormap


ONN = ONN_Cls.ONN_Simulation()
ONN.N = 8
ONN.BATCH_SIZE = 2**6
ONN.EPOCHS = 200
ONN.STEP_SIZE = 0.005
ONN.ITERATIONS = 5 # number of times to retry same loss/PhaseUncert
ONN.loss_diff = 0 # \sigma dB
ONN.loss_dB = np.linspace(0, 1.5, 5)
ONN.phase_uncert_theta = np.linspace(0., 0.4, 5)
ONN.phase_uncert_phi = np.linspace(0., 0.4, 5)

ONN.rng = 2
ONN.zeta = 0

onn_topo = ['R_P', 'C_Q_P', 'E_P']
# onn_topo = ['R_P', 'C_Q_P']
# onn_topo = ['R_P']
# onn_topo = [ 'C_Q_P']
for ii in [1]:
    for N in [12]:
        for ONN.onn_topo in onn_topo:
            ONN.get_topology_name()
            folder = f'/home/simon/Documents/neuroptica/linsep-datasets/N={N}_new/'
            ONN.X = np.loadtxt(folder + f'X_{ii}.txt', delimiter=',')
            ONN.y = np.loadtxt(folder + f'y_{ii}.txt', delimiter=',')
            ONN.Xt = np.loadtxt(folder + f'Xt_{ii}.txt', delimiter=',')
            ONN.yt = np.loadtxt(folder + f'yt_{ii}.txt', delimiter=',')
            ONN.N = N
            for ONN.rng in range(20):
                ONN.phases = []
                model, *_ =  onnClassTraining.train_single_onn(ONN)
                if max(ONN.val_accuracy) > 92:
                    ONN.accuracy = calc_acc.get_accuracy(ONN, model, ONN.Xt, ONN.yt, loss_diff=ONN.loss_diff)
                    ONN.FOLDER = f'Analysis/linsep/N={N}_{ii}_NoLoss'
                    ONN.createFOLDER()
                    ONN.saveAll(model)

                    ONN.same_phase_uncert = False
                    print('Different Phase Uncert')
                    ONN.accuracy_PT = calc_acc.get_accuracy(ONN, model, ONN.Xt, ONN.yt, loss_diff=ONN.loss_diff)

                    ONN.same_phase_uncert = True
                    print('Same Phase Uncert')
                    ONN.accuracy_LPU = calc_acc.get_accuracy(ONN, model, ONN.Xt, ONN.yt, loss_diff=ONN.loss_diff)
                    acc_colormap.colormap_me(ONN)

                    np.savetxt(f'{ONN.FOLDER}/all_topologies.txt', onn_topo, fmt='%s')
                    break
