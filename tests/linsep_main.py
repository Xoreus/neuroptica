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
ONN.EPOCHS = 100
ONN.STEP_SIZE = 0.01
ONN.ITERATIONS = 10 # number of times to retry same loss/PhaseUncert
ONN.loss_diff = 0 # \sigma dB
ONN.loss_dB = np.linspace(0, 1.5, 21)
ONN.phase_uncert_theta = np.linspace(0., 0.5, 19)
ONN.phase_uncert_phi = np.linspace(0., 0.5, 19)

# ONN.same_phase_uncert = True

ONN.same_phase_uncert = False
# ONN.loss_dB = [0]

ONN.rng = 2
ONN.zeta = 0

onn_topo = ['R_P', 'C_Q_P', 'E_P']
for ii in range(1):
    for N in [64]:
        for ONN.onn_topo in onn_topo:
            ONN.get_topology_name()
            folder = f'/home/simon/Documents/neuroptica/linsep-datasets/N={N}/'
            ONN.X = np.loadtxt(folder + f'X_{ii}.txt', delimiter=',')
            ONN.y = np.loadtxt(folder + f'y_{ii}.txt', delimiter=',')
            ONN.Xt = np.loadtxt(folder + f'Xt_{ii}.txt', delimiter=',')
            ONN.yt = np.loadtxt(folder + f'yt_{ii}.txt', delimiter=',')
            ONN.N = N
            for ONN.rng in range(20):
                ONN.phases = []
                model, *_ =  onnClassTraining.train_single_onn(ONN)
                if max(ONN.val_accuracy) > 85:
                    ONN.accuracy = calc_acc.get_accuracy(ONN, model, ONN.Xt, ONN.yt, loss_diff=ONN.loss_diff)
                    ONN.FOLDER = f'Analysis/linsep/N={N}/N={N}_{ii}'
                    ONN.createFOLDER()
                    ONN.saveAll(model)
                    if ONN.same_phase_uncert:
                        acc_colormap.PU(ONN)
                    elif len(ONN.loss_dB) == 1:
                        acc_colormap.PhiTheta(ONN)
                    else:
                        acc_colormap.cube_plotting(ONN)

                    np.savetxt(f'{ONN.FOLDER}/all_topologies.txt', onn_topo, fmt='%s')
                    break
