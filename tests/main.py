''' Main function for simulating Optical Neural Network
using Neuroptica and the MNIST dataset

Author: Simon Geoffroy-Gagnon
Edit: 19.02.2020
'''
import numpy as np
import sys
import calculate_accuracy as calc_acc
import ONN_Simulation_Class as ONN_Cls
import onnClassTraining
import acc_colormap
import N_Finder


onn = ONN_Cls.ONN_Simulation()
onn.BATCH_SIZE = 2**6
onn.EPOCHS = 200
onn.STEP_SIZE = 0.005
onn.ITERATIONS = 10 # number of times to retry same loss/PhaseUncert
onn.loss_diff = 0 # \sigma dB
onn.loss_dB = np.linspace(0, 3, 31)
onn.phase_uncert_theta = np.linspace(0, .5, 2)
onn.phase_uncert_phi = np.linspace(0., .5, 2)
onn.dataset_name = 'MNIST'
onn.SAMPLES = 3000

onn.rng = 4
onn.zeta = 0

onn_topo = ['R_P']
for onn.N in [4]:
    folder = '/home/simon/Documents/neuroptica/linsep-datasets/N=6'

    X, y, Xt, yt = N_Finder.get_dataset(folder, onn.N)

    for onn.topo in onn_topo:
        onn.get_topology_name()
        for onn.rng in range(22):
            onn.phases = []
            onn, model =  onnClassTraining.train_single_onn(onn)
            if max(onn.val_accuracy) > 10:
                onn.FOLDER = f'Analysis/multiple_trainings/N={onn.N}_{onn.rng}'
                onn.createFOLDER()
                onn.saveAll(model)
                np.savetxt(f'{onn.FOLDER}/all_topologies.txt', onn_topo, fmt='%s')
                break
