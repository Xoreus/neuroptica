''' phase_uncert_thetar simulating Optical Neural Network
using Neuroptica and linearly separable datasets
Now goes over every topology types with N = 4-32

Author: Simon Geoffroy-Gagnon
Edit: 2020.03.28
'''
import numpy as np
import calculate_accuracy as calc_acc
import ONN_Simulation_Class as ONN_Cls
import acc_colormap
import training_onn as train
import test_trained_onns as test
from collections import defaultdict

ONN = ONN_Cls.ONN_Simulation()
ONN.BATCH_SIZE = 2**6
ONN.EPOCHS = 200
ONN.STEP_SIZE = 0.005

ONN.ITERATIONS = 5*2 # number of times to retry same loss/PhaseUncert
ONN.loss_diff = 0 # \sigma dB

rng = 651231

onn_topo = ['R_P', 'C_Q_P', 'E_P', 'R_I_P']
# onn_topo = ['R_D_P']
for ii in range(1, 10):
    for ONN.N in [8]:
        ONN, rng = train.get_dataset(ONN, rng)
        for ONN.topo in onn_topo:
            ONN.get_topology_name()
            for ONN.rng in range(10):
                ONN.phases = []
                ONN.loss_dB = [0]
                ONN, model = train.train_single_onn(ONN)
                if max(ONN.val_accuracy) > 96:
                    ONN.FOLDER = f'Analysis/IL/N={ONN.N}_0dB_train/N={ONN.N}_{ii}'
                    # ONN.FOLDER = f'Analysis/Lossy_Training/N={ONN.N}_{ii}'
                    # ONN.FOLDER = f'/home/simon/Documents/neuroptica/tests/Analysis/phaseConvergence/N={ONN.N}'
                    ONN.createFOLDER()

                    ONN.loss_dB = np.linspace(0, 3.5, 41)
                    ONN.phase_uncert_theta = np.linspace(0., 0.5, 11)
                    ONN.phase_uncert_phi = np.linspace(0., 0.5, 11)

                    test.test_onn(ONN, model)

                    acc_colormap.colormap_me(ONN)
                    ONN.saveAll(model)
                    ONN.pickle_save()

                    np.savetxt(f'{ONN.FOLDER}/all_topologies.txt', onn_topo, fmt='%s')
                    break
