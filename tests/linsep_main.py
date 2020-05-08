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
ONN.EPOCHS = 300
ONN.STEP_SIZE = 0.005

ONN.ITERATIONS = 200 # number of times to retry same loss/PhaseUncert
rng = 123474
max_rng = 20
onn_topo = ['R_P', 'C_Q_P', 'E_P', 'R_I_P']
# onn_topo = ['R_P', 'C_Q_P']
# onn_topo = ['R_D_P', 'C_Q_P', 'R_I_D_P','E_D_P']
for ii in range(2):
    for ONN.N in [4, 8, 16, 32]:
        ONN, rng = train.get_dataset(ONN, rng, SAMPLES=50, EPOCHS=40)
        ONN.FOLDER = f'Analysis/Thesis/Loss_Layer/N={ONN.N}/N={ONN.N}_{ii}'
        ONN.createFOLDER()
        ONN.saveSimDataset()

        for ONN.topo in onn_topo:
            ONN.loss_diff = 0
            ONN.get_topology_name()
            for ONN.rng in range(max_rng):
                ONN.phases = []
                ONN.loss_dB = [0]
                ONN, model = train.train_single_onn(ONN)

                if max(ONN.val_accuracy) > 95 or ONN.rng == max_rng-1:
                    ONN.loss_dB = np.linspace(0, 1, 21)
                    ONN.phase_uncert_theta = np.linspace(0., 1, 21)
                    ONN.phase_uncert_phi = np.linspace(0., 2, 21)
                    test.test_PT(ONN, model)

                    test.test_LPU(ONN, model)
                    acc_colormap.colormap_me(ONN)
                    ONN.saveAll(model)
                    ONN.pickle_save()
                    break
