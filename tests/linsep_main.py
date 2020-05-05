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

ONN.ITERATIONS = 20 # number of times to retry same loss/PhaseUncert
rng = 5588151   

onn_topo = ['R_P', 'C_Q_P', 'E_P', 'R_I_P']
for ii in range(1):
    for ONN.N in [50]:
        ONN, rng = train.get_dataset(ONN, rng, SAMPLES=100, EPOCHS=50)
        ONN.FOLDER = f'Analysis/N={ONN.N}/N={ONN.N}_{ii}'
        ONN.createFOLDER()
        ONN.saveSimDataset()

        for ONN.topo in onn_topo:
            ONN.loss_diff = 0
            ONN.get_topology_name()
            for ONN.rng in range(10):
                ONN.phases = []
                ONN.loss_dB = [0]
                ONN, model = train.train_single_onn(ONN)
                if max(ONN.val_accuracy) > 95:
                    ONN.loss_dB = np.linspace(0, 0.5, 21)
                    ONN.phase_uncert_theta = np.linspace(0., 0.1, 21)
                    ONN.phase_uncert_phi = np.linspace(0., 0.1, 21)
                    ONN.loss_diff = 0
                    test.test_onn(ONN, model)
                    acc_colormap.colormap_me(ONN)
                    ONN.saveAll(model)
                    ONN.pickle_save()
                    break
