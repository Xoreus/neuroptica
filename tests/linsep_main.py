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

ONN = ONN_Cls.ONN_Simulation()
ONN.BATCH_SIZE = 2**6
ONN.EPOCHS = 200
ONN.STEP_SIZE = 0.005

ONN.ITERATIONS = 20 # number of times to retry same loss/PhaseUncert
rng_og = 11
max_rng = 10 # how many repititions 
onn_topo = ['R_P', 'C_Q_P', 'E_P', 'R_I_P']
# onn_topo = ['R_P', 'C_Q_P']
onn_topo = ['R_P']
# onn_topo = ['R_I_D_P','E_D_P', 'R_D_P', 'C_Q_P']
for ii in range(1):
    for ONN.N in [8]:
        loss_diff = [1, 0.025, 0.25, 0.4]

        for ld in loss_diff:
            rng = rng_og
            ONN, rng = train.get_dataset(ONN, rng, SAMPLES=40, EPOCHS=60)
            ONN.FOLDER = f'Analysis/perfectTraining/N={ONN.N}/N={ONN.N}_{ld}dB'
            ONN.createFOLDER()
            ONN.saveSimDataset()

            for ONN.topo in onn_topo:
                ONN.loss_diff = ld
                ONN.get_topology_name()
                for ONN.rng in range(max_rng):
                    np.random.seed(ONN.rng)
                    ONN.phases = []
                    ONN.loss_dB = [0]
                    ONN, model = train.train_single_onn(ONN, loss_function='mse')

                    if max(ONN.val_accuracy) > 85 or ONN.rng == max_rng-1:
                        ONN.loss_diff = ld
                        ONN.loss_dB = np.linspace(0, 4, 6)
                        ONN.phase_uncert_theta = np.linspace(0., 1, 6)
                        ONN.phase_uncert_phi = np.linspace(0., 1, 6)
                        test.test_PT(ONN, model)
                        test.test_LPU(ONN, model)
                        # acc_colormap.colormap_me(ONN)
                        ONN.saveAll(model)
                        ONN.pickle_save()
                        break

