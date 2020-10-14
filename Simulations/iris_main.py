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
import create_datasets
import training_onn as train
import test_trained_onns as test

onn = ONN_Cls.ONN_Simulation()
onn.BATCH_SIZE = 2**6
onn.EPOCHS = 500
onn.STEP_SIZE = 0.005
onn.ITERATIONS = 10 # number of times to retry same loss/PhaseUncert
onn.loss_diff = 0 # \sigma dB
onn.dataset_name = 'iris'
onn.SAMPLES = 1200

onn.rng = 4
max_rng = 6
onn_topo = ['R_P', 'C_Q_P', 'E_P', 'R_I_P']
for onn.N in [4]:

    onn.X, onn.y, onn.Xt, onn.yt = create_datasets.iris_dataset(nsamples=onn.SAMPLES)
    onn.FOLDER = f'Analysis/iris/N={onn.N}/N={onn.N}'
    onn.createFOLDER()
    onn.saveSimDataset()

    for onn.topo in onn_topo:
        onn.loss_diff = 0
        onn.get_topology_name()
        for onn.rng in range(max_rng):
            np.random.seed(onn.rng)
            onn.phases = []
            onn.loss_dB = [0]
            onn, model = train.train_single_onn(onn, loss_function='mse')

            if max(onn.val_accuracy) < 60 or onn.rng == max_rng-1:
                onn.loss_diff = 0.25
                onn.loss_dB = np.linspace(0, 4, 11)
                onn.phase_uncert_theta = np.linspace(0., 1, 2)
                onn.phase_uncert_phi = np.linspace(0., 1, 2)
                test.test_PT(onn, model)
                test.test_LPU(onn, model)
                onn.saveAll(model)
                onn.pickle_save()
                break
