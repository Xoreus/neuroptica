''' Main function for simulating Optical Neural Network
using Neuroptica and the MNIST dataset

Author: Simon Geoffroy-Gagnon
Edit: 19.02.2020
'''
import numpy as np
import sys
import calculate_accuracy as calc_acc
import ONN_Simulation_Class as ONN_Cls
import training_onn as train
import test_trained_onns as test
import acc_colormap


onn = ONN_Cls.ONN_Simulation()
onn.BATCH_SIZE = 2**6
onn.EPOCHS = 500
onn.STEP_SIZE = 0.005

onn.ITERATIONS = 50 # number of times to retry same loss/PhaseUncert
onn.loss_diff = 0 # \sigma dB
onn.loss_dB = np.linspace(0, .8, 41)
onn.phase_uncert_theta = np.linspace(0., 0.2, 41)
onn.phase_uncert_phi = np.linspace(0., 0.2, 41)

onn.rng = 4
onn.zeta = 0

onn_topo = ['R_P']
for onn.N in [32]:
    for onn.topo in onn_topo:
        onn.FOLDER = '/home/simon/Documents/neuroptica/tests/Analysis/N=4/N=4_2_100.000'
        print(onn.topo)
        onn = onn.pickle_load()
        print(onn.X)
        onn.FOLDER = '/home/simon/Documents/neuroptica/tests/Analysis/N=4/N=4_2_100.000_2'
        for onn.rng in range(10):
            onn.phases = []
            onn, model =  train.train_single_onn(onn)
            if max(onn.val_accuracy) > 90:
                # onn.FOLDER = f'Analysis/N={onn.N}/N={onn.N}_{onn.rng}'
                onn.createFOLDER()
                onn.saveAll(model)
                onn.pickle_save()
                break
