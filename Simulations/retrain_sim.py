''' Main function for simulating Optical Neural Network
retrains with same dataset but saves pickle and stuff
Author: Simon Geoffroy-Gagnon
Edit: 19.02.2020
'''
import numpy as np
import sys
import calculate_accuracy as calc_acc
import ONN_Simulation_Class as ONN_Cls
import training_onn as train
import test_trained_onns as test

onn = ONN_Cls.ONN_Simulation()
onn.BATCH_SIZE = 2**6
onn.EPOCHS = 300
onn.STEP_SIZE = 0.005

onn.rng = 5
onn.zeta = 0

# onn_topo = ['R_P', 'C_Q_P']
onn_topo = ['R_P', 'B_C_Q_P', 'E_P']
for onn.N in [8]:
    for onn.topo in onn_topo:
        onn.get_topology_name()
        onn.FOLDER = f'/home/edwar/Documents/Github_Projects/neuroptica/tests/Analysis/N=16'
        onn.X = np.loadtxt(onn.FOLDER + '/Datasets/X.txt',delimiter=',')
        onn.y = np.loadtxt(onn.FOLDER + '/Datasets/y.txt',delimiter=',')
        onn.Xt = np.loadtxt(onn.FOLDER + '/Datasets/Xt.txt',delimiter=',')
        onn.yt = np.loadtxt(onn.FOLDER + '/Datasets/yt.txt',delimiter=',')

        # onn.FOLDER = f'/storage/Research/OE_2020/N={onn.N}/N={onn.N}_retrained'

        for onn.rng in range(10):
            onn.phases = []
            onn.createFOLDER()
            onn, model =  train.train_single_onn(onn)
            if max(onn.val_accuracy) > 95:

                onn.ITERATIONS = 50 # number of times to retry same loss/PhaseUncert
                onn.loss_diff = 0 # \sigma dB
                onn.loss_dB = np.linspace(0, 1, 41)
                onn.phase_uncert_theta = np.linspace(0, 0.5, 41)
                onn.phase_uncert_phi = np.linspace(0, 0.5, 41)
                test.test_LPU(onn, model)
                test.test_PT(onn, model)

                onn.saveAll(model)
                onn.pickle_save()
                break

