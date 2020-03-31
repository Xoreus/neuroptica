''' phase_uncert_thetar simulating Optical Neural Network
using Neuroptica and linearly separable datasets
Now goes over every topology types with N = 96 and trains them ONLY

Author: Simon Geoffroy-Gagnon
Edit: 2020.03.09
'''
import numpy as np
import calculate_accuracy as calc_acc
import ONN_Simulation_Class as ONN_Cls
import ONN_Setups
import acc_colormap
import digital_NN_main as dnn
import create_datasets as cd
import test_trained_onns as test

import sys
sys.path.append(r'C:\Users\sgeoff1\Documents\neuroptica')
sys.path.append('/home/simon/Documents/neuroptica')
import neuroptica as neu

import random
import os
import time
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

def change_dataset_shape(onn):
    if 'C' in onn.topo and 'Q' in onn.topo:
        X = np.array([list(np.zeros(int((onn.N-2)))) + list(samples) for samples in onn.X])
        Xt = np.array([list(np.zeros(int((onn.N-2)))) + list(samples) for samples in onn.Xt])
    elif 'C' in onn.topo and 'W' in onn.topo:
        X = (np.array([list(np.zeros(int((onn.N-2)/2))) + list(samples) +
            list(np.zeros(int(np.ceil((onn.N-2)/2)))) for samples in onn.X]))
        Xt = (np.array([list(np.zeros(int((onn.N-2)/2))) + list(samples) +
            list(np.zeros(int(np.ceil((onn.N-2)/2)))) for samples in onn.Xt]))
    else:
        X, Xt = onn.X, onn.Xt

    return X, onn.y, Xt, onn.yt

def get_dataset(ONN, rng, lim=99, SAMPLES=100, EPOCHS=30):
    while True:
        print(f'RNG = {rng}, N = {ONN.N}')
        X, y, Xt, yt = cd.gaussian_dataset(targets=int(ONN.N), features=int(ONN.N), nsamples=SAMPLES*ONN.N, rng=rng)
        random.seed(rng)

        X = (X - np.min(X))/(np.max(X) - np.min(X))
        Xt = (Xt - np.min(Xt))/(np.max(Xt) - np.min(Xt))

        net, weights = dnn.create_train_dnn(X, y, Xt, yt, EPOCHS)
        print('Validation Accuracy: {:.1f}%'.format(dnn.get_current_accuracy(Xt, yt, net)*100))
        rng += 1
        if dnn.get_current_accuracy(Xt, yt, net)*100 > lim:
            ONN.X = X
            ONN.y = y
            ONN.Xt = Xt
            ONN.yt = yt
            print('This dataset works!\n')
            return ONN, rng

def train_single_onn(onn):
    X, y, Xt, yt = onn.normalize_dataset()
    t = time.time()
    print(f'\nmodel: {onn.topo}, Loss/MZI = {onn.loss_dB[0]:.2f} dB, Loss diff = {onn.loss_diff}, Phase Uncert = {onn.phase_uncert_theta[0]:.2f} Rad, dataset = {onn.dataset_name}, rng = {onn.rng}, N = {onn.N}')

    model = ONN_Setups.ONN_creation(onn)

    X, y, Xt, yt = change_dataset_shape(onn)

    # initialize the ADAM optimizer and fit the ONN to the training data
    optimizer = neu.InSituAdam(model, neu.MeanSquaredError, step_size=onn.STEP_SIZE)
    onn.losses, onn.trn_accuracy, onn.val_accuracy, onn.phases, onn.best_trf_matrix = optimizer.fit(X.T, y.T, Xt.T, yt.T, epochs=onn.EPOCHS, batch_size=onn.BATCH_SIZE, show_progress=True)
    print(f'time spent for current training: {(time.time() - t)/60:.2f} minutes')
    return onn, model

if __name__ == '__main__':
    ONN = ONN_Cls.ONN_Simulation()
    ONN.BATCH_SIZE = 2**6
    ONN.EPOCHS = 400
    ONN.STEP_SIZE = 0.005
    ONN.ITERATIONS = 30 # number of times to retry same loss/PhaseUncert
    ONN.loss_diff = 0 # \sigma dB
    ONN.rng = 2
    ONN.zeta = 0.75
    ONN.PT_Area = (ONN.phase_uncert_phi[1] - ONN.phase_uncert_phi[0])**2
    ONN.LPU_Area = (ONN.loss_dB[1] - ONN.loss_dB[0])*(ONN.phase_uncert_phi[1] - ONN.phase_uncert_phi[0])

    ONN.loss_dB = [0]

    ONN.loss_diff = 0
    ONN.phase_uncert_theta = [0]
    ONN.phase_uncert_phi = [0]

    onn_topo = ['R_P', 'C_Q_P', 'E_P', 'R_I_P']
    # onn_topo = ['E_P', 'R_I_P']
    # onn_topo = ['R_P']
    # onn_topo = ['C_Q_P']
    rng = 8111
    ONN.N = 8*2*2
    FoM = {}
    for ii in range(1):
        ONN, rng = get_dataset(ONN, rng, lim=90)
        for ONN.topo in onn_topo:
            while True:
                ONN.get_topology_name()
                ONN, model = train_single_onn(ONN)
                ONN.FOLDER = f'Analysis/Lossy_Training/N={ONN.N}_loss_diff'
                ONN.FOLDER = f'Analysis/output_ports_pwer/N={ONN.N}'
                if max(ONN.val_accuracy) > 87:
                    ONN.createFOLDER()

                    ONN.loss_dB = np.linspace(0, 1, 41)
                    ONN.phase_uncert_theta = np.linspace(0., 0.4, 41)
                    ONN.phase_uncert_phi = np.linspace(0., 0.4, 41)

                    model = ONN_Setups.ONN_creation(ONN)
                    model.set_all_phases_uncerts_losses(ONN.phases, 0, 0, 0, 0)

                    ONN, model = test.test_LPU(ONN, model)

                    ONN.pickle_save()
                    ONN.saveAll(model)
                    np.savetxt(f'{ONN.FOLDER}/all_topologies.txt', onn_topo, fmt='%s')  

                    ONN.loss_dB = [0]

                    ONN.loss_diff = 0
                    ONN.phase_uncert_theta = [0]
                    ONN.phase_uncert_phi = [0]
                    break
