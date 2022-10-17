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
import digital_NN_main as dnn
import create_datasets as cd
import test_trained_onns as test
import sys
sys.path.append('../../neuroptica')
import neuroptica as neu
from plot_scatter_matrix import plot_scatter_matrix
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
import matplotlib.pyplot as plt

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

def get_dataset(onn, rng, linear_sep_acc_limit=97, SAMPLES=80, EPOCHS=30, extra_channels=0):
    while True:
        print(f'RNG = {rng}, N = {onn.features}, Digital Neural Network')
        X, y, Xt, yt = cd.gaussian_dataset(targets=int(onn.classes), features=int(onn.features), nsamples=SAMPLES*onn.features, rng=rng)
        random.seed(rng)

        X = (X - np.min(X))/(np.max(X) - np.min(X))
        Xt = (Xt - np.min(Xt))/(np.max(Xt) - np.min(Xt))

        net, weights = dnn.create_train_dnn(X, y, Xt, yt, EPOCHS, hnum=0)
        print('Validation Accuracy: {:.1f}%\n'.format(dnn.get_current_accuracy(Xt, yt, net)*100))
        rng += 1
        if dnn.get_current_accuracy(Xt, yt, net)*100 > linear_sep_acc_limit:
            onn.X = X
            onn.y = y
            onn.Xt = Xt
            onn.yt = yt
            print('This dataset is linearly separable!\n')
            print(f"X: {np.shape(X)}")
            print(f"Y: {np.shape(y)}")
            print(f"Xt: {np.shape(Xt)}")
            print(f"Yt: {np.shape(yt)}")
            return onn, rng

def create_model(onn):
    X, y, Xt, yt = onn.normalize_dataset()

    print(f'\nmodel: {onn.topo}, Loss/MZI = {onn.loss_dB[0]:.2f} dB, Loss diff = {onn.loss_diff}, Phase Uncert = {onn.phase_uncert_theta[0]:.2f} Rad, dataset = {onn.dataset_name}, rng = {onn.rng}, N = {onn.N}')

    model = ONN_Setups.ONN_creation(onn)

    return model

def train_single_onn(onn, model, loss_function='cce'): # cce: categorical cross entropy, mse: mean square error
    t = time.time()
    # X, y, Xt, yt = change_dataset_shape(onn)
    X, y, Xt, yt = onn.X, onn.y, onn.Xt, onn.yt
    # initialize the ADAM optimizer and fit the ONN to the training data
    if loss_function == 'mse':
        optimizer = neu.InSituAdam(model, neu.MeanSquaredError, step_size=onn.STEP_SIZE)
    elif loss_function == 'cce':
        optimizer = neu.InSituAdam(model, neu.CategoricalCrossEntropy, step_size=onn.STEP_SIZE)
    onn.losses, onn.trn_accuracy, onn.val_accuracy, onn.phases, onn.best_trf_matrix = optimizer.fit(X.T, y.T, Xt.T, yt.T, epochs=onn.EPOCHS, batch_size=onn.BATCH_SIZE, show_progress=True)
    print(f'time spent for current training: {(time.time() - t)/60:.2f} minutes')
    return onn, model

if __name__ == '__main__':
    onn = ONN_Cls.ONN_Simulation()
    onn.BATCH_SIZE = 2**6
    onn.EPOCHS = 400
    onn.STEP_SIZE = 0.005
    onn.ITERATIONS = 30 # number of times to retry same loss/PhaseUncert
    onn.loss_diff = 0 # \sigma dB
    onn.rng = 2
    onn.zeta = 0.75
    onn.PT_Area = (onn.phase_uncert_phi[1] - onn.phase_uncert_phi[0])**2
    onn.LPU_Area = (onn.loss_dB[1] - onn.loss_dB[0])*(onn.phase_uncert_phi[1] - onn.phase_uncert_phi[0])

    onn.loss_dB = [0]

    onn.loss_diff = 0
    onn.phase_uncert_theta = [0]
    onn.phase_uncert_phi = [0]

    onn_topo = ['R_P', 'C_Q_P', 'E_P', 'R_I_P']
    # onn_topo = ['E_P', 'R_I_P']
    # onn_topo = ['R_P']
    # onn_topo = ['B_C_Q_P']
    rng = 8111
    onn.N = 8*2*2
    FoM = {}
    for ii in range(1):
        ONN, rng = get_dataset(ONN, rng, linear_sep_acc_limit=90)
        for onn.topo in onn_topo:
            while True:
                onn.get_topology_name()
                ONN, model = train_single_onn(ONN)
                onn.FOLDER = f'Analysis/Lossy_Training/N={onn.N}_loss_diff'
                onn.FOLDER = f'Analysis/output_ports_pwer/N={onn.N}'
                if max(onn.val_accuracy) > 87:
                    onn.createFOLDER()

                    onn.loss_dB = np.linspace(0, 1, 41)
                    onn.phase_uncert_theta = np.linspace(0., 0.4, 41)
                    onn.phase_uncert_phi = np.linspace(0., 0.4, 41)

                    model = ONN_Setups.ONN_creation(ONN)
                    model.set_all_phases_uncerts_losses(onn.phases, 0, 0, 0, 0)

                    ONN, model = test.test_LPU(ONN, model)

                    onn.pickle_save()
                    onn.saveAll(model)
                    np.savetxt(f'{onn.FOLDER}/all_topologies.txt', onn_topo, fmt='%s')  

                    onn.loss_dB = [0]

                    onn.loss_diff = 0
                    onn.phase_uncert_theta = [0]
                    onn.phase_uncert_phi = [0]
                    break
