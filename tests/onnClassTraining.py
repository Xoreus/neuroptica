"""
singleLossAnalysis.py - trains topology and tests with diff losses/phase uncerts (phi and theta)
Saves all required files for plotting in matlab (matlab is way better an making nice graphs...)

Author: Simon Geoffroy-Gagnon
Edit: 11.02.2020
"""
import pandas as pd
import sys
import random
import csv
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
from sklearn.model_selection import train_test_split
import time
import os
from copy import deepcopy

import ONN_Setups
import create_datasets as cd 
import setupSimulation as setSim
import calculate_accuracy as calc_acc
import ONN_Simulation_Class as ONN_Cls
import saveSimulationData as sSD
import digital_NN_main
import onnClassRetrained as retrain
sys.path.append('/home/simon/Documents/neuroptica')
import neuroptica as neu

def create_dataset(onn):
        if onn.dataset_name == 'MNIST' and onn.N == 4:
            onn.X, onn.y, onn.Xt, onn.yt = cd.MNIST_dataset([1,3,6,7], nsamples=onn.SAMPLES)
        elif onn.dataset_name == 'MNIST':
            onn.X, onn.y, onn.Xt, onn.yt = cd.MNIST_dataset(N=onn.N, nsamples=onn.SAMPLES)
        elif ONN.dataset_name == 'Gaussian':
            onn.X, onn.y, onn.Xt, onn.yt = cd.gaussian_dataset(targets=int(onn.N), features=int(onn.N), nsamples=onn.SAMPLES, rng=onn.rng)
        elif onn.dataset_name == 'Iris':
            onn.X, onn.y, onn.Xt, onn.yt = cd.iris_dataset(nsamples=int(onn.SAMPLES))

def retrain_ONN(ONN, rng_range):
    for ONN.rng in rng_range:
        ONN_Training(ONN, create_dataset_flag=False)

def ONN_Training(ONN, create_dataset_flag=True):
    ONN_Classes = []
    if create_dataset_flag: create_dataset(ONN)
    for onn in ONN.ONN_setup:
        ONN_Classes.append(deepcopy(ONN))
        ONN_Classes[-1].onn_topo = onn 
        ONN_Classes[-1].get_topology_name()
    for onn in ONN_Classes:
        random.seed(onn.rng)

        ROOT_FOLDER = r'Analysis/'
        FUNCTION = 'single_loss/'
        FOLDER = f'{onn.dataset_name}_N={onn.N}_loss-diff={onn.loss_diff}_rng{onn.rng}'
        onn.FOLDER = ROOT_FOLDER + FUNCTION + FOLDER 

        setSim.createFOLDER(onn.FOLDER)

        X, y, Xt, yt = onn.normalize_dataset()
        Xog, Xtog = X, Xt
        onn.saveSimDataset()

        t = time.time()
        print(f'model: {onn.onn_topo}, Loss = {onn.loss_dB[0]:.3f} dB, Phase Uncert = {onn.phase_uncert_theta[0]:.3f} Rad, dataset = {onn.dataset_name}, rng = {onn.rng}')

        model = ONN_Setups.ONN_creation(onn)

        X = Xog
        Xt = Xtog

        if 'C' in onn.onn_topo and 'Q' in onn.onn_topo:
            X = np.array([list(np.zeros(int((onn.N-2)))) + list(samples) for samples in onn.X])
            Xt = np.array([list(np.zeros(int((onn.N-2)))) + list(samples) for samples in onn.Xt])
        elif 'C' in onn.onn_topo and 'W' in onn.onn_topo:
            X = (np.array([list(np.zeros(int((onn.N-2)/2))) + list(samples) + 
                list(np.zeros(int(np.ceil((onn.N-2)/2)))) for samples in onn.X]))
            Xt = (np.array([list(np.zeros(int((onn.N-2)/2))) + list(samples) + 
                list(np.zeros(int(np.ceil((onn.N-2)/2)))) for samples in onn.Xt]))

        # initialize the ADAM optimizer and fit the ONN to the training data
        optimizer = neu.InSituAdam(model, neu.MeanSquaredError, step_size=ONN.STEP_SIZE)

        onn.losses, onn.trn_accuracy, onn.val_accuracy, onn.phases, onn.best_trf_matrix = optimizer.fit(X.T, y.T, Xt.T, yt.T, epochs=onn.EPOCHS, batch_size=onn.BATCH_SIZE, show_progress=True)

        # onn.saveTrainingData()
        onn.accuracy = calc_acc.get_accuracy(onn, model, Xt, yt)
        onn.saveSimData(model)
        ONN.accuracy.append(onn.accuracy)
        onn.saveAccuracyData()
        print(f'time spent for current training and testing all loss/phase uncert: {(time.time() - t)/60:.2f} minutes')
        onn.saveSelf()

    folder = os.path.split(onn.FOLDER)[-1] 
    print('\n' + folder + '\n')
    ONN.FOLDER = onn.FOLDER
    ONN.get_all_topologies()
    ONN.X, ONN.Xt, ONN.y, ONN.yt = onn.X, onn.Xt, onn.y, onn.yt
    with open(onn.FOLDER + '/ONN_Pickled_Class.P', 'wb') as f:
        pickle.dump(ONN, f)
    ONN.saveSelf()

if __name__ == '__main__':
    ONN = ONN_Cls.ONN_Simulation()
    ONN.N = 4
    ONN.BATCH_SIZE = 2**6
    ONN.EPOCHS = 1000
    ONN.STEP_SIZE = 0.001
    ONN.SAMPLES = 400
    ONN.ITERATIONS = 1 # number of times to retry same loss/PhaseUncert
    ONN.loss_diff = 0.5 # \sigma dB
    ONN.loss_dB = np.linspace(0, 1, 5)
    ONN.phase_uncert_theta = np.linspace(0.0, 1.5, 16)
    ONN.phase_uncert_phi = np.linspace(0.0, 1.5, 16)
    ONN.same_phase_uncert = False
    ONN.rng = 4

    # ONN.dataset_name = 'MNIST'
    ONN.dataset_name = 'Gaussian'
    # ONN.dataset_name = 'Iris'
    # ONN.ONN_setup = np.array(['R_I_P', 'C_Q_P', 'R_P', 'C_W_P', 'E_P', 'R_D_P', 'R_D_I_P'])
    ONN.ONN_setup = np.array(['R_P', 'I_P', 'C_Q_P', 'C_W_P'])

    ONN_Training(ONN)
    print('Starting different rng simulations')
    retrain_ONN(ONN, [5])
