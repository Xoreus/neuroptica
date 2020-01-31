"""
newSingleLossAnalysis.py
Testing nonlinearities with RI, RDI, R+I, the whole thing. Saves all required files for plotting in matlab (matlab is way better an making nice graphs...)
plus its good to save all data no matter what

Author: Simon Geoffroy-Gagnon
Edit: 30.01.2020
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
from sklearn.model_selection import train_test_split
import time
import os

import ONN_Setups
import create_datasets as cd 
import setupSimulation as setSim

import calculate_accuracy as calc_acc
import ONN_Simulation_Class as ONN_Cls
import saveSimulationData as sSD
import digital_NN_main

sys.path.append('/home/simon/Documents/neuroptica')
import neuroptica as neu

ONN = ONN_Cls.ONN_Simulation()

ONN.N = 4
ONN.BATCH_SIZE = 2**6
ONN.EPOCHS = 1300
ONN.STEP_SIZE = 0.0005
ONN.SAMPLES = 700
ONN.DATASET_NUM = 1
ONN.ITERATIONS = 100 # number of times to retry same loss/PhaseUncert
ONN.loss_diff = 0.1 # +/- dB
ONN.loss_dB = np.linspace(0, 3, 61)
ONN.phase_uncert = np.linspace(0, 1.5, 61)
ONN.RNG_RANGE = [2]

# ONN.dataset_name = 'MNIST'
# ONN.dataset_name = 'Gauss'
ONN.dataset_name = 'Iris'

# ONN_setup = np.array(['R_P', 'R_I_P', 'R_D_I_P', 'R_D_P', 'C_Q_P', 'C_W_P'])
ONN.ONN_setup = np.array(['R_D_P', 'C_Q_P'])

for ONN.rng in ONN.RNG_RANGE:
    random.seed(ONN.rng)

    ROOT_FOLDER = r'/home/simon/Documents/neuroptica/tests/Analysis/'
    FUNCTION = r'SingleLossAnalysis/'
    ONN.FOLDER = ROOT_FOLDER + FUNCTION + f'Reck+Diamond_{ONN.dataset_name}_loss-diff={ONN.loss_diff}_rng{ONN.rng}'

    # ONN.FOLDER = ROOT_FOLDER + FUNCTION + f'test'

    setSim.createFOLDER(ONN.FOLDER)

    got_accuracy = [0 for _ in range(len(ONN.ONN_setup))]

    ONN.saveSimSettings()

    if ONN.dataset_name == 'MNIST':
        ONN.X, ONN.y, ONN.Xt, ONN.yt = cd.MNIST_dataset([1,3,6,7], N=ONN.N, nsamples=ONN.SAMPLES)
    elif ONN.dataset_name == 'Gauss':
        ONN.X, ONN.y, ONN.Xt, ONN.yt = cd.gaussian_dataset(targets=int(ONN.N), 
                features=int(ONN.N), nsamples=ONN.SAMPLES, rng=ONN.rng)
    elif ONN.dataset_name == 'Iris':
        ONN.X, ONN.y, ONN.Xt, ONN.yt = cd.iris_dataset(nsamples=int(ONN.SAMPLES))

    X, y, Xt, yt = ONN.normalize_dataset()
    Xog, Xtog = X, Xt

    ONN.saveSimDataset()

    for ONN_Idx, ONN_Model in enumerate(ONN.ONN_setup):
        t = time.time()
        print(f'model: {ONN_Model}, Loss = {0:.3f} dB, Phase Uncert = {0:.3f} Rad, dataset = {ONN.dataset_name}, rng = {ONN.rng}')

        model = ONN_Setups.ONN_creation(ONN_Model, N=ONN.N)

        X = Xog
        Xt = Xtog

        if 'C' in ONN_Model and 'Q' in ONN_Model:
            X = np.array([list(np.zeros(int((ONN.N-2)))) + list(samples) for samples in ONN.X])
            Xt = np.array([list(np.zeros(int((ONN.N-2)))) + list(samples) for samples in ONN.Xt])
        elif 'C' in ONN_Model and 'W' in ONN_Model:
            X = (np.array([list(np.zeros(int((ONN.N-2)/2))) + list(samples) + 
                list(np.zeros(int(np.ceil((ONN.N-2)/2)))) for samples in ONN.X]))
            Xt = (np.array([list(np.zeros(int((ONN.N-2)/2))) + list(samples) + 
                list(np.zeros(int(np.ceil((ONN.N-2)/2)))) for samples in ONN.Xt]))

        # initialize the ADAM optimizer and fit the ONN to the training data
        optimizer = neu.InSituAdam(model, neu.MeanSquaredError, step_size=ONN.STEP_SIZE)

        currentSimResults  = optimizer.fit(X.T, y.T, Xt.T, yt.T, 
                epochs=ONN.EPOCHS, batch_size=ONN.BATCH_SIZE, show_progress=True)
        currentSimSettings = ONN.FOLDER, ONN_Model, ONN.loss_dB[0], ONN.phase_uncert[0], ONN.N, ONN.dataset_name

        sSD.saveSimData(currentSimSettings, currentSimResults, model)
        
        phases = currentSimResults[3]
        ONN.Phases.append(phases)
        accuracy = calc_acc.get_accuracy(ONN, model, Xt, yt)

        print(f'time spent for current training and testing all loss/phase uncert: {(time.time() - t)/60:.2f} minutes')
        got_accuracy[ONN_Idx]=1
        sSD.saveAccuracyData(ONN.FOLDER, currentSimSettings, accuracy)

    # Now test the same dataset using a Digital Neural Networks, 
    # just to see the difference between unitary and non-unitary matrix
    digital_NN_main.create_train_dnn(ONN.X, ONN.y, ONN.Xt, ONN.yt, ONN.FOLDER, EPOCHS = 100)

    with open(ONN.FOLDER + '/ONN_Pickled_Class.P', 'wb') as f:
        pickle.dump(ONN, f)

    print(f'Simulation in Folder {ONN.FOLDER} completed')
