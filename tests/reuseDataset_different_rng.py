"""
reuseDataset_different_rng.py
this is for testing whether or not a model achieves the global minimum always or it depends on the current RNG set.
This will help see if Diamond topo is acutally better than Reck.

Author: Simon Geoffroy-Gagnon
Edit: 29.01.2020
"""
import pandas as pd
import sys
import random
import csv
import re
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

# Get old ONN class using pickle load
FOLDER = '/home/simon/Documents/neuroptica/tests/Analysis/SingleLossAnalysis/'
LOWER_FOLD = 'samePhaseUncertTest_Gaussian_N=4_loss-diff=0.5_rng9/'
PKL_NAME = 'ONN_Pickled_Class.P'
OG_FOLDER = FOLDER + LOWER_FOLD + PKL_NAME

with open(OG_FOLDER, 'rb') as f:
    ONN = pickle.load(f)

ONN.loss_dB = np.linspace(0, 2, 5)
ONN.phase_uncert_theta = np.linspace(0, 1.5, 21)
ONN.phase_uncert_phi = np.linspace(0, 1.5, 21)
ONN.RNG_RANGE = []
ONN.STEP_SIZE = 0.001
ONN.ITERATIONS = 20
# ONN.ONN_setup = np.array(['C_Q_P', 'R_P', 'C_W_P'])
# ONN.ONN_setup = np.array([])

for ONN.rng in ONN.RNG_RANGE:
    random.seed(ONN.rng)

    ONN.FOLDER = re.sub('\d$', f'{ONN.rng}', ONN.FOLDER) + 'Retest'
    print(ONN.FOLDER)
    setSim.createFOLDER(ONN.FOLDER)

    got_accuracy = [0 for _ in range(len(ONN.ONN_setup))]

    ONN.saveSimSettings()

    Xog, Xtog = ONN.X, ONN.Xt

    ONN.saveSimDataset()

    for ONN_Idx, ONN_Model in enumerate(ONN.ONN_setup):
        t = time.time()
        print(f'model: {ONN_Model}, Loss = {0:.3f} dB, Phase Uncert = {0:.3f} Rad, dataset = {ONN.dataset_name}, rng = {ONN.rng}')

        model = ONN_Setups.ONN_creation(ONN_Model, N=ONN.N, loss_diff=ONN.loss_diff, loss_dB=ONN.loss_dB[0], phase_uncert=ONN.phase_uncert_theta[0])

        X, y, Xt, yt = ONN.normalize_dataset()
        Xog, Xtog = X, Xt

        if 'C' in ONN_Model and 'Q' in ONN_Model:
            X = np.array([list(np.zeros(int((ONN.N-2)))) + list(samples) for samples in ONN.X])
            Xt = np.array([list(np.zeros(int((ONN.N-2)))) + list(samples) for samples in ONN.Xt])
        elif 'C' in ONN_Model and 'W' in ONN_Model:
            X = (np.array([list(np.zeros(int((ONN.N-2)/2))) + 
                list(samples) + list(np.zeros(int(np.ceil((ONN.N-2)/2)))) for samples in ONN.X]))
            Xt = (np.array([list(np.zeros(int((ONN.N-2)/2))) + list(samples) + 
                list(np.zeros(int(np.ceil((ONN.N-2)/2)))) for samples in ONN.Xt]))

        # initialize the ADAM optimizer and fit the ONN to the training data
        optimizer = neu.InSituAdam(model, neu.MeanSquaredError, step_size=ONN.STEP_SIZE)

        currentSimResults = optimizer.fit(X.T, y.T, Xt.T, yt.T, epochs=ONN.EPOCHS, batch_size=ONN.BATCH_SIZE, show_progress=True)
        currentSimSettings = ONN.FOLDER, ONN_Model, ONN.loss_dB[0], ONN.phase_uncert_theta[0], ONN.N, ONN.dataset_name

        sSD.saveSimData(currentSimSettings, currentSimResults, model)
        
        phases = currentSimResults[3]
        ONN.Phases.append(phases)
        accuracy = calc_acc.get_accuracy(ONN, model, Xt, yt)

        print(f'time spent for current training and testing all loss/phase uncert: {(time.time() - t)/60:.2f} minutes')
        got_accuracy[ONN_Idx]=1
        sSD.saveAccuracyData(ONN.FOLDER, currentSimSettings, accuracy)

    # Now test the same dataset using a Digital Neural Networks, just to see the difference between unitary and non-unitary matrix
    if 0:
        digital_NN_main.create_train_dnn(ONN.X, ONN.y, ONN.Xt, ONN.yt, ONN.FOLDER, EPOCHS = 300)

    with open(ONN.FOLDER + '/ONN_Pickled_Class.P', 'wb') as f:
        pickle.dump(ONN, f)

    print(f'Simulation in Folder {ONN.FOLDER} completed')
