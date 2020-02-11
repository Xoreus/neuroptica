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
import re
import matplotlib as mpl
mpl.use('Agg')
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

sys.path.append('/home/simon/Documents/neuroptica')
import neuroptica as neu

# Get old ONN class using pickle load
FOLDER = '/home/simon/Documents/neuroptica/tests/Analysis/SingleLossAnalysis/'
LOWER_FOLD = 'samePhaseUncertTest_Gaussian_N=4_loss-diff=0.5_rng93/'
PKL_NAME = 'ONN_Pickled_Class.P'
OG_FOLDER = FOLDER + LOWER_FOLD + PKL_NAME

with open(OG_FOLDER, 'rb') as f:
    ONN = pickle.load(f)

ONN.RNG_RANGE = [1,  2]

ONN_Classes = []
for onn in ONN.ONN_setup:
    ONN_Classes.append(deepcopy(ONN))
    ONN_Classes[-1].onn_topo = onn 
    ONN_Classes[-1].get_topology_name()

for onn in ONN_Classes:
    for onn.rng in onn.RNG_RANGE:
        random.seed(onn.rng)

        print(onn.FOLDER)
        onn.FOLDER = re.sub('\d+$', f'{onn.rng}', onn.FOLDER)
        print(onn.FOLDER)
        setSim.createFOLDER(onn.FOLDER)

        X, y, Xt, yt = onn.normalize_dataset()
        Xog, Xtog = X, Xt
        onn.saveSimDataset()

        t = time.time()
        print(f'model: {onn.onn_topo}, Loss = {onn.loss_dB[0]:.3f} dB, Phase Uncert = {onn.phase_uncert_theta[0]:.3f} Rad, dataset = {onn.dataset_name}, rng = {onn.rng}')

        model = ONN_Setups.ONN_creation(onn.onn_topo, N=onn.N, loss_diff=onn.loss_diff, loss_dB=onn.loss_dB[0], phase_uncert=onn.phase_uncert_theta[0])

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

        onn.accuracy = calc_acc.get_accuracy(onn, model, Xt, yt)
        ONN.accuracy.append(onn.accuracy)
        onn.saveAccuracyData()
        print(f'time spent for current training and testing all loss/phase uncert: {(time.time() - t)/60:.2f} minutes')

        folder = os.path.split(onn.FOLDER)[-1] 
        print('\n' + folder + '\n')
        with open(onn.FOLDER + '/ONN_Pickled_Class.P', 'wb') as f:
            pickle.dump(ONN, f)

        ONN.FOLDER = onn.FOLDER
        ONN.get_all_topologies()
        ONN.saveSelf()
        print(f'Simulation in Folder {folder} completed')

        onn.saveSelf()
        

