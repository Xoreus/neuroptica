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

ONN = ONN_Cls.ONN_Simulation()

ONN.N = 4
ONN.BATCH_SIZE = 2**6
ONN.EPOCHS = 1400
ONN.STEP_SIZE = 0.001
ONN.SAMPLES = 600
ONN.ITERATIONS = 31 # number of times to retry same loss/PhaseUncert
ONN.loss_diff = 0.5 # \sigma dB
ONN.loss_dB = np.linspace(0, 2, 3)

ONN.phase_uncert_theta = np.linspace(0, 1.5, 26)
ONN.phase_uncert_phi = np.linspace(0, 1.5, 26)
ONN.RNG_RANGE = [10]
ONN.same_phase_uncert = True

# ONN.dataset_name = 'MNIST'
ONN.dataset_name = 'Gaussian'
# ONN.dataset_name = 'Iris'

# ONN.ONN_setup = np.array(['R_I_P', 'C_Q_P', 'R_P', 'C_W_P', 'E_P', 'R_D_P', 'R_D_I_P'])
ONN.ONN_setup = np.array(['C_Q_P', 'R_P'])

ONN_Classes = []
for onn in ONN.ONN_setup:
    ONN_Classes.append(deepcopy(ONN))
    ONN_Classes[-1].onn_topo = onn 
    ONN_Classes[-1].get_topology_name()

for onn in ONN_Classes:
    for onn.rng in onn.RNG_RANGE:
        random.seed(onn.rng)

        ROOT_FOLDER = r'/home/simon/Documents/neuroptica/tests/Analysis/'
        FUNCTION = r'SingleLossAnalysis/'
        FOLDER = f'samePhaseUncertTest_{onn.dataset_name}_N={onn.N}_loss-diff={onn.loss_diff}_rng{onn.rng}'

        onn.FOLDER = ROOT_FOLDER + FUNCTION + FOLDER 

        setSim.createFOLDER(onn.FOLDER)

        if onn.dataset_name == 'MNIST':
            onn.X, onn.y, onn.Xt, onn.yt = cd.MNIST_dataset([1,3,6,7], nsamples=onn.SAMPLES)
        elif ONN.dataset_name == 'Gaussian':
            onn.X, onn.y, onn.Xt, onn.yt = cd.gaussian_dataset(targets=int(onn.N), features=int(onn.N), nsamples=onn.SAMPLES, rng=onn.rng)
        elif onn.dataset_name == 'Iris':
            onn.X, onn.y, onn.Xt, onn.yt = cd.iris_dataset(nsamples=int(onn.SAMPLES))

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
    onn.saveSelf()
        
folder = os.path.split(onn.FOLDER)[-1] 
print('\n' + folder + '\n')
# print(onn.FOLDER)
ONN.FOLDER = onn.FOLDER
ONN.get_all_topologies()
ONN.X, ONN.Xt, ONN.y, ONN.yt = onn.X, onn.Xt, onn.y, onn.yt
with open(onn.FOLDER + '/ONN_Pickled_Class.P', 'wb') as f:
    pickle.dump(ONN, f)

ONN.saveSelf()
print(f'Simulation in Folder {folder} completed')

