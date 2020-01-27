"""
Retrained_Topology_Analysis.py
This time, analyzing if addng a small amnt of noise helps to reduce overfitting (regularization)
also comparing diamond layer to central diamond layer, maybe itle help with loss?
also looks at adding etra MZIs in series (by doing R_P, R_I_P, R_D_I_P, R_D_I_R_D_I_P) to look at effects of increasing loss and/or phase uncert 
Retrains models for every loss/phase_uncert to look at how the loss and uncert dissapear when we train the model
Testing nonlinearities with RI, RDI, R+I, the whole thing. Saves all required files for plotting in matlab (matlab is way better an making nice graphs...), plus its good to save all data no matter what

Author: Simon Geoffroy-Gagnon
Edit: 17.01.2020
"""
import pandas as pd
import sys
import random
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
import time
import os

import ONN_Setups
import create_datasets as cd 
import setupSimulation as setSim
from saveSimulationData import saveSimData, saveSimSettings, saveAccuracyData, saveSimData, saveNonlin

import digital_NN_main

sys.path.append('/home/simon/Documents/neuroptica')
import neuroptica as neu

# Set random seed to always get same data
rng = 2 
random.seed(rng)

FOLDER = r'MNIST_stepsz=0.005'
setSim.createFOLDER(FOLDER)

N = 4
BATCH_SIZE = 2**6
EPOCHS = 1000
STEP_SIZE = 0.005
SAMPLES = 700
DATASET_NUM = 1
ITERATIONS = 200 # number of times to retry same loss/PhaseUncert
losses_dB = np.linspace(0, 1, 11)
phase_uncerts = np.linspace(0, 0.05, 11)

dataset_name = 'MNIST'
# dataset_name = 'Gauss'
# dataset_name = 'Iris'

ONN_setup = np.array(['C_Q_P', 'R_I_P', 'R_D_P'])

got_accuracy = [0 for _ in range(len(ONN_setup))]

simSettings = N, EPOCHS, STEP_SIZE, SAMPLES, DATASET_NUM, ITERATIONS, losses_dB, phase_uncerts, dataset_name, ONN_setup
saveSimSettings(FOLDER, simSettings)

if 1:
    Nonlinearities = {'a2c0.15_bpReLU2':neu.bpReLU(N, alpha=2, cutoff=0.15), }
    saveNonlin(FOLDER, Nonlinearities)

for ii in range(DATASET_NUM):
    if dataset_name == 'MNIST':
        X, y, Xt, yt = cd.get_data([1,3,6,7], N=N, nsamples=SAMPLES)
    elif dataset_name == 'Gauss':
        X, y, Xt, yt = cd.blob_maker(targets=int(N), features=int(N), nsamples=SAMPLES, random_state=5)
    elif dataset_name == 'Iris':
        X, y, Xt, yt = cd.iris_dataset(nsamples=int(SAMPLES))

    X = (X - np.min(X))/(np.max(X) - np.min(X))
    Xt = (Xt - np.min(Xt))/(np.max(Xt) - np.min(Xt))
    Xog, Xtog = X, Xt

    setSim.saveSimData(FOLDER, dataset_name, ii, N, X, y, Xt, yt)

    for NonLin_key, Nonlinearity in Nonlinearities.items():
        for ONN_Idx, ONN_Model in enumerate(ONN_setup):
            accuracy = []
            for loss in losses_dB:
                acc_array = []
                for phase_uncert in phase_uncerts:
                    t = time.time()
                    print(f'model: {ONN_Model}, Loss = {loss:.3f} dB, Phase Uncert = {phase_uncert:.3f} Rad, dataset = {dataset_name}')

                    model = ONN_Setups.ONN_creation(ONN_Model, N=N)

                    X = Xog
                    Xt = Xtog

                    if 'C' in ONN_Model and 'Q' in ONN_Model:
                        X = np.array([list(np.zeros(int((N-2)))) + list(samples) for samples in X])
                        Xt = np.array([list(np.zeros(int((N-2)))) + list(samples) for samples in Xt])
                    elif 'C' in ONN_Model and 'W' in ONN_Model:
                        X = np.array([list(np.zeros(int((N-2)/2))) + list(samples) + list(np.zeros(np.ceil((N-2)/2)))
                            for samples in X])
                        Xt = np.array([list(np.zeros(int((N-2)/2))) + list(samples) + list(np.zeros(np.ceil((N-2)/2))) 
                            for samples in Xt])

                    # Create phase array
                    phases = model.get_all_phases()
                    model.set_all_phases_uncerts_losses(phases, phase_uncert, loss)

                    # initialize the ADAM optimizer and fit the ONN to the training data
                    optimizer = neu.InSituAdam(model, neu.MeanSquaredError, step_size=STEP_SIZE)

                    currentSimResults  = optimizer.fit(X.T, y.T, Xt.T, yt.T, epochs=EPOCHS, batch_size=BATCH_SIZE, show_progress=True)
                    currentSimSettings = FOLDER, ONN_Model, loss, phase_uncert, N, ii, NonLin_key, dataset_name

                    saveSimData(currentSimSettings, currentSimResults, model)
                    
                    model.set_all_phases_uncerts_losses(currentSimResults[3], phase_uncert, loss)

                    acc = []    
                    for _ in range(ITERATIONS):
                        Y_hat = model.forward_pass(Xt.T)
                        pred = np.array([np.argmax(yhat) for yhat in Y_hat.T])
                        gt = np.array([np.argmax(tru) for tru in yt])
                        acc.append(np.sum(pred == gt)/yt.shape[0]*100)
                    acc_array.append(np.mean(acc))
                    # print(acc_array)
                    print(f'time spent for current training: {(time.time() - t)/60:.2f} minutes')

                accuracy.append(acc_array)

            got_accuracy[ONN_Idx]=1

            currentSimSettings = FOLDER, ONN_Model, losses_dB[0], phase_uncerts[0], N, ii, NonLin_key, dataset_name
            saveAccuracyData(FOLDER, currentSimSettings, accuracy)

# Now test the same dataset using a Digital Neural Networks, just to see the difference between unitary and non-unitary matrix
digital_NN_main.create_train_dnn(X, y, Xt, yt, FOLDER, EPOCHS = 500)
