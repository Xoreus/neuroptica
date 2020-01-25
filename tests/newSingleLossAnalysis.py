"""
newSingleLossAnalysis.py
Testing nonlinearities with RI, RDI, R+I, the whole thing. Saves all required files for plotting in matlab (matlab is way better an making nice graphs...), plus its good to save all data no matter what

Author: Simon Geoffroy-Gagnon
Edit: 15.01.2020
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

for rng in [8, 9]:
    random.seed(rng)

    ROOT_FOLDER = r'Analysis/'
    FUNCTION = r'SingleLossAnalysis/'
    FOLDER = ROOT_FOLDER + FUNCTION + f'test{rng}'
    setSim.createFOLDER(FOLDER)

    N = 4
    BATCH_SIZE = 2**6
    EPOCHS = 100
    STEP_SIZE = 0.005
    SAMPLES = 400
    DATASET_NUM = 1
    ITERATIONS = 100 # number of times to retry same loss/PhaseUncert
    loss_diff = 0.0 # +/- dB
    losses_dB = np.linspace(0, 2, 6)
    phase_uncerts = np.linspace(0, 2, 21)

    # dataset_name = 'MNIST'
    dataset_name = 'Gauss'
    # dataset_name = 'Iris'

    ONN_setup = np.array(['R_P', 'R_I_P', 'R_D_I_P', 'R_D_P', 'C_Q_P'])

    got_accuracy = [0 for _ in range(len(ONN_setup))]

    simSettings = N, EPOCHS, STEP_SIZE, SAMPLES, DATASET_NUM, ITERATIONS, losses_dB, phase_uncerts, dataset_name, ONN_setup, loss_diff
    saveSimSettings(FOLDER, simSettings)

    if 1:
        Nonlinearities = {'a2c0.15_bpReLU2':neu.bpReLU(N, alpha=2, cutoff=0.15), }
        saveNonlin(FOLDER, Nonlinearities)

    for ii in range(DATASET_NUM):
        if dataset_name == 'MNIST':
            X, y, Xt, yt = cd.get_data([1,3,6,7], N=N, nsamples=SAMPLES)
        elif dataset_name == 'Gauss':
            X, y, Xt, yt = cd.blob_maker(targets=int(N), features=int(N), nsamples=SAMPLES, random_state=rng)
        elif dataset_name == 'Iris':
            X, y, Xt, yt = cd.iris_dataset(nsamples=int(SAMPLES))

        X = (X - np.min(X))/(np.max(X) - np.min(X))
        Xt = (Xt - np.min(Xt))/(np.max(Xt) - np.min(Xt))
        Xog, Xtog = X, Xt

        setSim.saveSimData(FOLDER, dataset_name, ii, N, X, y, Xt, yt)

        for NonLin_key, Nonlinearity in Nonlinearities.items():
            for ONN_Idx, ONN_Model in enumerate(ONN_setup):
                accuracy = []
                t = time.time()
                print(f'model: {ONN_Model}, Loss = {0:.3f} dB, Phase Uncert = {0:.3f} Rad, dataset = {dataset_name}')

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

                # initialize the ADAM optimizer and fit the ONN to the training data
                optimizer = neu.InSituAdam(model, neu.MeanSquaredError, step_size=STEP_SIZE)

                currentSimResults  = optimizer.fit(X.T, y.T, Xt.T, yt.T, epochs=EPOCHS, batch_size=BATCH_SIZE, show_progress=True)
                currentSimSettings = FOLDER, ONN_Model, 0, 0, N, ii, NonLin_key, dataset_name

                saveSimData(currentSimSettings, currentSimResults, model)
                
                phases = currentSimResults[3]
                for loss_dB in losses_dB:
                    acc_array = []
                    for phase_uncert in phase_uncerts:
                        acc = []    
                        model.set_all_phases_uncerts_losses(phases, phase_uncert, loss_dB, loss_diff)
                        for _ in range(ITERATIONS):
                            Y_hat = model.forward_pass(Xt.T)
                            pred = np.array([np.argmax(yhat) for yhat in Y_hat.T])
                            gt = np.array([np.argmax(tru) for tru in yt])
                            acc.append(np.sum(pred == gt)/yt.shape[0]*100)
                        acc_array.append(np.mean(acc))
                        # print(f'Loss={loss:.6f} dB, Phase Uncert={phase_uncert:.3f} Rad --> Acc = {np.mean(acc)}')
                        # print(phase_uncert, acc_array[-1])

                    accuracy.append(acc_array)

                print(f'time spent for current training and testing all loss/phase uncert: {(time.time() - t)/60:.2f} minutes')
                got_accuracy[ONN_Idx]=1
                saveAccuracyData(FOLDER, currentSimSettings, accuracy)

    # Now test the same dataset using a Digital Neural Networks, just to see the difference between unitary and non-unitary matrix
    # digital_NN_main.create_train_dnn(X, y, Xt, yt, FOLDER, EPOCHS = 500)

    print(f'Simulation in Folder {FOLDER} completed')