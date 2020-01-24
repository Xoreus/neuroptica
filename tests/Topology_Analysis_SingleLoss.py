"""
Topology_Analysis_SingleLoss.py
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

sys.path.append('/home/simon/Documents/neuroptica')
import neuroptica as neu

# Set random seed to always get same data
rng = 2 
random.seed(rng)

FOLDER = r'NiceFigures_AdditionalPoints'

setSim.createFOLDER(FOLDER)

N = 4
BATCH_SIZE = 2**6
EPOCHS = 700
STEP_SIZE = 0.0005
SAMPLES = 1000
DATASET_NUM = 1
ITERATIONS = 500 # number of times to retry same loss/PhaseUncert
losses_dB = np.linspace(0, 2, 21) # dB
phase_uncerts = np.linspace(0, 3, 31) # Rad Std dev

# dataset_name = 'MNIST'
# dataset_name = 'Gauss'
dataset_name = 'Iris'

setup = np.array(['R_D_P', 'R_P', 'I_P', 'R_I_P', 'R_D_I_P', 'C_Q_P', 'R_R_R_P','R_R_R_R_P','R_R_R_R_R_P'])

got_accuracy = [0 for _ in range(len(setup))]

if 1:
    " save loss_dB, phase_uncert, ITERATIONS, ONN_Setups, and N "
    np.savetxt(f'{FOLDER}/LossdB.txt', losses_dB, delimiter=',', fmt='%.3f')
    np.savetxt(f'{FOLDER}/PhaseUncert.txt', phase_uncerts, delimiter=',', fmt='%.3f')
    np.savetxt(f'{FOLDER}/ITERATIONS.txt', [ ITERATIONS ], delimiter=',', fmt='%.d')
    np.savetxt(FOLDER+'/ONN_Setups.txt', [x for x in setup], delimiter=" ", fmt="%s")
    np.savetxt(f'{FOLDER}/N.txt', [N])

if 1:
    eo_settings = { 'alpha': 0.2, 'g':0.4 * np.pi, 'phi_b': -1 * np.pi }

    Nonlinearities = {  'a2c0.15_bpReLU2':neu.bpReLU(N, alpha=2, cutoff=0.15)}

    keys = list(Nonlinearities.keys())
    np.savetxt(FOLDER+'/Nonlinearities.txt', keys, delimiter=" ", fmt="%s")

if 1:
    for key, activ in Nonlinearities.items():
        x = np.linspace(0.01, 1, 1000)
        plt.plot(x, np.abs(activ.forward_pass(x)), label=key)
        plt.xlabel("Input field (a.u.)")
        plt.ylabel("Output field (a.u.)")
    plt.legend()
    # plt.show()
    plt.savefig(FOLDER + '/Figures/' + 'nonlin_activation.png')

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

    for NonLin_key, Nonlinearity in Nonlinearities.items():
        for ONN_Idx, ONN_Model in enumerate(setup):
            accuracy = []
            if 'N' in ONN_Model or not got_accuracy[ONN_Idx]:
                t = time.time()
                message = f'model: {ONN_Model}'
                if 'N' in ONN_Model: message += f', Nonlin: {NonLin_key}'
                print(message)

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

                currentSimResults = optimizer.fit(X.T, y.T, Xt.T, yt.T, epochs=EPOCHS, batch_size=BATCH_SIZE, show_progress=True)
                currentSimSettings = FOLDER, ONN_Model, 0, 0, N, ii, NonLin_key, dataset_name

                saveSimData(currentSimSettings, currentSimResults, model)

                # Now calculate the accuracy when adding phase noise and/or mzi loss
                phases = currentSimResults[3]
                for loss in losses_dB:
                    acc_array = []
                    for phase_uncert in phase_uncerts:
                        model.set_all_phases_uncerts_losses(phases, phase_uncert, loss)
                        acc = []    
                        for _ in range(ITERATIONS):
                            Y_hat = model.forward_pass(Xt.T)
                            # print(Y_hat)
                            pred = np.array([np.argmax(yhat) for yhat in Y_hat.T])
                            gt = np.array([np.argmax(tru) for tru in yt])
                            acc.append(np.sum(pred == gt)/yt.shape[0]*100)

                        acc_array.append(np.mean(acc))
                        print(phase_uncert, acc_array[-1])
                    accuracy.append(acc_array)

            np.savetxt(f"{FOLDER}/acc_{ONN_Model}_loss={losses_dB[0]:.3f}_uncert={phase_uncerts[0]:.3f}_{N}Feat_{NonLin_key}_set{ii}.txt", np.array(accuracy).T, delimiter=',', fmt='%.3f')
            got_accuracy[ONN_Idx]=1
