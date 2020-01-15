"""
Nonlineatity_analysis_Retrained.py
This time, analyzing if addng a small amnt of noise helps to reduce overfitting (regularization)
also comparing diamond layer to central diamond layer, maybe itle help with loss?
also looks at adding etra MZIs in series (by doing R_P, R_I_P, R_D_I_P, R_D_I_R_D_I_P) to look at effects of increasing loss and/or phase uncert 
Retrains models for every loss/phase_uncert to look at how the loss and uncert dissapear when we train the model
Testing nonlinearities with RI, RDI, R+I, the whole thing. Saves all required files for plotting in matlab (matlab is way better an making nice graphs...), plus its good to save all data no matter what

Author: Simon Geoffroy-Gagnon
Edit: 05.01.2020
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
import plot_scatter_matrix as psm
import SaveSimulationData as SSD
sys.path.append('/home/simon/Documents/neuroptica')
import neuroptica as neu

# Set random seed to always get same data
rng = 2 
random.seed(rng)

FOLDER = r'test_new_code_retrained'

SSD.createFOLDER(FOLDER)

N = 4
BATCH_SIZE = 2**6
EPOCHS = 10 
STEP_SIZE = 0.0005
SAMPLES = 300
DATASET_NUM = 1
ITERATIONS = 100 # number of times to retry same loss/PhaseUncert
losses_dB = np.linspace(0, 1, 3) # dB
phase_uncerts = np.linspace(0, 1, 3) # Rad Std dev

# dataset_name = 'MNIST'
# dataset_name = 'Gauss'
dataset_name = 'Iris'

# setup = np.array(['R_I_P','R_P', 'R_I_PN', 'RI_P_RI_P','C_P_M'])
# setup = np.array(['R_P_I_P', 'C_P_M']) # , 'CC_PM', 'C_P_N_C_PM'])
# setup = np.array(['R_P','R_I_P','R_D_I_P','R_D_I_R_D_I_P'])
# setup = np.array(['C_Q_P', 'C_W_P'])
setup = np.array(['C_Q_P', 'R_I_P', 'R_P'])
# setup = np.array(['R_I_P'])

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

    Nonlinearities = {  'a2c0.15_bpReLU2':neu.bpReLU(N, alpha=2, cutoff=0.15), 
                        # 'a3c0.10_bpReLU1':neu.bpReLU(N, alpha=3, cutoff=0.10), 
                        # 'a4c0.05_bpReLU3':neu.bpReLU(N, alpha=4, cutoff=0.05),
                        # 's0.4s10_sigmoid':neu.SS_Sigmoid(N, Shift=0.4, Squeeze=10), 
                        # 's0.2s30_sigmoid':neu.SS_Sigmoid(N, Shift=0.2, Squeeze=30), 
                        # 's0.1s40_sigmoid':neu.SS_Sigmoid(N, Shift=0.1, Squeeze=40),
                        # 'c0.1_modReLU':neu.modReLU(N, cutoff=0.1),
                        # 'c0.2_modReLU':neu.modReLU(N, cutoff=0.2),
                        # 'c0.07_modReLU':neu.modReLU(N, cutoff=0.07)
                        }

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

    SSD.saveSimData(FOLDER, dataset_name, ii, N, X, y, Xt, yt)

    for NonLin_key, Nonlinearity in Nonlinearities.items():
        for ONN_Idx, ONN_Model in enumerate(setup):
            accuracy = []
            for loss in losses_dB:
                acc_array = []
                for phase_uncert in phase_uncerts:
                    t = time.time()
                    if 'N' in ONN_Model or not got_accuracy[ONN_Idx]:
                        message = f'model: {ONN_Model}, Loss = {loss:.2f} dB, Phase Uncert = {phase_uncert:.2f} Rad'
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

                        # Create phase array
                        phases = model.get_all_phases()
                        model.set_all_phases_uncerts_losses(phases, phase_uncert, loss)

                        # initialize the ADAM optimizer and fit the ONN to the training data
                        optimizer = neu.InSituAdam(model, neu.MeanSquaredError, step_size=STEP_SIZE)
                        losses, trn_accuracy, val_accuracy = optimizer.fit(X.T, y.T, Xt.T, yt.T, epochs=EPOCHS, batch_size=BATCH_SIZE, show_progress=True)

                        # Plot loss, training acc and val acc
                        ax1 = plt.plot()
                        plt.plot(losses, color='b')
                        plt.xlabel('Epoch')
                        plt.ylabel("$\mathcal{L}$", color='b')
                        ax2 = plt.gca().twinx()
                        ax2.plot(trn_accuracy, color='r')
                        ax2.plot(val_accuracy, color='g')
                        plt.ylabel('Accuracy', color='r')
                        plt.legend(['Training Accuracy', 'Validation Accuracy'])
                        plt.title(f'Gradient Descent, Max Validation Accuracy: {max(val_accuracy):.2f}')
                        plt.ylim([0, 100])
                        plt.savefig(f'{FOLDER}/Figures_Fitting/{ONN_Model}_loss={loss:.2f}dB_uncert={phase_uncert:.2f}Rad_{N}Features_#{ii}_{NonLin_key}.png')
                        plt.clf()

                        # save a txt file containing the loss, trn acc, val acc, in case i want to replot it using matlab
                        np.savetxt(f'{FOLDER}/Data_Fitting/{ONN_Model}_loss={loss:.2f}dB_uncert={phase_uncert:.2f}Rad_{N}Features_#{ii}_{NonLin_key}.txt',np.array(
                            [losses, trn_accuracy, val_accuracy]).T, delimiter=',', fmt='%.4f')
                        
                        # Create phase array
                        phases = model.get_all_phases()
                        phases_flat = [item for sublist in phases for item in sublist]
                        df = pd.DataFrame(phases_flat, columns=['Theta','Phi'])
                        df.to_csv(f'{FOLDER}/Phases/Phases_{ONN_Model}_loss={loss:.2f}dB_uncert={phase_uncert:.2f}Rad_{N}Features_#{ii}_{NonLin_key}.txt')

                        acc = []    
                        for _ in range(ITERATIONS):
                            Y_hat = model.forward_pass(Xt.T)
                            pred = np.array([np.argmax(yhat) for yhat in Y_hat.T])
                            gt = np.array([np.argmax(tru) for tru in yt])
                            acc.append(np.sum(pred == gt)/yt.shape[0]*100)
                        acc_array.append(np.mean(acc))
                        print(f'time spent for current training: {(time.time() - t)/60:.2f} minutes')

                accuracy.append(acc_array)

            got_accuracy[ONN_Idx]=1
            np.savetxt(f"{FOLDER}/acc_{ONN_Model}_loss={0:.2f}_uncert={0:.2f}_{N}Feat_{NonLin_key}_set{ii}.txt", np.array(accuracy).T, delimiter=',', fmt='%.3f')


