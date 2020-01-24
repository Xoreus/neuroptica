"""
Nonlineatity_analysis.py
Testing nonlinearities with RI, RDI, R+I, the whole thing. Saves all required files for plotting in matlab (matlab is way better an making nice graphs...), plus its good to save all data no matter what

Author: Simon Geoffroy-Gagnon
Edit: 2019.12.09
"""
import pandas as pd
from sklearn.datasets.samples_generator import make_blobs
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
import PCA_MNIST as mnist
import plot_scatter_matrix as psm
import iris_fourth_flower as iris

# Set random seed to always get same data
rng = 6 
random.seed(rng)

sys.path.append('/home/simon/Documents/neuroptica')
import neuroptica as neu

def blob_maker(targets=4, features=4, nsamples=10000,
               cluster_std=.1, random_state=rng):

    # generate 2d classification dataset
    X, y = make_blobs(n_samples=nsamples, centers=targets,
                      n_features=features,
                      cluster_std=cluster_std,
                      center_box=(0, 1), shuffle=False, random_state=random_state)
    ohe_labels = pd.get_dummies(y).values
    x, xt, y, yt = train_test_split(X, ohe_labels, test_size=0.2)

    return x, y, xt, yt

MNIST = False
N = 4
BATCH_SIZE = 2**6
EPOCHS = 200
STEP_SIZE = 0.0005
FOLDER = r'Diamond_tests_Comparison_RP/'
SAMPLES = 5000
DATASET_NUM = 1
ITERATIONS = 100 # number of times to retry same loss/PhaseUncert
losses_dB = np.linspace(0,2,11) # dB
phase_uncerts = np.linspace(0, 2, 21) 

dataset_name = 'MNIST'
dataset_name == 'Gauss'
dataset_name = 'Iris'

if not os.path.isdir(FOLDER):
    os.makedirs(FOLDER + 'Figures')

# setup = np.array(['R_I_P','R_R_P', 'R_I_PN', 'R_R_PN', 'RI_P_RI_P', 'RR_P_RR_P'])
# setup = np.array(['C_P_M']) # , 'CC_PM', 'C_P_N_C_PM'])
setup = np.array(['R_P'])

got_accuracy = [0 for _ in range(len(setup))]

if 1:
    " save loss_dB and phase_uncert too "
    np.savetxt(f'{FOLDER}LossdB_{N}Features.txt', losses_dB, delimiter=',', fmt='%.3f')
    np.savetxt(f'{FOLDER}PhaseUncert{N}Features.txt', phase_uncerts, delimiter=',', fmt='%.3f')
    np.savetxt(f'{FOLDER}ITERATIONS.txt', [ ITERATIONS ], delimiter=',', fmt='%.d')
    np.savetxt(FOLDER+'ONN_Setups.txt', [x for x in setup], delimiter=" ", fmt="%s")

if 1:
    eo_settings = { 'alpha': 0.2, 'g':0.4 * np.pi, 'phi_b': -1 * np.pi }

    Nonlinearities = {  'a2c0.15_bpReLU2':neu.bpReLU(N, alpha=2, cutoff=0.15), 
                        'a3c0.10_bpReLU1':neu.bpReLU(N, alpha=3, cutoff=0.10), 
                        'a4c0.05_bpReLU3':neu.bpReLU(N, alpha=4, cutoff=0.05),
                        's0.4s10_sigmoid':neu.SS_Sigmoid(N, Shift=0.4, Squeeze=10), 
                        's0.2s30_sigmoid':neu.SS_Sigmoid(N, Shift=0.2, Squeeze=30), 
                        's0.1s40_sigmoid':neu.SS_Sigmoid(N, Shift=0.1, Squeeze=40),
                        'c0.1_modReLU':neu.modReLU(N, cutoff=0.1),
                        'c0.2_modReLU':neu.modReLU(N, cutoff=0.2),
                        'c0.07_modReLU':neu.modReLU(N, cutoff=0.07)
                        }

    keys = list(Nonlinearities.keys())
    np.savetxt(FOLDER+'Nonlinearities.txt', keys, delimiter=" ", fmt="%s")

if 1:
    for key, activ in Nonlinearities.items():
        x = np.linspace(0.01, 1, 1000)
        plt.plot(x, np.abs(activ.forward_pass(x)), label=key)
        plt.xlabel("Input field (a.u.)")
        plt.ylabel("Output field (a.u.)")
    plt.legend()
    # plt.show()
    plt.savefig(FOLDER + 'Figures/' + 'nonlin_activation.png')
    
for ii in range(DATASET_NUM):
    if dataset_name == 'MNIST':
        X, y, Xt, yt, *_ = mnist.get_data([1,3,6,7], N=N)
    elif dataset_name == 'Gauss':
        X, y, Xt, yt = blob_maker(targets=int(N), features=int(N))
    elif dataset_name == 'Iris':
        X, y, Xt, yt = iris.iris_dataset(augment=int(SAMPLES/4))


    rand_ind = random.sample(list(range(len(X))), int(SAMPLES*0.8))
    X = X[rand_ind]
    y = y[rand_ind]
    rand_ind = random.sample(list(range(len(Xt))), int(SAMPLES*0.8/10))
    Xt = Xt[rand_ind]
    yt = yt[rand_ind]

    X = (X - np.min(X))/(np.max(X) - np.min(X))
    axes = psm.plot_scatter_matrix(X, y)
    plt.savefig(f'{FOLDER}Figures/{dataset_name}_Dataset#{ii}.png')
    plt.clf()
    mpl.rcParams.update(mpl.rcParamsDefault)

    np.savetxt(f'{FOLDER}{dataset_name}_X{N}Features{ii}.txt',X, delimiter=',',fmt='%.3f')
    np.savetxt(f'{FOLDER}{dataset_name}_y{N}Features{ii}.txt',y, delimiter=',',fmt='%.3f')
    np.savetxt(f'{FOLDER}{dataset_name}_Xt{N}Features{ii}.txt',Xt, delimiter=',',fmt='%.3f')
    np.savetxt(f'{FOLDER}{dataset_name}_yt{N}Features{ii}.txt',yt, delimiter=',',fmt='%.3f')
    
    for NonLin_key, Nonlinearity in Nonlinearities.items():
        for ONN_Idx, ONN_Model in enumerate(setup):
            if 'N' in ONN_Model or not got_accuracy[ONN_Idx]:
                message = f'model: {ONN_Model}'
                if 'N' in ONN_Model: message += f', Nonlin: {NonLin_key}'
                print(message)

                accuracy = []

                model = ONN_Setups.ONN_creation(ONN_Model, N=N)

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
                plt.title(f'Gradient Descent, Final Validation Accuracy: {val_accuracy[-1]:.2f}')
                plt.ylim([0, 100])
                # plt.show()
                plt.savefig(f'{FOLDER}{ONN_Model}_loss=0dB_uncert=0Rad_{N}Features_#{ii}_{NonLin_key}.pdf')
                plt.clf()

                # save a txt file containing the loss, trn acc, val acc, in case i want to replot it using matlab
                np.savetxt(f'{FOLDER}{ONN_Model}_loss=0dB_uncert=0Rad_{N}Features_#{ii}_{NonLin_key}.txt',np.array([losses, trn_accuracy, val_accuracy]).T, delimiter=',', fmt='%.4f')

                # Create phase array
                phases = []
                for layer in model.layers:
                    if hasattr(layer, 'mesh'):
                        phases.append([x for x in layer.mesh.all_tunable_params()])
                phases_flat = [item for sublist in phases for item in sublist]
                df = pd.DataFrame(phases_flat, columns=['Theta','Phi'])
                df.to_csv(f'{FOLDER}phases_for_{ONN_Model}_#{ii}.txt')

                # Now calculate the accuracy when adding phase noise and/or mzi loss
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

                    accuracy.append(acc_array)


                np.savetxt(f'{FOLDER}accuracy_{ONN_Model}_{N}Features_#{ii}_{NonLin_key}.txt', np.array(accuracy).T, delimiter=',', fmt='%.3f')
            got_accuracy[ONN_Idx]=1
