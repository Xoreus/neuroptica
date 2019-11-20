"""
Testing nonlinearities with RI, RDI, R+I, the whole thing. Saves all required files for plotting in matlab (matlab is way better an making nice graphs...), plus its good to save all data no matter what
Author: Simon Geoffroy-Gagnon
Edit: 17.11.2019
"""
import pandas as pd
from sklearn.datasets.samples_generator import make_blobs
import sys
import random
import csv
import numpy as np
import PCA_MNIST as mnist
import plot_scatter_matrix as psm
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
import time
import os
import ONN_Setups


# Set random seed to always get same data
rng = 6 
random.seed(rng)

sys.path.append('/home/simon/Documents/neuroptica')
import neuroptica as neu

def blob_maker(targets=4, features=4, nsamples=10000,
               cluster_std=.1):

    # generate 2d classification dataset
    X, y = make_blobs(n_samples=nsamples, centers=targets,
                      n_features=features,
                      cluster_std=cluster_std,
                      center_box=(0, 1), shuffle=False)
    ohe_labels = pd.get_dummies(y).values
    x, xt, y, yt = train_test_split(X, ohe_labels, test_size=0.2)

    return x, y, xt, yt

def printf(format, *args):
    sys.stdout.write(format % args)

# Number of input features?
N = 4
BATCH_SIZE = 2**5
EPOCHS = 140
STEP_SIZE = 0.001
FOLDER = r'nonlinearity_MNIST_analysis_additional_tests/'
SAMPLES = 2000
DATASET_NUM = 0
ITERATIONS = 200# number of times to retry same loss/PhaseUncert
losses_dB = np.linspace(0,1,2)# in dB
phase_uncerts = np.linspace(0, 1, 21) 


# Check if FOLDER is already created; create it if not
if not os.path.isdir(FOLDER):
    os.mkdir(FOLDER)

# 0: R, 1: RI, 2: RDI, 3: RD, 4: R nl I, 5: RI nl RI, 6: RDI nl RDI 
setup = ['RP', 'RIP', 'RDIP', 'RDP', 'RDI_RDI', 'RI_RI', 'RNI','RINRI', 'RDI_N_RDI_P', 'R_N_I_R_N_I','RI_N_RI_N_RI', 'RDI_N_RDI_N_RDI']
includes_Nonlin = [0,0,0,0,0,0,1,1,1,1,1,1]
got_accuracy = [1 for _ in range(len(_Nonlin))]

# save loss_dB and phase_uncert too
np.savetxt(f'{FOLDER}LossdB_{N}Features.txt', losses_dB, delimiter=',', fmt='%.3f')
np.savetxt(f'{FOLDER}PhaseUncert{N}Features.txt', phase_uncerts, delimiter=',', fmt='%.3f')
np.savetxt(f'{FOLDER}ITERATIONS.txt', [ ITERATIONS ], delimiter=',', fmt='%.d')
np.savetxt(FOLDER+'ONN_Setups.txt', [x for x in setup], delimiter=" ", fmt="%s")

eo_settings = { 'alpha': 0.2,
                'g':     0.4 * np.pi,
                'phi_b': -1 * np.pi }

Nonlinearities = {'bpReLU1':neu.bpReLU(N, alpha=1, cutoff=0.1), 'bpReLU2':neu.bpReLU(N, alpha=1, cutoff=0.15),'AbsSquared':neu.AbsSquared(N), 'Sigmoid':neu.Sigmoid(N)}
keys = list(Nonlinearities.keys())
np.savetxt(FOLDER+'Nonlinearities.txt', keys, delimiter=" ", fmt="%s")

if 0:
    eo_activation = neu.ElectroOpticActivation(1, **eo_settings)
    eo_activation = neu.Squeezed_SoftMax(1)
    eo_activation = neu.bpReLU(1, alpha=1, cutoff=0.25)
    # eo_activation = neu.modReLU(1, cutoff=0.05)
    # eo_activation = neu.Sigmoid(1)
    # eo_activation = neu.AbsSquared(1)
    # eo_activation = neu.cReLU(1)
    # eo_activation = neu.Squeezed_SoftMax(1, squeeze=5)


    x = np.linspace(0.01, 1, 100)
    plt.plot(x, np.real(eo_activation.forward_pass(x)),label="Re")
    plt.plot(x, np.imag(eo_activation.forward_pass(x)),label="Im")
    plt.plot(x, np.abs(eo_activation.forward_pass(x)), label="Abs")
    plt.xlabel("Input field (a.u.)")
    plt.ylabel("Output field (a.u.)")
    plt.legend()
    plt.show()
    # plt.savefig('eo_activation.png')
    
for ii in range(DATASET_NUM):

    X, y, Xt, yt, *_ = mnist.get_data([1,3,6,7], N=N)
    rand_ind = random.sample(list(range(len(X))), SAMPLES)
    X = X[rand_ind]
    y = y[rand_ind]
    rand_ind = random.sample(list(range(len(Xt))), int(SAMPLES/10))
    Xt = Xt[rand_ind]
    yt = yt[rand_ind]

    # X, y, Xt, yt = blob_maker(targets=N, features=N, nsamples=SAMPLES)
    # Normalize inputs
    X = (X - np.min(X))/(np.max(X) - np.min(X))
    # axes = psm.plot_scatter_matrix(X, y)
    # plt.savefig(f'{FOLDER}Dataset#{ii}.png')
    # plt.clf()
    # mpl.rcParams.update(mpl.rcParamsDefault)

    np.savetxt(f'{FOLDER}X{N}Features{ii}.txt',X, delimiter=',',fmt='%.3f')
    np.savetxt(f'{FOLDER}y{N}Features{ii}.txt',y, delimiter=',',fmt='%.3f')
    np.savetxt(f'{FOLDER}Xt{N}Features{ii}.txt',Xt, delimiter=',',fmt='%.3f')
    np.savetxt(f'{FOLDER}yt{N}Features{ii}.txt',yt, delimiter=',',fmt='%.3f')

    for NonLin_key, Nonlinearity in Nonlinearities.items():
        for ONN_Idx, ONN_Model in enumerate(setup):
            if includes_Nonlin[ONN_Idx] or got_accuracy[ONN_Idx]:
                print(f'model: {ONN_Model}, Nonlin: {NonLin_key}')
                accuracy = []
                # First train the network with 0 phase noise and 0 loss
                loss = 0
                phase_uncert = 0 

                model = ONN_Setups.ONN_creation(ONN_Model)

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
                plt.savefig(f'{FOLDER}{ONN_Model}_loss={loss:.2f}_uncert={phase_uncert:.3f}_{N}Features_#{ii}_{NonLin_key}.pdf')
                plt.clf()

                # save a txt file containing the loss, trn acc, val acc, in case i want to replot it using matlab
                np.savetxt(f'{FOLDER}{ONN_Model}_loss={loss}_uncert={phase_uncert:.3f}_{N}Features_#{ii}_{NonLin_key}.txt',np.array([losses, trn_accuracy, val_accuracy]).T, delimiter=',', fmt='%.4f')

                # Create phase array
                phases = []
                for layer in model.layers:
                    if hasattr(layer, 'mesh'):
                        phases.append([x for x in layer.mesh.all_tunable_params()])
                #print(phases)
                phases_flat = [item for sublist in phases for item in sublist]

                df = pd.DataFrame(phases_flat, columns=['Theta','Phi'])
                # print(df)
                df.to_csv(f'{FOLDER}phases_for_{ONN_Model}_#{ii}.txt')

                for loss in losses_dB:
                    # Now calculate the accuracy when adding phase noise and/or mzi loss
                    acc_array = []
                    for phase_uncert in phase_uncerts:
                        # print(f'loss = {loss:.2f} dB, Phase Uncert = {phase_uncert:.2f}')
                        model = ONN_Setups.ONN_creation(ONN_Model, N, loss, phase_uncert, Nonlinearity, phases)

                        acc = []    
                        # calculate validation accuracy. Since phase is shifted randomly after iteration, this is good
                        for _ in range(ITERATIONS):
                            Y_hat = model.forward_pass(Xt.T)
                            pred = np.array([np.argmax(yhat) for yhat in Y_hat.T])
                            gt = np.array([np.argmax(tru) for tru in yt])
                            acc.append(np.sum(pred == gt)/len(Xt)*100)

                        acc_array.append(np.mean(acc))

                    accuracy.append(acc_array)

                # save the accuracies of the current model. will be a 2D array for accuracies at every loss_dB and phase_uncert
                np.savetxt(f'{FOLDER}accuracy_{ONN_Model}_{N}Features_#{ii}_{NonLin_key}.txt', np.array(accuracy).T, delimiter=',', fmt='%.3f')
            got_accuracy[ONN_Idx]=0

