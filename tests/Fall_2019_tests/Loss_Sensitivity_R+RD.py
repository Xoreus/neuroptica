"""
Testing the inverted reck mesh, single MZI
Author: Simon Geoffroy-Gagnon
Edit: 07.11.2019
"""
import pandas as pd
from sklearn.datasets.samples_generator import make_blobs
import sys
import random
import numpy as np
import plot_scatter_matrix as psm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
import os

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
EPOCHS = 700
STEP_SIZE = 0.001
FOLDER = 'loss_sensitivity_R+RD/'
SAMPLES = 1000

# Check if FOLDER is already created; create it if not
if not os.path.isdir(FOLDER):
    os.mkdir(FOLDER)

for ii in range(20):
    X, y, Xt, yt = blob_maker(targets=N, features=N, nsamples=SAMPLES)
    # Normalize inputs
    X = (X - np.min(X))/(np.max(X) - np.min(X))
    axes = psm.plot_scatter_matrix(X, y)
    plt.clf()
    plt.savefig(f'{FOLDER}Dataset#{ii}.png')
    plt.clf()
    losses_dB = np.linspace(0,3,31)# in dB
    iterations = 2 # number of times to retry same loss
    phase_uncerts = np.linspace(0, 0, 1) 

    # save loss_dB and phase_uncert too
    np.savetxt(f'{FOLDER}LossdB_{N}Features{ii}.txt',losses_dB, delimiter=',',fmt='%.3f')
    np.savetxt(f'{FOLDER}PhaseUncert{N}Features{ii}.txt',phase_uncerts, delimiter=',',fmt='%.3f')

    setup = ['Reck', 'invReck', 'Reck+invReck', 'Reck+DMM+invReck', 'Reck+DMM']
    if 1:
        RIdx = [f'MZI{ONN_Model}' for ONN_Model in range(int(N*(N-1)/2))]
        iIdx = [f'Inverted MZI{ONN_Model}' for ONN_Model in range(int(N*(N-1)/2))]
        RiRIdx =  ( [f'MZI{ONN_Model}' for ONN_Model in range(int(N*(N-1)/2))] +
                    [f'Inverted MZI{ONN_Model}' for ONN_Model in range(int(N*(N-1)/2), int(N*(N-1)))])

        RDiRIdx = ( [f'MZI{ONN_Model}' for ONN_Model in range(int(N*(N-1)/2))] +
                    [f'DMM MZI{ONN_Model}' for ONN_Model in range(int(N*(N-1)/2),int(N*(N-1)/2+N))] +
                    [f'Inverted MZI{ONN_Model}' for ONN_Model in range(int(N+N*(N-1)/2), int(N+N*(N-1)))])

        RDIdx = [f'MZI{ONN_Model}' for ONN_Model in range(int(N*(N-1)/2))] + [f'DMM MZI{ONN_Model}' for ONN_Model in range(int(N*(N-1)/2),int(N*(N-1)/2+N))]

        setupIdx = [RIdx, iIdx, RiRIdx, RDiRIdx, RDIdx]

    for ONN_Model in [3]: 
        print(f'model{ONN_Model}')
        accuracy = []
        loss = 0
        # First train the network with 0 phase noise
        phase_uncert = 0 
        if ONN_Model == 0: # Reck
            model = neu.Sequential([
                neu.ReckLayer(N, include_phase_shifter_layer=False, thetas=[None]*int(N*(N-1)/2), phis=[None]*int(N*(N-1)/2), loss=loss, phase_uncert=phase_uncert),

                # neu.AddMask(2*N),
                # neu.DMM_layer(2*N, thetas=[None]*int(N), phis=[None]*int(N)),
                # neu.DropMask(N=2*N, keep_ports=range(0, 2*N, 2)), # makes the DMM thetas pi
                # neu.DropMask(N=2*N, keep_ports=range(1, 2*N+1, 2)), # makes the DMM thetas 0 or 2*pi

                # neu.flipped_ReckLayer(N, include_phase_shifter_layer=False, thetas=[None]*int(N*(N-1)/2), phis=[None]*int(N*(N-1)/2), loss=loss),

                neu.Activation(neu.AbsSquared(N)), # photodetector measurement
            ])
        if ONN_Model == 1: # Inverted Reck
            model = neu.Sequential([
                # neu.ReckLayer(N, include_phase_shifter_layer=False, thetas=[None]*int(N*(N-1)/2), phis=[None]*int(N*(N-1)/2), loss=loss, phase_uncert=phase_uncert),

                # neu.AddMask(2*N),
                # neu.DMM_layer(2*N, thetas=[None]*int(N), phis=[None]*int(N)),
                # neu.DropMask(N=2*N, keep_ports=range(0, 2*N, 2)), # makes the DMM thetas pi
                # neu.DropMask(N=2*N, keep_ports=range(1, 2*N+1, 2)), # makes the DMM thetas 0 or 2*pi

                neu.flipped_ReckLayer(N, include_phase_shifter_layer=False, thetas=[None]*int(N*(N-1)/2), phis=[None]*int(N*(N-1)/2), loss=loss),

                neu.Activation(neu.AbsSquared(N)), # photodetector measurement
            ])
        if ONN_Model == 2: # Reck + Inverted Reck
            model = neu.Sequential([
                neu.ReckLayer(N, include_phase_shifter_layer=False, thetas=[None]*int(N*(N-1)/2), phis=[None]*int(N*(N-1)/2), loss=loss, phase_uncert=phase_uncert),

                # neu.AddMask(2*N),
                # neu.DMM_layer(2*N, thetas=[None]*int(N), phis=[None]*int(N)),
                # neu.DropMask(N=2*N, keep_ports=range(0, 2*N, 2)), # makes the DMM thetas pi
                # neu.DropMask(N=2*N, keep_ports=range(1, 2*N+1, 2)), # makes the DMM thetas 0 or 2*pi

                neu.flipped_ReckLayer(N, include_phase_shifter_layer=False, thetas=[None]*int(N*(N-1)/2), phis=[None]*int(N*(N-1)/2), loss=loss),

                neu.Activation(neu.AbsSquared(N)), # photodetector measurement
            ])
        if ONN_Model == 3: # Reck + DMM + Inverted Reck
            model = neu.Sequential([
                neu.ReckLayer(N, include_phase_shifter_layer=False, thetas=[None]*int(N*(N-1)/2), phis=[None]*int(N*(N-1)/2), loss=loss, phase_uncert=phase_uncert),

                neu.AddMask(2*N),
                neu.DMM_layer(2*N, thetas=[None]*int(N), phis=[None]*int(N)),
                # neu.DropMask(N=2*N, keep_ports=range(0, 2*N, 2)), # makes the DMM thetas pi
                neu.DropMask(N=2*N, keep_ports=range(1, 2*N+1, 2)), # makes the DMM thetas 0 or 2*pi

                neu.flipped_ReckLayer(N, include_phase_shifter_layer=False, thetas=[None]*int(N*(N-1)/2), phis=[None]*int(N*(N-1)/2), loss=loss),

                neu.Activation(neu.AbsSquared(N)), # photodetector measurement
            ])
        if ONN_Model == 4:# Reck + DMM
            model = neu.Sequential([
                neu.ReckLayer(N, include_phase_shifter_layer=False, thetas=[None], phis=[None], loss=loss, phase_uncert=phase_uncert),

                neu.AddMask(2*N),
                neu.DMM_layer(2*N, thetas=[None], phis=[None], phase_uncert=phase_uncert, loss=loss),
                # neu.DropMask(N=2*N, keep_ports=range(0, 2*N, 2)), # makes the DMM thetas pi
                neu.DropMask(N=2*N, keep_ports=range(1, 2*N+1, 2)), # makes the DMM thetas 0 or 2*pi

                # neu.flipped_ReckLayer(N, include_phase_shifter_layer=False, thetas=thetas[-int(N*(N-1)/2):], phis=phis[-int(N*(N-1)/2):], loss=loss, phase_uncert=phase_uncert),

                neu.Activation(neu.AbsSquared(N)), # photodetector measurement
            ])

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
        plt.savefig(f'{FOLDER}{setup[ONN_Model]}_loss={loss}_uncert={phase_uncert:.3f}{N}Features{ii}.pdf')
        plt.clf()

        # save a txt file containing the loss, trn acc, val acc, in case i want to replot it using matlab
        np.savetxt(f'{FOLDER}{setup[ONN_Model]}_loss={loss}_uncert={phase_uncert:.3f}{N}Features{ii}.txt',np.array([losses, trn_accuracy, val_accuracy]).T, delimiter=',', fmt='%.4f')

        # Create phase array
        phases = []
        for layer in model.layers:
            if hasattr(layer, 'mesh'):
                phases.append([x for x in layer.mesh.all_tunable_params()])

        # separate phase array into theta and phi
        thetas = [item for sublist in phases for (item, _) in sublist]
        phis = [item for sublist in phases for (_, item) in sublist]

        df = pd.DataFrame(np.array([thetas, phis]).T, columns=['Theta','Phi'],  index=setupIdx[ONN_Model])
        print(df)
        df.to_csv(f'{FOLDER}phases_for_{setup[ONN_Model]}{ii}.txt')

        for loss in losses_dB:
            # print(f'loss = {loss} dB')

            # Now calculate the accuracy when adding phase noise and/or mzi loss
            acc_array = []
            for phase_uncert in phase_uncerts:
                print(f'loss = {loss:.2f} dB, Phase Uncert = {phase_uncert:.2f}')
                if ONN_Model == 0:
                    model = neu.Sequential([
                        neu.ReckLayer(N, include_phase_shifter_layer=False, thetas=thetas, phis=phis, loss=loss, phase_uncert=phase_uncert),

                        # neu.AddMask(2*N),
                        # neu.DMM_layer(2*N, thetas=[None]*int(N), phis=[None]*int(N)),
                        # neu.DropMask(N=2*N, keep_ports=range(0, 2*N, 2)), # makes the DMM thetas pi
                        # neu.DropMask(N=2*N, keep_ports=range(1, 2*N+1, 2)), # makes the DMM thetas 0 or 2*pi

                        # neu.flipped_ReckLayer(N, include_phase_shifter_layer=False, thetas=[None]*int(N*(N-1)/2), phis=[None]*int(N*(N-1)/2), loss=loss),

                        neu.Activation(neu.AbsSquared(N)), # photodetector measurement
                    ])
                if ONN_Model == 1:
                    model = neu.Sequential([
                        # neu.ReckLayer(N, include_phase_shifter_layer=False, thetas=[None]*int(N*(N-1)/2), phis=[None]*int(N*(N-1)/2), loss=loss, phase_uncert=phase_uncert),

                        # neu.AddMask(2*N),
                        # neu.DMM_layer(2*N, thetas=[None]*int(N), phis=[None]*int(N)),
                        # neu.DropMask(N=2*N, keep_ports=range(0, 2*N, 2)), # makes the DMM thetas pi
                        # neu.DropMask(N=2*N, keep_ports=range(1, 2*N+1, 2)), # makes the DMM thetas 0 or 2*pi

                        neu.flipped_ReckLayer(N, include_phase_shifter_layer=False, thetas=thetas, phis=phis, loss=loss, phase_uncert=phase_uncert),

                        neu.Activation(neu.AbsSquared(N)), # photodetector measurement
                    ])
                if ONN_Model == 2:
                    model = neu.Sequential([
                        neu.ReckLayer(N, include_phase_shifter_layer=False, thetas=thetas[:int(N*(N-1)/2)], phis=phis[:int(N*(N-1)/2)], loss=loss, phase_uncert=phase_uncert),

                        # neu.AddMask(2*N),
                        # neu.DMM_layer(2*N, thetas=[None]*int(N), phis=[None]*int(N)),
                        # neu.DropMask(N=2*N, keep_ports=range(0, 2*N, 2)), # makes the DMM thetas pi
                        # neu.DropMask(N=2*N, keep_ports=range(1, 2*N+1, 2)), # makes the DMM thetas 0 or 2*pi

                        neu.flipped_ReckLayer(N, include_phase_shifter_layer=False, thetas=thetas[int(N*(N-1)/2):], phis=phis[int(N*(N-1)/2):], loss=loss, phase_uncert=phase_uncert),

                        neu.Activation(neu.AbsSquared(N)), # photodetector measurement
                    ])
                if ONN_Model == 3:
                    model = neu.Sequential([
                        neu.ReckLayer(N, include_phase_shifter_layer=False, thetas=thetas[:int(N*(N-1)/2)], phis=phis[:int(N*(N-1)/2)], loss=loss, phase_uncert=phase_uncert),

                        neu.AddMask(2*N),
                        neu.DMM_layer(2*N, thetas=thetas[int(N*(N-1)/2):int(N*(N-1)/2)+int(N)], phis=phis[int(N*(N-1)/2):int(N*(N-1)/2)+int(N)], phase_uncert=phase_uncert, loss=loss),
                        # neu.DropMask(N=2*N, keep_ports=range(0, 2*N, 2)), # makes the DMM thetas pi
                        neu.DropMask(N=2*N, keep_ports=range(1, 2*N+1, 2)), # makes the DMM thetas 0 or 2*pi

                        neu.flipped_ReckLayer(N, include_phase_shifter_layer=False, thetas=thetas[-int(N*(N-1)/2):], phis=phis[-int(N*(N-1)/2):], loss=loss, phase_uncert=phase_uncert),

                        neu.Activation(neu.AbsSquared(N)), # photodetector measurement
                    ])
                if ONN_Model == 4:
                    model = neu.Sequential([
                        neu.ReckLayer(N, include_phase_shifter_layer=False, thetas=thetas[:int(N*(N-1)/2)], phis=phis[:int(N*(N-1)/2)], loss=loss, phase_uncert=phase_uncert),

                        neu.AddMask(2*N),
                        neu.DMM_layer(2*N, thetas=thetas[int(N*(N-1)/2):int(N*(N-1)/2)+int(N)], phis=phis[int(N*(N-1)/2):int(N*(N-1)/2)+int(N)], phase_uncert=phase_uncert, loss=loss),
                        # neu.DropMask(N=2*N, keep_ports=range(0, 2*N, 2)), # makes the DMM thetas pi
                        neu.DropMask(N=2*N, keep_ports=range(1, 2*N+1, 2)), # makes the DMM thetas 0 or 2*pi

                        # neu.flipped_ReckLayer(N, include_phase_shifter_layer=False, thetas=thetas[-int(N*(N-1)/2):], phis=phis[-int(N*(N-1)/2):], loss=loss, phase_uncert=phase_uncert),

                        neu.Activation(neu.AbsSquared(N)), # photodetector measurement
                    ])

                acc = []    
                # calculate validation accuracy. Since phase is shifted randomly after iteration, this is good
                for _ in range(iterations):
                    Y_hat = model.forward_pass(Xt.T)
                    pred = np.array([np.argmax(yhat) for yhat in Y_hat.T])
                    gt = np.array([np.argmax(tru) for tru in yt])
                    acc.append(np.sum(pred == gt)/len(Xt)*100)
                acc_array.append(np.mean(acc))

            accuracy.append(acc_array)

        # save the accuracies of the current model. will be a 2D array for accuracies at every loss_dB and phase_uncert
        np.savetxt(f'{FOLDER}accuracy_{setup[ONN_Model]}{N}Features{ii}.txt', np.array(accuracy).T, delimiter=',', fmt='%.3f')

