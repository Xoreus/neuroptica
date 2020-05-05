"""
digital_NN_main.py
Simple single layer fully connected neural network, used to test whether or not the unitary-ness of
ONNs actually affect accuracy

Author: Simon Geoffroy-Gagnon
Edit: 29.01.2020
"""
import sys
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import create_datasets as cd
sys.path.append('/home/simon/Documents/neuroptica/tests/neural_network_digital')
sys.path.append(r'C:\Users\sgeoff1\Documents\neuroptica_PhiTheta\tests\neural_network_digital')
import Digital_Neural_Network as dnn
import neural_network as nn
import matplotlib.pyplot as plt
import random

def get_current_accuracy(xtst, ytst, net):
    correct_pred = 0
    for idx, x_curr in enumerate(xtst):
        net.setInput(x_curr)
        net.feedForward()
        if all(ytst[idx] == net.getThResults()):
            correct_pred += 1
    return correct_pred/len(xtst)

def create_train_dnn(X, y, Xt, yt, EPOCHS=300, hnum=0):

    h_num = hnum
    beta = 1
    eta = .05
    alpha = 0.1

    val_accuracy = []

    trn_accuracy = []
    losses = []

    # Build network
    net = dnn.build_network(X, y, h_num=h_num)

    # Train model with Gradient Descent
    for epoch in range(EPOCHS):
        err = 0
        # Shuffle data at the start of each epoch
        instances, targets = shuffle(X, y)
        nn.Neuron.alpha = alpha
        nn.Neuron.beta = beta
        for ii in range(len(instances)):
            nn.Neuron.eta = eta/(1+epoch*beta)
            net.setInput(instances[ii])
            net.feedForward()
            net.backPropagate(targets[ii])
            err = err + net.getError(targets[ii])

        losses.append(err)
        val_accuracy.append(get_current_accuracy(Xt, yt, net)*100)
        trn_accuracy.append(get_current_accuracy(X, y, net)*100)

        if not np.mod(epoch, 10):
            print("error: ", err)

    print('Done Epochs')

    # Get network weights
    weights = net.getWeights()
    return net, weights

if __name__ == '__main__':
    import os
    SAMPLES = 300
    rng = 8
    EPOCHS = 30
    for N in [4]:
        # for rng in range(100, 200):
        ii = 0
        for rng in [2]:
            FOLDER = f'Analysis/DNN/Digital_Neural_Network_{SAMPLES*N}_{rng}_N={N}'
            print(f'RNG = {rng}, N = {N}')
            X, y, Xt, yt = cd.iris_dataset(nsamples=1000)
            random.seed(rng)

            X = (X - np.min(X))/(np.max(X) - np.min(X))
            Xt = (Xt - np.min(Xt))/(np.max(Xt) - np.min(Xt))
            Xog, Xtog = X, Xt

            net, weights = create_train_dnn(X, y, Xt, yt, EPOCHS, hnum=0)

                    # Val Accuracy
            print('Validation Accuracy: {:.1f}%'.format(get_current_accuracy(
                    Xt, yt, net)*100))

            if get_current_accuracy(Xt, yt, net)*100 > 50:
                datasetFolder = f'../linsep-datasets/N={N}'
                if not os.path.isdir(datasetFolder):
                    os.makedirs(datasetFolder)

                np.savetxt(f'{datasetFolder}/X_{ii}.txt', X, delimiter=',', fmt='%.6f')
                np.savetxt(f'{datasetFolder}/Xt_{ii}.txt', Xt, delimiter=',', fmt='%.6f')
                np.savetxt(f'{datasetFolder}/y_{ii}.txt', y, delimiter=',', fmt='%.6f')
                np.savetxt(f'{datasetFolder}/yt_{ii}.txt', yt, delimiter=',', fmt='%.6f')
                ii += 1
                print('This dataset works!\n')
                break

