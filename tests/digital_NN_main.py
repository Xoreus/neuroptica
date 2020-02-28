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
sys.path.append('/home/simon/Documents/neuroptica/digital_neural_network')
import Digital_Neural_Network as dnn
import og_dnn
import neural_network as nn
import matplotlib.pyplot as plt

def get_current_accuracy(xtst, ytst, net):
    correct_pred = 0
    for idx, x_curr in enumerate(xtst):
        net.setInput(x_curr)
        net.feedForward()
        if all(ytst[idx] == net.getThResults()):
            correct_pred += 1
    return correct_pred/len(xtst)

def create_train_dnn(X, y, Xt, yt, FOLDER, EPOCHS=300, update_time=50, h_num=0):
    beta = 1
    eta = .05
    alpha = 0.1
    val_accuracy = []
    trn_accuracy = []
    losses = []

    # Build network
    topology = []
    topology.append(4)
    topology.append(4)
    topology.append(4)
    net = og_dnn.Network(topology)

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

        if not np.mod(epoch, update_time):
            print(f"error: {err:.2f}, validation accuracy = {val_accuracy[-1]:.2f}% ")

    print('Done Epochs')

    # Get network weights
    weights = net.getWeights()

    # Plot loss, training acc and val acc
    if 0:
        plt.plot()
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
        plt.savefig(f'{FOLDER}/Figures_Fitting/DigitalNeuralNetwork-accuracy_losses.png')

        df = pd.DataFrame({'Losses':losses, 'Training Accuracy':trn_accuracy, 'Validation Accuracy':val_accuracy})
        df.to_csv(f'{FOLDER}/Data_Fitting/DigitalNeuralNetwork_{len(X[0])}Features.txt')

    return net, weights

if __name__ == '__main__':
    dataset_name = 'Gauss'
    SAMPLES = 150
    rng = 7
    EPOCHS = 300

    for N in [4]:
        for rng in range(2):
            FOLDER = f'Analysis/DNN/Digital_Neural_Network_{SAMPLES*N}_{rng}_N={N}'
            print(f'RNG = {rng}, N = {N}')
            if dataset_name == 'MNIST':
                X, y, Xt, yt = cd.get_data([1,3,6,7], N=N, nsamples=SAMPLES)
            elif dataset_name == 'Gauss':
                X, y, Xt, yt = cd.gaussian_dataset(targets=int(N), features=int(N), nsamples=SAMPLES*N, rng=rng)
            elif dataset_name == 'Iris':
                X, y, Xt, yt = cd.iris_dataset(nsamples=int(SAMPLES))

            X = (X - np.min(X))/(np.max(X) - np.min(X))
            Xt = (Xt - np.min(Xt))/(np.max(Xt) - np.min(Xt))
            Xog, Xtog = X, Xt


            net, weights = create_train_dnn(X, y, Xt, yt, FOLDER, EPOCHS, h_num=1)
            # plot_scatter_matrix.plot_scatter_matrix(X, y)
            # plt.savefig(f'{FOLDER}\\Datasets\\GaussianDataset.png')

            # Val Accuracy
            print('Validation Accuracy: {:.1f}%'.format(get_current_accuracy(
                    Xt, yt, net)*100))

            # Test Accuracy
            print('Training Accuracy: {:.1f}%'.format(get_current_accuracy(
                    X, y, net)*100))

            if get_current_accuracy(Xt, yt, net)*100 > 98:
                np.savetxt(f'../linsep-datasets/X.txt', X, delimiter=',', fmt='%.6f')
                np.savetxt(f'../linsep-datasets/Xt.txt', Xt, delimiter=',', fmt='%.6f')
                np.savetxt(f'../linsep-datasets/y.txt', y, delimiter=',', fmt='%.6f')
                np.savetxt(f'../linsep-datasets/yt.txt', yt, delimiter=',', fmt='%.6f')
                break
