"""
digital_NN_main.py
Simple single layer fully connected neural network, used to test whether or not the unitary-ness of 
ONNs actually affect accuracy

Author: Simon Geoffroy-Gagnon
Edit: 29.01.2020
"""
import sys
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append('/home/simon/Documents/neuroptica/digital_neural_network/')
import Digital_Neural_Network as dnn
import neural_network as nn
import create_datasets as cd 
import create_datasets as cd 
import setupSimulation as setSim
import saveSimulationData as sSD
import plot_scatter_matrix

def get_current_accuracy(xtst, ytst, net):
    correct_pred = 0
    for idx, x_curr in enumerate(xtst):
        net.setInput(x_curr)
        net.feedForward()
        if all(ytst[idx] == net.getThResults()):
            correct_pred += 1
    return correct_pred/len(xtst)

def create_train_dnn(X, y, Xt, yt, FOLDER, EPOCHS=300, h_num = 0):
    
    saving = True
    plotting = True

    batch_size = 2**6

    beta = 1
    eta = .05
    alpha = 0.1

    val_accuracy = []

    trn_accuracy = []
    losses = []

    setSim.createFOLDER(FOLDER)

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

        if not np.mod(epoch, 100):
            print("error: ", err)

    print('Done Epochs')


    # Get network weights
    weights = net.getWeights()

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
    plt.savefig(f'{FOLDER}/Figures_Fitting/DigitalNeuralNetwork-accuracy_losses.png')

    df = pd.DataFrame({'Losses':losses, 'Training Accuracy':trn_accuracy, 'Validation Accuracy':val_accuracy})
    df.to_csv(f'{FOLDER}/Data_Fitting/DigitalNeuralNetwork_{len(X[0])}Features.txt')


    # Val Accuracy
    print('Validation Accuracy: {:.1f}%'.format(get_current_accuracy(
            Xt, yt, net)*100))

    # Test Accuracy
    print('Training Accuracy: {:.1f}%'.format(get_current_accuracy(
            X, y, net)*100))

    return net, weights

if __name__ == '__main__':
    dataset_name = 'Iris'
    SAMPLES = 2000
    rng = 8
    EPOCHS = 300
    FOLDER = f'/home/simon/Documents/neuroptica/tests/Analysis/DNN/Iris-0h-sig/'
    N = 4
    ii = 0

    if dataset_name == 'MNIST':
        X, y, Xt, yt = cd.get_data([1,3,6,7], N=N, nsamples=SAMPLES)
    elif dataset_name == 'Gauss':
        X, y, Xt, yt = cd.gaussian_dataset(targets=int(N), features=int(N), nsamples=SAMPLES, rng=rng)
    elif dataset_name == 'Iris':
        X, y, Xt, yt = cd.iris_dataset(nsamples=int(SAMPLES))

    # X = np.genfromtxt(f'{FOLDER}/Datasets/Gaussian_X_4Features_4Classes_Samples=560_Dataset.txt', delimiter=',')
    # y = np.genfromtxt(f'{FOLDER}/Datasets/Gaussian_y_4Features_4Classes_Samples=560_Dataset.txt', delimiter=',')
    # Xt = np.genfromtxt(f'{FOLDER}/Datasets/Gaussian_Xt_4Features_4Classes_Samples=560_Dataset.txt', delimiter=',')
    # yt = np.genfromtxt(f'{FOLDER}/Datasets/Gaussian_yt_4Features_4Classes_Samples=560_Dataset.txt', delimiter=',')

    X = (X - np.min(X))/(np.max(X) - np.min(X))
    Xt = (Xt - np.min(Xt))/(np.max(Xt) - np.min(Xt))
    Xog, Xtog = X, Xt


    create_train_dnn(X, y, Xt, yt, FOLDER, EPOCHS, h_num=0)
    ax = plot_scatter_matrix.plot_scatter_matrix(X, y)
    plt.savefig(f'{FOLDER}/Datasets/GaussianDataset.pdf')



