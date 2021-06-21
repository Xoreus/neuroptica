#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 08:42:44 2018

@author: simon
Neural Network that takes in either the Iris dataset (with an added flower) or
the gaussian blob maker dataset. It then trains the network, and outputs the
weight matrix, along with saving the dataset and the weights through pickle.

NN code adapted from:
    https://thecodacus.com/neural-network-scratch-python-no-libraries/
"""

import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/edwar/Documents/Github_Projects/neuroptica/tests/')
import create_datasets
import neural_network as nn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from numpy import genfromtxt
import random

random.seed(5)
np.random.seed(5)


def build_network(xtrn, ytrn, h_num=0, biasing=False):
    topology = []
    topology.append(len(xtrn[0]))
    for _ in range(h_num):
        topology.append(len(xtrn[0]))
    topology.append(len(ytrn[0]))
    net = nn.Network(topology, biasing)
    nn.Neuron.eta = 0.09
    nn.Neuron.alpha = 0.015

    return net

def load_data():
    pickle_path = os.getcwd() + '/../Weights/'
    pickle_weights = pickle_path + 'weights.pkl'
    pickle_features = pickle_path + 'features.pkl'
    pickle_targets = pickle_path + 'targets.pkl'

    with open(pickle_weights, 'rb') as f:
        weights = pickle.load(f)
    with open(pickle_features, 'rb') as f:
        x = pickle.load(f)
    with open(pickle_targets, 'rb') as f:
        y = pickle.load(f)
    return weights, x, y

def load_csv_data():
    pickle_path = os.getcwd() + '/../Weights/'
    pickle_features = pickle_path + 'x_q.csv'
    pickle_targets = pickle_path + 'y.csv'
    x = genfromtxt(pickle_features, delimiter=',')
    y = genfromtxt(pickle_targets, delimiter=',')
    return x, y

def get_current_accuracy(xtst, ytst, net):
    # Test set predictions
    correct_pred = 0
    for idx, x_curr in enumerate(xtst):
        net.setInput(x_curr)
        net.feedForward()
        if all(ytst[idx] == net.getThResults()):
            correct_pred += 1
    return correct_pred/len(xtst)

def main():
    Epochs = 500
    h_num = 0
    dataset = 'iris'
    # dataset = 'blobs'
    saving = True
    plotting = True
    beta = 0.1
    eta = 1
    alpha = 0.1
    val_accuracy = []
    trn_accuracy = []
    cumul_error = []

    if dataset == 'iris':
        xtrn, ytrn, xval, yval = create_datasets.iris_dataset(nsamples=1200)
    if dataset == 'blobs':
#        x, y = blob_maker(plotting=False, nsamples=200, cluster_std=.075)
#        x = np.loadtxt('../4x4_ONN/x_q.txt')
#        y = np.loadtxt('../4x4_ONN/y_q.txt')
#        x = np.loadtxt('x.out')
#        y = np.loadtxt('y.out')
        x = np.genfromtxt('X.csv', delimiter=',')
        y = np.genfromtxt('y.csv', delimiter=',')

    # Build network
    net = build_network(xtrn, ytrn, h_num=h_num)
    # Split Data into train set and test set
    # xtrn, xtst, ytrn, ytst = train_test_split(x, y, shuffle=True, stratify=y)
    # xtrn, xval, ytrn, yval = train_test_split(xtrn, ytrn, shuffle=True,
                                              # stratify=ytrn)


    # Train model with Gradient Descent
    for epoch in range(Epochs):
        err = 0
        # Shuffle data at the start of each epoch
        instances, targets = shuffle(xtrn, ytrn)
        nn.Neuron.alpha = alpha
        for ii in range(len(instances)):
            nn.Neuron.eta = eta/(1+epoch*beta)
            net.setInput(instances[ii])
            net.feedForward()
            net.backPropagate(targets[ii])
            err = err + net.getError(targets[ii])

        cumul_error.append(err)
        val_accuracy.append(get_current_accuracy(xval, yval, net))
        trn_accuracy.append(get_current_accuracy(xtrn, ytrn, net))

        if not np.mod(epoch, 100):
            print("error: ", err)

    print('Done Epochs')


    # Get network weights
    weights = net.getWeights()

    if plotting:
        plt.figure(num=None, figsize=(18, 10), dpi=80, facecolor='w',edgecolor='k')
        plt.rcParams.update({'font.size': 24})
        plt.plot(np.array(cumul_error)/max(cumul_error), linewidth=3)
        plt.title('Cumulative Error, normalized')
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        if saving: plt.savefig('Figures/cumul_err_{}_hnum={}.pdf'.format(dataset, h_num))
        plt.show()

        plt.figure(num=None, figsize=(18, 10), dpi=80, facecolor='w',edgecolor='k')
        plt.rcParams.update({'font.size': 24})
        plt.plot(trn_accuracy, linewidth=3, label='Train Accuracy')
        plt.plot(val_accuracy, linewidth=3, label='Validation Accuracy')

        plt.title('Validation Accuracy after each epoch')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        if saving: plt.savefig('Figures/accuracy_{}_hnum={}.pdf'.format(dataset, h_num))
        plt.show()

    # Possibly save the data and weights
    if saving:
        pickle_path = os.getcwd() + '/../Weights/'
        pickle_weights = pickle_path + 'weights_{}.pkl'.format(dataset)
        pickle_features = pickle_path + 'features_{}.pkl'.format(dataset)
        pickle_targets = pickle_path + 'targets_{}.pkl'.format(dataset)

        with open(pickle_weights, 'wb') as f:
            pickle.dump(weights, f)
        with open(pickle_features, 'wb') as f:
            pickle.dump(x, f)
        with open(pickle_targets, 'wb') as f:
            pickle.dump(y, f)

        np.savetxt(pickle_path+"x_{}.csv".format(dataset), x, delimiter=",")
        np.savetxt(pickle_path+"y_{}.csv".format(dataset), y, delimiter=",")
        np.savetxt(pickle_path+"w_{}.csv".format(dataset), weights,
                   delimiter=",")


    # Val Accuracy
    print('Validation Accuracy: {:.1f}%'.format(get_current_accuracy(
            xval, yval, net)*100))

    # Test Accuracy
    print('Testing Accuracy: {:.1f}%'.format(get_current_accuracy(
            xtst, ytst, net)*100))

    return net, weights, x, y


if __name__ == '__main__':
    # Set random seed
    outputs = 6
    net, weights, x, y = main()

    corr = 0
    for idx, inst in enumerate(x):
        net.setInput(inst)
        net.feedForward()
        ypred = net.getThResults()

    idx = 55
    net.setInput(x[idx])
    net.feedForward()

    print("y predicted: {}, y actual: {}".format(
          nn.sigmoid(np.array(weights[:outputs])[:, :outputs].dot(x[idx])),
          np.array(y[idx])))
    print('\n')
    print(np.matrix(weights[:outputs]))
    print('\n')

    w = np.array(weights[:outputs])
    W = np.matrix(weights[:outputs])
    zh = w.dot(x[idx].T).squeeze()
    yh = nn.ReLU(zh)
    print('x_in = {}'.format(x[idx]))
    print('z_h = {}'.format(zh))
    print('Y = {}, y_h = {}'.format(y[idx], yh))

if 0:
    x = np.loadtxt('../4x4_ONN/x_q.txt')
    y = np.loadtxt('../4x4_ONN/y_q.txt')

    net = build_network(x, y, h_num=0)

    xtrn, xtst, ytrn, ytst = train_test_split(x, y, shuffle=True, stratify=y)

    xtrn, xval, ytrn, yval = train_test_split(xtrn, ytrn, shuffle=True,
                                              stratify=ytrn)

    # Shuffle data at the start of each epoch
    instances, targets = shuffle(xtrn, ytrn)
    nn.Neuron.alpha = 0.01
    nn.Neuron.eta = 0.001
    net.setInput(instances[1])
    net.feedForward()
    print(net.getResults())
    print(np.matrix(net.getWeights()))
