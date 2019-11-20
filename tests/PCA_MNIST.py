# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 15:02:18 2019
123
@author: sgeoff1
"""
import os
from urllib.request import urlretrieve
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE # Only works for visualization for 2d :(
from sklearn.decomposition import PCA
# import scatter_matrix_plotter as pb

# We then define functions for loading MNIST images and labels.
# For convenience, they also download the requested files if needed.
import gzip

def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
    print("Downloading %s" % filename)
    urlretrieve(source + filename, filename)

def load_mnist_images(filename):
    if not os.path.exists(filename):
        download(filename)
    # Read the inputs in Yann LeCun's binary format.
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    # The inputs are vectors now, we reshape them to monochrome 2D images,
    # following the shape convention: (examples, channels, rows, columns)
    data = data.reshape(-1, 1, 28, 28)
    # The inputs come as bytes, we convert them to float32 in range [0,1].
    # (Actually to range [0, 255/256], for compatibility to the version
    # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
    return data / np.float32(256)


def load_mnist_labels(filename):
    if not os.path.exists(filename):
        download(filename)
    # Read the labels in Yann LeCun's binary format.
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    # The labels are vectors of integers now, that's exactly what we want.
    return data

def acc(W, X, Y, zeta=0):
    pred = np.argmax(abs(X.dot(W)), axis=1)
    c = np.sum(np.argmax(Y, axis=1) == pred)
    return c

def get_data(digits=None, N=4): # this is for unnormalized MNIST: [1,3,6,7]):
    # Download MNIST dataset
    X_train = load_mnist_images('train-images-idx3-ubyte.gz').reshape(60_000, -1)
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz').reshape(10_000, -1)
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    if digits is None:
        digits = random.sample(range(0, 10), N)
    # print(digits)
    # NEED 4 DIGITS ##########
    train_mask = np.isin(y_train, digits)
    test_mask = np.isin(y_test, digits)

    X_train_4_digits, y_train_4_digits = X_train[train_mask], y_train[train_mask]
    X_test_4_digits, y_test_4_digits = X_test[test_mask], y_test[test_mask]

    # Create dimensionality reducer (PCA with 4 dimensions) and fit it to
    # X_train_4_digits
    pca = PCA(n_components=N)
    pca.fit(X_train_4_digits)

    # transform the training and testing datasets
    X = pca.transform(X_train_4_digits)

    y = pd.get_dummies(y_train_4_digits, len(digits)).values

    Xt = pca.transform(X_test_4_digits)

    yt = pd.get_dummies(y_test_4_digits, len(digits)).values

    return X, y, Xt, yt, X_train_4_digits, y_train_4_digits
