""" 
This code creates different datasets to be used for training an Optical Neural Network (4x4).
MNIST - takes 4 digits and normalizes them from 0 to 1
IRIS - takes the iris dataset, adds a flower, and augments the remaining flowers using their respective distributions
GAUSSIAN - just takes the sklearn function to create multiple gaussian distributions

Author: Simon Geoffroy-Gagnon
Edit: 2020.01.13
"""
import os
from urllib.request import urlretrieve
import matplotlib.pyplot as plt
import pandas as pd
import random
import gzip
import numpy as np
from sklearn import datasets
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split    
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs

# Set random seed to always get same data
rng = 5 
random.seed(rng)

# MNIST DATASET CREATOR

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

def MNIST_dataset(digits=None, N=4, nsamples=1000): # this is for unnormalized MNIST: [1,3,6,7]):
    " Download MNIST dataset "
    X_train = load_mnist_images('train-images-idx3-ubyte.gz').reshape(60_000, -1)
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz').reshape(10_000, -1)
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    if digits is None:
        digits = random.sample(range(0, 10), N)

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

    X, Xt, y, yt = train_test_split(X, y, test_size=0.2)          
    
    rand_ind = random.sample(list(range(len(X))), int(nsamples))
    X = X[rand_ind]
    y = y[rand_ind]
    rand_ind = random.sample(list(range(len(Xt))), int(nsamples*0.2))
    Xt = Xt[rand_ind]
    yt = yt[rand_ind]

    return X, y, Xt, yt

def iris_dataset(print_plots=False, divide_mean=1.25, save=False, nsamples=1):
    " IRIS DATASET MAKER "
    iris = datasets.load_iris()

    predictors = [i for i in iris.feature_names]

    # Get first mean and covariance matrix
    mean1 = iris.data[0:50].mean(axis=0)
    cov1 = np.cov(iris.data[0:50].T)
    # Get second mean and covariance matrix
    mean2 = iris.data[50:100].mean(axis=0)
    cov2 = np.cov(iris.data[50:100].T)
    # Get third mean and covariance matrix
    mean3 = iris.data[100:150].mean(axis=0)
    cov3 = np.cov(iris.data[100:150].T)

    delta_means = []
    for ii in range(len(iris.data[0])):
        delta_mean1 = abs((mean1[ii] - mean2[ii])/2)
        delta_mean2 = abs((mean2[ii] - mean3[ii])/2)
        delta_mean3 = abs((mean3[ii] - mean1[ii])/2)
        delta_means.append(((delta_mean1 + delta_mean2 +
                             delta_mean3)/divide_mean))

    # Get covariance of new flower
    # create new flower
    new_flower = np.random.multivariate_normal(mean1 + np.array(delta_means)/2,
                                               cov2, int(50 + nsamples/4))

    # Augment other flowers
    augment_flower1 = np.random.multivariate_normal(mean1, cov1, int(nsamples/4))
    augment_flower2 = np.random.multivariate_normal(mean2, cov2, int(nsamples/4))
    augment_flower3 = np.random.multivariate_normal(mean3, cov3, int(nsamples/4))

    iris_new = []
    target = []
    iris_new = np.vstack([iris.data[:50], augment_flower1,
                          iris.data[50:100], augment_flower2,
                          iris.data[100:150], augment_flower3,
                          new_flower])

    target = np.hstack([np.ones(50+int(nsamples/4))*0, np.ones(50+int(nsamples/4))*1,
                        np.ones(50+int(nsamples/4))*2, np.ones(50+int(nsamples/4))*3])

    dataset = datasets.base.Bunch(data=iris_new, target=target,
                                  feature_names = iris.feature_names,
                                  target_names = ['setosa', 'versicolor',
                                                  'virginica','germanica'])

    df_features = pd.DataFrame(dict(x_0=dataset.data[:,0],
                                    x_1=dataset.data[:,1],
                                    x_2=dataset.data[:,2],
                                    x_3=dataset.data[:,3],
                                    label=dataset.target))

    #df_targets = pd.DataFrame({'color': ['red', 'blue', 'green','pink']})
    df_targets = pd.get_dummies(dataset.target, prefix='color')

    # Rename features
    features = {'x_{}'.format(x):iris.feature_names[x] for x in range(4)}
    df_features.rename(columns = features, inplace = True)

    df_features[predictors] = (df_features[predictors] -
       df_features[predictors].min()) / (
               df_features[predictors].max() -
               df_features[predictors].min())

    ## The indices of the features that we are plotting
    if print_plots:
        df_features.rename(columns = {'label':'Label'}, inplace = True)

        #now plot using pandas
        color_wheel = {0: 'red',  1: 'green', 2: 'blue', 3: 'black'}

        colors = df_features["Label"].map(lambda x: color_wheel.get(x))
        # Preprocess the data

        fig = scatter_matrix(df_features[predictors], alpha=0.6,
                             figsize=(10, 8),
                             diagonal='hist', color=colors)

        for item in fig:
            for ax in item:
                # We change the fontsize of minor ticks label
                ax.tick_params(axis='both', which='major', labelsize=5)
                ax.tick_params(axis='both', which='minor', labelsize=5)
    #            ax.set_yticklabels([0, 0.5, 1])
    #            ax.set_xticklabels([0, 0.5, 1])

        plt.rcParams.update({'font.size': 10})

        plt.suptitle('Scatter-Matrix', fontsize=30)
        # plt.savefig('Figures/iris_scatter_matrix.png')
        plt.show()
    if save:
        df_features[iris.feature_names].to_csv(
                r'../Weights/4_flowers_100_X.txt',
                   header=None, index=None, sep=',')

        pd.get_dummies(df_features['Label']).to_csv(
                r'../Weights/4_flowers_100_Y.txt', header=None, index=None,
                sep=',')

    X, y = df_features[predictors].values, df_targets.values
    X, Xt, y, yt = train_test_split(X, y, test_size=0.2)          
    return X, y, Xt, yt

def plot_OG_iris():
    iris = datasets.load_iris()
    predictors = [i for i in iris.feature_names]

    dataset = datasets.base.Bunch(data=iris.data, target=iris.target,
                                  feature_names = iris.feature_names,
                                  target_names = ['setosa', 'versicolor',
                                                  'virginica'])

    df_features = pd.DataFrame(dict(x_0=dataset.data[:,0],
                                    x_1=dataset.data[:,1],
                                    x_2=dataset.data[:,2],
                                    x_3=dataset.data[:,3],
                                    label=dataset.target))

    df_features.rename(columns = {'label':'Label'}, inplace = True)

    #now plot using pandas
    color_wheel = {0: 'red',  1: 'green', 2: 'blue', 3: 'black'}

    colors = df_features["Label"].map(lambda x: color_wheel.get(x))

    # Rename features
    features = {'x_{}'.format(x):iris.feature_names[x] for x in range(4)}
    df_features.rename(columns = features, inplace = True)

    plt.rcParams.update({'font.size': 12})

    fig = scatter_matrix(df_features[predictors], alpha=0.8,
                         figsize=(10, 8),
                         diagonal='hist', color=colors)

    for item in fig:
        for ax in item:
            # We change the fontsize of minor ticks label
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.tick_params(axis='both', which='minor', labelsize=0)
#            ax.set_yticklabels([0, 0.5, 1])
#            ax.set_xticklabels([0, 0.5, 1])



    plt.suptitle('', fontname='Calibri', fontsize=20)
    plt.savefig('Figures/iris_scatter_matrix_OG.png')

def gaussian_dataset(targets=4, features=4, nsamples=10000, cluster_std=.1, rng=1):
    " GAUSSIAN BLOB MAKER "
    X, y = make_blobs(n_samples=nsamples, centers=targets, n_features=features, cluster_std=cluster_std,
                      center_box=(0, 1), shuffle=False, random_state=rng)
    ohe_labels = pd.get_dummies(y).values
    X, Xt, y, yt = train_test_split(X, ohe_labels, test_size=0.2)
    return X, y, Xt, yt
