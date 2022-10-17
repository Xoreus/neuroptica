"""
This code creates different datasets to be used for training an Optical Neural Network.
MNIST - takes N digits and normalizes them from 0 to 1
IRIS - takes the iris dataset, adds a flower, and augments the remaining flowers using their respective distributions
GAUSSIAN - just takes the sklearn function to create multiple gaussian distributions

Author: Simon Geoffroy-Gagnon
Edit: 2020.06.26
"""
import os
from urllib.request import urlretrieve
import matplotlib as mpl
# mpl.use('Agg') # If you cant plt.show() anything, use this
mpl.use('TKAgg')
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

from matplotlib import rc,rcParams
rc('font', weight='bold')
rc('text', usetex=True)
import matplotlib.pyplot as plt
import pandas as pd
import random
import gzip
import numpy as np
from sklearn import datasets
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
import plot_scatter_matrix 

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

def MNIST_dataset(classes=4, features=4, nsamples=100, digits=[1,3,6,7]): # this is for unnormalized MNIST: [1,3,6,7]):
    " Download MNIST dataset "
    X_train = load_mnist_images('train-images-idx3-ubyte.gz').reshape(60_000, -1) # shape: (60000 rows, 784 column), i.e. 60000 28*28 pictures of chars
    #print(X_train[0])
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz') # y_train's shape: (60000,)
    #print(y_train[0])
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz').reshape(10_000, -1)
    #print(X_test[8])
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')
    #print(y_test[8])

    if classes != 4:
        digits = random.sample(range(10), classes)

    print(f"Classes: {digits}") # [9, 4, 5, 6, 7, 8, 0, 1, 3, 2]

    # Array of Trues that come about when y_train is a digit
    train_mask = np.isin(y_train, digits) # (60000,) of 1s or 0s
    test_mask = np.isin(y_test, digits)

    # Removes values that are not apart of the digits that are being tested
    X_train_4_digits, y_train_4_digits = X_train[train_mask], y_train[train_mask]
    # X_train_4_digits's shape:(60000, 784), all 60000 samples are digits
    # y_train_4_digits's shape (60000,)

    # Create dimensionality reducer (PCA with 4 dimensions) and fit it to dataset
    pca = PCA(n_components=features) # a number was represented by 784 values originally, now only {features} values
    pca.fit(X_train_4_digits)

    # transform the training and testing datasets
    X = pca.transform(X_train_4_digits) # shape of X: (60000, {features})
    y = pd.get_dummies(y_train_4_digits, len(digits)).values # dimension of y is: (60000, 10), these are 60000 labels
    X, Xt, y, yt = train_test_split(X, y, test_size=0.2) # each label one hot encoded vector of length 10, indicating the correct digit this is.

    rand_ind = random.sample(list(range(len(X))), int(nsamples*classes))
    X = X[rand_ind]
    y = y[rand_ind]
    rand_ind = random.sample(list(range(len(Xt))), int(nsamples*0.2*classes))
    Xt = Xt[rand_ind]
    yt = yt[rand_ind]

    return np.array(X), np.array(y), np.array(Xt), np.array(yt)

def FFT_MNIST(half_square_length=2, classes=10, nsamples=100): # FFT of MNIST, 
    " Download MNIST dataset "
    X_train = load_mnist_images('train-images-idx3-ubyte.gz').squeeze()
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz').squeeze()
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')
    
    digits = random.sample(range(0, 10), classes)

    train_mask = np.isin(y_train, digits)
    test_mask = np.isin(y_test, digits)

    X_train, y_train = X_train[train_mask], y_train[train_mask]
    X_test, y_test = X_test[test_mask], y_test[test_mask]

    rand_ind = random.sample(list(range(len(X_train))), int(nsamples*classes))
    X_train = X_train[rand_ind]
    y_train = y_train[rand_ind]
    rand_ind = random.sample(list(range(len(X_test))), int(nsamples*0.2*classes))
    X_test = X_test[rand_ind]
    y_test= y_test[rand_ind]

    X_train = np.array([np.fft.fft2(X) for X in X_train])
    X_test = np.array([np.fft.fft2(X) for X in X_test])

    # To show images of FFT'ed MNIST samples #
    # plt.imshow(np.absolute(X_train[1,:,:]), cmap='gray')
    # plt.show()
    # plt.imshow(np.absolute(np.fft.fftshift(X_train[1,:,:])), cmap='gray')
    # plt.show()

    X = [[X[:half_square_length,:half_square_length], X[-half_square_length:,:half_square_length], X[-half_square_length:,-half_square_length:], X[:half_square_length, -half_square_length:]] for X in X_train]
    y = pd.get_dummies(y_train, len(digits)).values
    X = np.reshape(X, [int(nsamples*classes), 4*(half_square_length)**2])
    Xt = [[X[:half_square_length,:half_square_length], X[-half_square_length:,:half_square_length], X[-half_square_length:,-half_square_length:], X[:half_square_length, -half_square_length:]] for X in X_test]
    yt = pd.get_dummies(y_test, len(digits)).values
    Xt = np.reshape(Xt, [int(nsamples*0.2*classes), (2*half_square_length)**2])

    return (np.array(X)), np.array(y), (np.array(Xt)), np.array(yt)
    
def FFT_MNIST_PCA(features=10, classes=10, nsamples=100): # FFT of MNIST, 
    " Download half_square_length dataset "
    X_train = load_mnist_images('train-images-idx3-ubyte.gz').squeeze()
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz').squeeze()
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')
    
    digits = random.sample(range(0, 10), classes)

    train_mask = np.isin(y_train, digits)
    test_mask = np.isin(y_test, digits)

    X_train, y_train = X_train[train_mask], y_train[train_mask]
    X_test, y_test = X_test[test_mask], y_test[test_mask]

    rand_ind = random.sample(list(range(len(X_train))), int(nsamples*classes))
    X_train = X_train[rand_ind]
    y_train = y_train[rand_ind]
    rand_ind = random.sample(list(range(len(X_test))), int(nsamples*0.2*classes))
    X_test = X_test[rand_ind]
    y_test= y_test[rand_ind]

    X_train = np.array([np.fft.fft2(X) for X in X_train])
    X_test = np.array([np.fft.fft2(X) for X in X_test])

    X_train = np.reshape(X_train, [nsamples*classes, -1])
    X_test = np.reshape(X_test, [int(nsamples*0.2*classes), -1])

    pca = PCA(n_components=features)
    X = pca.fit_transform(np.absolute(X_train))
    Xt = pca.transform(np.absolute(X_test))

    y = pd.get_dummies(y_train, len(digits)).values
    yt = pd.get_dummies(y_test, len(digits)).values

    return (np.array(X)), np.array(y), (np.array(Xt)), np.array(yt)
    
def iris_dataset_augment(divide_mean=1.25, save=False, nsamples=1):
    " IRIS DATASET MAKER "
    iris = datasets.load_iris()

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

    y = pd.get_dummies(list(target)).values

    X = iris_new

    X, Xt, y, yt = train_test_split(X, y, test_size=0.2)
    return np.array(X), np.array(y), np.array(Xt), np.array(yt)

def iris_dataset(nsamples=1):
    " IRIS DATASET MAKER "
    iris = datasets.load_iris()

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

    # Augment other flowers
    augment_flower1 = np.random.multivariate_normal(mean1, cov1, int(nsamples))
    augment_flower2 = np.random.multivariate_normal(mean2, cov2, int(nsamples))
    augment_flower3 = np.random.multivariate_normal(mean3, cov3, int(nsamples))

    target = []
    iris = np.vstack([iris.data[:50], augment_flower1,
                          iris.data[50:100], augment_flower2,
                          iris.data[100:150], augment_flower3])
                          

    target = np.hstack([np.ones(50+int(nsamples))*0, np.ones(50+int(nsamples))*1,
                        np.ones(50+int(nsamples))*2])

    y = pd.get_dummies(list(target)).values

    X = iris

    X, Xt, y, yt = train_test_split(X, y, test_size=0.2)
    return np.array(X), np.array(y), np.array(Xt), np.array(yt)

def plot_agmented_iris(nsamples=300):
    iris = datasets.load_iris()
    predictors = [i[:-5] for i in iris.feature_names]
    X, y, *_ = iris_dataset(nsamples=nsamples)
    y = [np.argmax(yy) for yy in y]
    X = {iris.feature_names[x][:-5]:X[:,x] for x in range(4)} 
    df = pd.DataFrame(X)
    df.loc[:,'Label'] = y
    print(df)
    #now plot using pandas
    color_wheel = {0: 'red',  2: 'green', 1: 'blue', 3: 'black'}

    colors = df["Label"].map(lambda x: color_wheel.get(x))

    # Rename features
    features = {'x_{}'.format(x):iris.feature_names[x] for x in range(4)}
    df.rename(columns = features, inplace = True)

    plt.rcParams.update({'font.size': 12})

    fig = scatter_matrix(df[predictors], alpha=0.8, figsize=(10, 10), diagonal='kde', color=colors)

    for item in fig:
        for ax in item:
            # We change the fontsize of minor ticks label
            ax.tick_params(axis='both', which='major', labelsize=0)
            ax.tick_params(axis='both', which='minor', labelsize=0)

            ax.xaxis.label.set_size(27)
            ax.yaxis.label.set_size(27)


    plt.suptitle('')
    plt.savefig('/home/edwar/Documents/Github_Projects/neuroptica/Simulations/Analysis/Crop_Me/Iris-Augmented.pdf')

def plot_OG_iris():
    iris = datasets.load_iris()
    predictors = [i for i in iris.feature_names]
    predictor = [pred[:-5] for pred in predictors]

    df = pd.DataFrame(dict(x_0=iris.data[:,0],
                                    x_1=iris.data[:,1],
                                    x_2=iris.data[:,2],
                                    x_3=iris.data[:,3],
                                    label=iris.target))

    df.rename(columns = {'label':'Label'}, inplace = True)

    #now plot using pandas
    color_wheel = {0: 'red',  1: 'green', 2: 'blue', 3: 'black'}

    colors = df["Label"].map(lambda x: color_wheel.get(x))

    # Rename features
    features = {'x_{}'.format(x):iris.feature_names[x] for x in range(4)}
    df.rename(columns = features, inplace = True)
     # df.rename(columns = predictors,  inplace = True)
    df2 = df.set_axis(predictor + ['Label'], axis=1, inplace=False)
    plt.rcParams.update({'font.size': 12})

    fig = scatter_matrix(df2[predictor], alpha=0.8, figsize=(10, 10), diagonal='kde', color=colors)

    for item in fig:
        for ax in item:
            # We change the fontsize of minor ticks label
            ax.tick_params(axis='both', which='major', labelsize=0)
            ax.tick_params(axis='both', which='minor', labelsize=0)
            ax.xaxis.label.set_size(27)
            ax.yaxis.label.set_size(27)


    plt.suptitle('', fontname='Calibri', fontsize=34)
    plt.savefig('/home/edwar/Documents/Github_Projects/neuroptica/tests/Crop_Me/Iris-OG.pdf')

def gaussian_dataset(targets=4, features=4, nsamples=10000, cluster_std=.1, rng=1):
    " GAUSSIAN BLOB MAKER "
    X, y = make_blobs(n_samples=nsamples, centers=targets, n_features=features, cluster_std=cluster_std,
                      center_box=(0, 1), shuffle=False, random_state=rng)
    ohe_labels = pd.get_dummies(y).values
    X, Xt, y, yt = train_test_split(X, ohe_labels, test_size=0.2)
    return np.array(X), np.array(y), np.array(Xt), np.array(yt)

if __name__ == '__main__':
    FFT_MNIST()
