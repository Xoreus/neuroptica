"""
This code creates different datasets to be used for training an Optical Neural Network.
MNIST - takes N digits and normalizes them from 0 to 1
IRIS - takes the iris dataset, adds a flower, and augments the remaining flowers using their respective distributions
GAUSSIAN - just takes the sklearn function to create multiple gaussian distributions

Author: Simon Geoffroy-Gagnon
Edit: 2020.06.26
"""
import os
import shutil
from urllib.request import urlretrieve
from torchvision import datasets, transforms

# matplotlib.rcParams['mathtext.fontset'] = 'stix'
# matplotlib.rcParams['font.family'] = 'STIXGeneral'
# matplotlib.rcParams['mathtext.fontset'] = 'custom'
# matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
# matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
# matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

# from matplotlib import rc,rcParams
# rc('font', weight='bold')
# rc('text', usetex=True)
import matplotlib
matplotlib.use('TkAgg')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
import matplotlib.pyplot as plt
import pandas as pd
import random
import gzip
import numpy as np
# from sklearn import datasets
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
import pickle
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


def load_MNIST_dataset():
    transforms_img = transforms.Compose([transforms.ToTensor()])
    mnist_train = datasets.MNIST(root='data', train=True, download=True, transform=transforms_img)
    mnist_test = datasets.MNIST(root='data', train=False, download=True, transform=transforms_img)
    return mnist_train, mnist_test
# collect all instances with label "0" and "1"
# Label 0: [0, cutoff]
# Label 1: [cutoff, 9]
def collect_label_binary_duration(dataset, cutoff = 4):
    data_collected, label_collected = [], []
    labeled = 0

    for (d, target) in (dataset):
        data_collected.append(d.numpy())

        if target <= cutoff:
            label_collected.append([1, 0])
            labeled += 1
        else:
            label_collected.append([0, 1])

    print("Contains label 1 #", labeled)
    return np.concatenate(data_collected).reshape(len(data_collected), 28*28), np.concatenate(label_collected).reshape(len(label_collected), 2)

def MNIST_dataset(classes=4, features=4, nsamples=100, digits=[1,3,6,7]): # this is for unnormalized MNIST: [1,3,6,7]):
    # random.seed() # comment out this line to get same classes each run
    " Download MNIST dataset "
    # X_train = load_mnist_images('train-images-idx3-ubyte.gz').reshape(60_000, -1) # shape: (60000 rows, 784 column), i.e. 60000 28*28 pictures of chars
    #print(X_train[0])
    # y_train = load_mnist_labels('train-labels-idx1-ubyte.gz') # y_train's shape: (60000,)
    #print(y_train[0])
    # X_test = load_mnist_images('t10k-images-idx3-ubyte.gz').reshape(10_000, -1)
    #print(X_test[8])
    # y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')
    #print(y_test[8])

    # ============= group mnist [0-4], [5-9] into categories (0, 1) and (1, 0)
    # methods author: Xuening Dong
    X, y = collect_label_binary_duration(load_MNIST_dataset()[0], cutoff=4)
    X_test, y_test = collect_label_binary_duration(load_MNIST_dataset()[1], cutoff=4)
    X, Xt, y, yt = train_test_split(X, y, test_size=1/6) # each label one hot encoded vector of length 10, indicating the correct digit this is.
    # Create dimensionality reducer (PCA with N dimensions) and fit it to dataset
    pca = PCA(n_components=features) # a number was represented by 784 values originally, now only {features} values
    pca.fit(X)
    X = pca.transform(X) # shape of X: (60000, {features})
    Xt = pca.transform(Xt) # shape of X: (60000, {features})
    X_test = pca.transform(X_test) # shape of X: (60000, {features})
    # print(f"X: {X.shape}")
    # print(f"Xt: {Xt.shape}")
    # print(f"X_test: {X_test.shape}")
    # print(f"y: {y.shape}")
    # print(f"yt: {yt.shape}")
    # print(f"y_test: {y_test.shape}")
    # print(f"dataset range: [{np.min(X):.3f}, {np.max(X):.3f}], [{np.min(Xt):.3f}, {np.max(Xt):.3f}], [{np.min(X_test):.3f}, {np.max(X_test):.3f}]")
    return np.array(X), np.array(y), np.array(Xt), np.array(yt), np.array(X_test), np.array(y_test)
    # ==========================================================

    if classes != 4:
        digits = random.sample(range(10), classes)

    print(f"Classes: {digits}") # [9, 4, 5, 6, 7, 8, 0, 1, 3, 2]
    # Array of Trues that come about when y_train is a digit
    train_mask = np.isin(y_train, digits) # (60000,) of 1s or 0s
    test_mask = np.isin(y_test, digits)

    # Removes values that are not apart of the digits that are being tested
    X_train, y_train = X_train[train_mask], y_train[train_mask]
    X_test, y_test = X_test[test_mask], y_test[test_mask]
    # X_train_4_digits's shape:(60000, 784), all 60000 samples are digits
    # y_train_4_digits's shape (60000,)
    # print(f"{X_train.shape} training+validation samples remaining")
    # print(f"{X_test.shape} testing samples remaining")

    # Create dimensionality reducer (PCA with N dimensions) and fit it to dataset
    pca = PCA(n_components=features) # a number was represented by 784 values originally, now only {features} values
    pca.fit(X_train)
    X = pca.transform(X_train) # shape of X: (60000, {features})
    X_test = pca.transform(X_test) # shape of X: (60000, {features})

    # convert categorical to one-hot: e.g. first 4 samples: [4, 9, 4, 4] -> [[1,0],[0,1],[1,0],[1,0]]
    y = pd.get_dummies(y_train, len(digits)).values # dimension of y is: (60000, 10), these are 60000 labels
    y_test = pd.get_dummies(y_test, len(digits)).values # dimension of y is: (60000, 10), these are 60000 labels
    # train-validation split
    X, Xt, y, yt = train_test_split(X, y, test_size=0.2) # each label one hot encoded vector of length 10, indicating the correct digit this is.

    # sample a subset from train and validation sets to use in simulation
    # print(len(list(range(len(X)))), nsamples*0.8*classes)
    rand_ind = random.sample(list(range(len(X))), int(nsamples*0.8*classes))
    X = X[rand_ind]
    y = y[rand_ind]
    rand_ind = random.sample(list(range(len(Xt))), int(nsamples*0.2*classes))
    Xt = Xt[rand_ind]
    yt = yt[rand_ind]

    return np.array(X), np.array(y), np.array(Xt), np.array(yt), np.array(X_test), np.array(y_test)

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

def load_dataset_pickle(filepath, img_number = 1000):
    data, labels = [], []
    img_shape = (100, 100, 3)


    with open(filepath[0], 'rb') as f:
        for i in range(img_number):
            a = np.array(pickle.load(f))

            if a.shape != img_shape:
                print(a.shape)

            data.append(a)


    with open(filepath[1], 'rb') as f:

        for i in range(img_number):
            b = np.expand_dims(pickle.load(f), 0)

            labels.append(b)

    return np.stack(data), np.stack(labels)

def WIDER_FACE(classes=2, features=8, nsamples=100):
    filepaths = ["../../validation.pkl", "../../validation_labels.pkl"]
    validation_data, validation_labels = load_dataset_pickle(filepath = filepaths, img_number = 5000)
    from copy import deepcopy

    test_label = deepcopy(validation_labels)

    removed_index = []

    for i in range(len(test_label) - 4):

        # sliding window of size 5
        curr_range = test_label[i:i+4]

        # continue if there is 1 valid image
        if np.count_nonzero(curr_range == 1) > 1:
            continue
        elif np.count_nonzero(curr_range == 1) == 1 and curr_range[-1] == 1:
            removed_index.append(i)
            removed_index.append(i + 1)
            continue
        elif np.count_nonzero(curr_range == 1) == 1:
            continue

        # remove the first two of them
        if i not in removed_index:
            removed_index.append(i)
            removed_index.append(i + 1)
            removed_index.append(i + 2)
    
    validation_labels = np.delete(validation_labels, removed_index)
    validation_data = np.delete(validation_data, removed_index, axis = 0)
    filepaths = ["../../train.pkl", "../../train_labels.pkl"]
    train_data, train_labels = load_dataset_pickle(filepath = filepaths, img_number = 10000)
    filepaths = ["../../train_p2.pkl", "../../train_labels_p2.pkl"]
    train_data2, train_labels2 = load_dataset_pickle(filepath = filepaths, img_number = 8000)
    train_data = np.concatenate((train_data, train_data2), axis = 0)
    train_labels = np.concatenate((train_labels, train_labels2), axis = 0)
    filepaths = ["../../train_p3.pkl", "../../train_labels_p3.pkl"]
    train_data2, train_labels2 = load_dataset_pickle(filepath = filepaths, img_number = 2000)
    train_data = np.concatenate((train_data, train_data2), axis = 0)
    train_labels = np.concatenate((train_labels, train_labels2), axis = 0)
    
    test_label = deepcopy(train_labels)
    removed_index = []
    for i in range(len(test_label) - 4):

        # sliding window of size 5
        curr_range = test_label[i:i+4]

        # continue if there is 1 valid image
        if np.count_nonzero(curr_range == 1) > 1:
            continue
        elif np.count_nonzero(curr_range == 1) == 1 and curr_range[-1] == 1:
            removed_index.append(i)
            removed_index.append(i + 1)
            continue
        elif np.count_nonzero(curr_range == 1) == 1:
            continue

        # remove the first two of them
        if i not in removed_index:
            removed_index.append(i)
            removed_index.append(i + 1)
            removed_index.append(i + 2)
    train_labels = np.delete(train_labels, removed_index)[:10000]
    train_data = np.delete(train_data, removed_index, axis = 0)[:10000]
    filepaths = ["../../test.pkl", "../../test_labels.pkl"]
    test_data, test_labels = load_dataset_pickle(filepath = filepaths, img_number = 10000)
    
    test_label = deepcopy(test_labels)
    removed_index = []
    for i in range(len(test_label) - 4):

        # sliding window of size 5
        curr_range = test_label[i:i+4]

        # continue if there is 1 valid image
        if np.count_nonzero(curr_range == 1) > 1:
            continue
        elif np.count_nonzero(curr_range == 1) == 1 and curr_range[-1] == 1:
            removed_index.append(i)
            removed_index.append(i + 1)
            continue
        elif np.count_nonzero(curr_range == 1) == 1:
            continue

        # remove the first two of them
        if i not in removed_index:
            removed_index.append(i)
            removed_index.append(i + 1)
            removed_index.append(i + 2)
    test_labels = np.delete(test_labels, removed_index)
    test_data = np.delete(test_data, removed_index, axis = 0)
    
    # PCA
    train_data_flatten = train_data.reshape((train_data.shape[0], 100*100*3))
    validation_data_flatten = validation_data.reshape((validation_data.shape[0], 100*100*3))
    test_data_flatten = test_data.reshape((test_data.shape[0], 100*100*3))
    pca_func = PCA(n_components = features)
    pca_func.fit(train_data_flatten)
    new_train_data, new_validation_data, new_test_data = pca_func.transform(train_data_flatten), pca_func.transform(validation_data_flatten), pca_func.transform(test_data_flatten)

    
    # normalization (all use training max/min?)
    x_train = (new_train_data - np.min(new_train_data))/(np.max(new_train_data) - np.min(new_train_data)) - 0.5
    x_validation = (new_validation_data - np.min(new_validation_data))/(np.max(new_validation_data) - np.min(new_validation_data)) - 0.5
    x_test = (new_test_data - np.min(new_test_data))/(np.max(new_test_data) - np.min(new_test_data)) - 0.5
    y_train = np.eye(2)[train_labels.squeeze()]
    y_val = np.eye(2)[validation_labels.squeeze()]
    y_test = np.eye(2)[test_labels.squeeze()]
    

    return x_train, y_train, x_validation, y_val, x_test, y_test

def load_CIFAR_10_dataset():
    transforms_img = transforms.Compose([transforms.ToTensor()])

    cifar_train = datasets.CIFAR10(root='data', train=True, download=True, transform=transforms_img)
    cifar_test = datasets.CIFAR10(root='data', train=False, download=True, transform=transforms_img)
    return cifar_train, cifar_test

# modify the CIFAR-10 dataset into a binary classification task
# class "animal": birds, cats, deer, dog
# class "vehicle": airplanes, cars, ships, trucks
def modify_CIFAR_10(dataset):
    data, label = [], []

    animal_label, vehicle_label = [2, 3, 4, 5], [0, 1, 8, 9]

    for (d, target) in (dataset):
        if target in animal_label or target in vehicle_label:
            data.append(d.numpy())

            if target in animal_label:
                label.append([1, 0])
            else:
                label.append([0, 1])

    return np.concatenate(data).reshape(len(data), 32*32*3), np.concatenate(label).reshape(len(label), 2)

def CIFAR_10(classes=2, features=8, nsamples=100, filename='cifar-10-python.tar.gz'):
    # ======= Frontier: binarized CIFAR-10 (Xuening) ====
    cifar_train, cifar_test = load_CIFAR_10_dataset()

    train_set, train_label = modify_CIFAR_10(cifar_train)
    test_set, test_label = modify_CIFAR_10(cifar_test)
    print(train_set.shape)
    print(train_label.shape)
    print(test_set.shape)
    print(test_label.shape)
    '''PCA'''
    X, Xv, y, yv = train_test_split(train_set, train_label, test_size=0.2, stratify=train_label)
    pca = PCA(n_components=features)
    pca.fit(X)
    X = pca.transform(X)
    Xv = pca.transform(Xv)
    Xt = pca.transform(test_set)
    # print(X.shape, Xv.shape, Xt.shape)
    # print(y.shape, yv.shape, yt.shape)
    # exit()
    return X, y, Xv, yv, Xt, test_label
    # ===================================================
    if not os.path.exists(filename):
        print("Downloading CIFAR-10...")
        download(filename, source="https://www.cs.toronto.edu/~kriz/")
    # (after mannual unzip)
    extracted_folder = os.getcwd()+"/cifar-10-batches-py/"
    if not os.path.exists(extracted_folder):
        print("Unpacking CIFAR-10...")
        shutil.unpack_archive(filename)
    n=5
    data = np.zeros((10000*n, 3072)) # to store 5 files each with 10000*((32*32)*3) images
    label = np.zeros(10000*n) # to store labels (1-9)
    test_data = np.zeros((10000, 3072))
    test_label = np.zeros(10000)
    # label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # train_img = data.reshape(50000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
    # test_img = test_data.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
    # plt.figure(figsize=(16,20))
    # plt.subplot(231, title=f"{label_names[int(label[498])]}"), plt.imshow(train_img[498])
    # plt.subplot(232, title=f"{label_names[int(label[721])]}"), plt.imshow(train_img[721])
    # plt.subplot(233, title=f"{label_names[int(label[49999])]}"), plt.imshow(train_img[49999])
    # plt.subplot(234, title=f"{label_names[int(test_label[45])]}"), plt.imshow(test_img[45])
    # plt.subplot(235, title=f"{label_names[int(test_label[26])]}"), plt.imshow(test_img[26])
    # plt.subplot(236, title=f"{label_names[int(test_label[9999])]}"), plt.imshow(test_img[9999])
    # plt.show()
    count = 0
    for file in os.listdir(extracted_folder):
        if "_batch" in file: # training/testing samples
            print(f"{count}:{file}")
            path = f"cifar-10-batches-py/{file}"
            with open(path, 'rb') as f_out:
                dict = pickle.load(f_out, encoding='bytes')
                if "data" in file: # training samples
                    data[count*10000:(count+1)*10000] = dict[b'data']
                    label[count*10000:(count+1)*10000] = dict[b'labels']
                    count += 1
                elif "test" in file: # test samples
                    test_data[:] = dict[b'data']
                    test_label[:] = dict[b'labels']
    '''class selection'''
    class_choice = np.logical_or(label==0, label==5) # cat(3), dog(5)
    class_choice_test = np.logical_or(test_label==0, test_label==5) # cat(3), dog(5)
    two_class, two_label = data[class_choice], label[class_choice]
    two_class_test, two_label_test = test_data[class_choice_test], test_label[class_choice_test]
    '''Process labels: categorical -> one-hot'''
    diag = np.eye(2)
    two_label = diag[np.where(two_label==np.unique(two_label)[0], 0, 1)]
    yt = diag[np.where(two_label_test==np.unique(two_label_test)[0], 0, 1)]
    '''Process data: PCA'''
    # data = (data[:, 0:1024] + data[:, 1024:2048] + data[:, 2048:3072]) / 3 # gray data
    X, Xv, y, yv = train_test_split(two_class, two_label, test_size=0.2, stratify=two_label)
    pca = PCA(n_components=features)
    pca.fit(X)
    X = pca.transform(X)
    Xv = pca.transform(Xv)
    Xt = pca.transform(two_class_test)
    # print(X.shape, Xv.shape, Xt.shape)
    # print(y.shape, yv.shape, yt.shape)
    # exit()
    return X, y, Xv, yv, Xt, yt



if __name__ == '__main__':
    FFT_MNIST()
