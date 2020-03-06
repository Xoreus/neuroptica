# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 13:11:50 2019
Plots scatter-matrix of any dataset
@author: sgeoff1
"""

import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import random
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
rc('text', usetex=True)
import create_datasets as cd

rng = 4 
random.seed(rng)

def blob_maker(targets=4, features=4, nsamples=10000,
               cluster_std=.1):

    # generate 2d classification dataset
    X, y = make_blobs(n_samples=nsamples, centers=targets,
                      n_features=features,
                      cluster_std=cluster_std,
                      center_box=(0, 1), random_state=rng, shuffle=False)
    ohe_labels = pd.get_dummies(y).values
    x, xt, y, yt = train_test_split(X, ohe_labels, test_size=0.2)

    return x, y, xt, yt

def plot_scatter_matrix(X, Y,  figsize=(15, 12), label='X', start_at=0, fontsz=64):
    plt.rcParams.update({'font.size': fontsz})
    df = pd.DataFrame(X)
    df['Labels'] = [np.argmax(y) for y in Y]

    #now plot using pandas
    color_wheel = {0:'r', 2:'b', 1:'g', 3:'k', 4:'c', 5:'m', 6:'y', 7:'tab:blue', 8:'tab:orange', 9:'tab:purple'}
    colors = df["Labels"].map(lambda x: color_wheel.get(x))

    features = [f'${label}_%d$' % x for x in range(start_at, len(X[1])+start_at)]
    df = df.rename(columns={v:f'${label}_{v+start_at}$' for v in range(len(X))})

    axes = scatter_matrix(df[features], alpha=.8, figsize=figsize, diagonal='kde',
                          color=colors, s=100, range_padding=0.1)
    plt.title('')


    for item in axes:
        for idx, ax in enumerate(item):
            ax.spines['top'].set_linewidth(1)
            ax.spines['right'].set_linewidth(1)
            ax.spines['left'].set_linewidth(1)
            ax.spines['bottom'].set_linewidth(1)

            # ax.spines['top'].set_color('#d3d3d3')
            # ax.spines['right'].set_color('#d3d3d3')
            # ax.spines['left'].set_color('#d3d3d3')
            # ax.spines['bottom'].set_color('#d3d3d3')
            ax.set_yticklabels('')
            ax.set_xticklabels('')
            ax.tick_params(axis='both', which='major', labelsize=54)
    #        ax.tick_params(axis='both', which='minor', labelsize=42)
            ax.tick_params(axis='both', pad=10)
#            ax.tick_params(axis='x', pad=30)
            ax.xaxis.labelpad = 20
    return axes

if __name__ == "__main__":
    import os
    SAMPLES = 300
    # # X, Y, Xt, Yt = blob_maker(nsamples=SAMPLES)
    # X = np.loadtxt('/home/simon/Documents/neuroptica/linsep-datasets/N=4/X.txt', delimiter=',')
    # y = np.loadtxt('/home/simon/Documents/neuroptica/linsep-datasets/N=4/y.txt', delimiter=',')
    # FOLDER = '/home/simon/Documents/neuroptica/tests/Analysis'
    # if not os.path.isdir(FOLDER):
    #     os.makedirs(FOLDER)
    # axes = plot_scatter_matrix(X, y,  figsize=(14, 14), label='\mathrm{I}', start_at=1)
    # plt.savefig(f"{FOLDER}/gaussDataset.pdf")
    # cd.plot_OG_iris()

    # X, y, *_ =  cd.iris_dataset(nsamples=SAMPLES)
    # axes = plot_scatter_matrix(X, y, figsize=(15, 12))

    # plt.suptitle('', fontname='Calibri', fontsize=34)
    # plt.savefig('/home/simon/Documents/Thesis/Figures/Iris-SampleDataset_300.pdf')
    

    X, y, *_ =  cd.MNIST_dataset(nsamples=SAMPLES, N=4)
    axes = plot_scatter_matrix(X, y, figsize=(15, 12), fontsz=40)

    plt.suptitle('', fontname='Calibri', fontsize=34)
    plt.savefig('/home/simon/Documents/Thesis/Figures/MNIST-SampleDataset_N=4.pdf')

    X, y, *_ =  cd.MNIST_dataset(nsamples=600, N=10)
    axes = plot_scatter_matrix(X, y, figsize=(15, 12), fontsz=40)

    # plt.suptitle('', fontname='Calibri', fontsize=30)
    plt.savefig('/home/simon/Documents/Thesis/Figures/MNIST-SampleDataset_N=10.pdf')
