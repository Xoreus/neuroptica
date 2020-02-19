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

def plot_scatter_matrix(X, Y,  figsize=(15, 15)):
    plt.rcParams.update({'font.size': 44})
    df = pd.DataFrame(X)
    df['Labels'] = [np.argmax(y) for y in Y]

    #now plot using pandas
    color_wheel = {0: 'r', 1: 'b', 2: 'g', 3: 'k', 4: 'c', 5: 'm', 6: 'y', 7:'tab:blue', 8:'tab:orange',9:'tab:purple'}
    colors = df["Labels"].map(lambda x: color_wheel.get(x))

    # features = ['I$_%d$' % x for x in range(1, len(X[1])+1)]
    features = ['$X_%d$' % x for x in range(len(X[1]))]

    # df = df.rename(columns={v:f'I$_{v+1}$' for v in range(len(X))})
    df = df.rename(columns={v:f'$X_{v}$' for v in range(len(X))})

    axes = scatter_matrix(df[features], alpha=.8, figsize=figsize,
                          diagonal='kde',
                          color=colors, s=100, range_padding=0.1)


    for item in axes:
        for idx, ax in enumerate(item):
            # ax.set_yticks([0, 0.5, 1])
            # ax.set_xticks([0, 0.5, 1])
#            ax.set_yticklabels([0, 0.5, 1])
#            ax.set_xticklabels([0, 0.5, 1])
            ax.set_yticklabels('')
            ax.set_xticklabels('')


#            ax.subplots_adjust(hspace=3)

            # We change the fontsize of minor ticks label
#            ax.set_ylim([-0.2, 1.2])
#            ax.set_xlim([-0.05, 1.1])
            ax.tick_params(axis='both', which='major', labelsize=24)
    #        ax.tick_params(axis='both', which='minor', labelsize=42)
            ax.tick_params(axis='both', pad=10)
#            ax.tick_params(axis='x', pad=30)
            ax.xaxis.labelpad = 20
    return axes

if __name__ == "__main__":
    import os
    SAMPLES = 300
    # X, Y, Xt, Yt = blob_maker(nsamples=SAMPLES)
    # FOLDER = '/home/simon/Documents/Thesis/Figures'
    # if not os.path.isdir(FOLDER):
    #     os.makedirs(FOLDER)
    # axes = plot_scatter_matrix(X, Y)
    # plt.savefig(f"{FOLDER}/Gauss-SampleDataset_{SAMPLES}.png")

    X, y, *_ =  cd.iris_dataset(nsamples=SAMPLES)
    axes = plot_scatter_matrix(X, y)
    
    # plt.show()
    # X = (X - np.min(X))/(np.max(X) - np.min(X))
    plt.savefig(f"Figures/Iris-SampleDataset_{SAMPLES}.pdf")

