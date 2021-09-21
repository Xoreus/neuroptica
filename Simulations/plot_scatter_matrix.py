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
    # SAMPLES = 30_000
    SAMPLES = 300
    X, Y, Xt, Yt = blob_maker(nsamples=SAMPLES, features=2,  targets=2)
    axes = plot_scatter_matrix(X, Y,  figsize=(15, 15), label='X', start_at=0, fontsz=54)
    plt.savefig(f'/home/edwar/Documents/Github_Projects/neuroptica/Simulations/Crop_Me/gauss_S={SAMPLES}.pdf')

