"""
Saves dataset to FOLDER

Author: Simon Geoffroy-Gagnon
Edit: 15.01.15
"""
import numpy as np
import matplotlib.pyplot as plt
import plot_scatter_matrix as psm
import matplotlib as mpl
import os
import pandas as pd
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import random


def plot_scatter_matrix(X, Y,  figsize=(20, 15)):
    plt.rcParams.update({'font.size': 44})
    df = pd.DataFrame(X)
    df['Labels'] = [np.argmax(y) for y in Y]

    #now plot using pandas
    color_wheel = {0: 'r', 1: 'b', 2: 'g', 3: 'k', 4: 'c', 5: 'm', 6: 'y', 7:'tab:blue', 8:'tab:orange',9:'tab:purple'}
    colors = df["Labels"].map(lambda x: color_wheel.get(x))

    features = ['$X_%d$' % x for x in range(len(X[1]))]

    df = df.rename(columns={v:'$X_%d$' % v for v in range(len(X))})


    axes = scatter_matrix(df[features], alpha=.8, figsize=figsize,
                          diagonal='kde',
                          color=colors, s=100, range_padding=0.1)


    for item in axes:
        for idx, ax in enumerate(item):
            ax.set_yticks([0, 0.5, 1])
            ax.set_xticks([0, 0.5, 1])
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

def createFOLDER(FOLDER):
    if not os.path.isdir(FOLDER):
        os.makedirs(FOLDER)
    if not os.path.isdir(FOLDER + '/Figures'):
        os.makedirs(FOLDER + '/Figures')
    if not os.path.isdir(FOLDER + '/Figures_Fitting'):
        os.makedirs(FOLDER + '/Figures_Fitting')
    if not os.path.isdir(FOLDER + '/Data_Fitting'):
        os.makedirs(FOLDER + '/Data_Fitting')
    if not os.path.isdir(FOLDER + '/Phases'):
        os.makedirs(FOLDER + '/Phases')
    if not os.path.isdir(FOLDER + '/Datasets'):
        os.makedirs(FOLDER + '/Datasets')
    if not os.path.isdir(FOLDER + '/Losses_per_MZI'):
        os.makedirs(FOLDER + '/Losses_per_MZI')

def saveSimData(FOLDER, dataset_name, ii, N, X, y, Xt, yt):
    axes = plot_scatter_matrix(X, y)
    plt.savefig(f'{FOLDER}/Datasets/{dataset_name}_Samples={len(X)}_Dataset#{ii}.png')
    plt.clf()
    mpl.rcParams.update(mpl.rcParamsDefault)

    np.savetxt(f'{FOLDER}/Datasets/{dataset_name}_X_{N}Features_Samples={len(X)}_Dataset#{ii}.txt',X, delimiter=',',fmt='%.3f')
    np.savetxt(f'{FOLDER}/Datasets/{dataset_name}_y_{N}Features_Samples={len(X)}_Dataset#{ii}.txt',y, delimiter=',',fmt='%.3f')
    np.savetxt(f'{FOLDER}/Datasets/{dataset_name}_Xt_{N}Features_Samples={len(X)}_Dataset#{ii}.txt',Xt, delimiter=',',fmt='%.3f')
    np.savetxt(f'{FOLDER}/Datasets/{dataset_name}_yt_{N}Features_Samples={len(X)}_Dataset#{ii}.txt',yt, delimiter=',',fmt='%.3f')
