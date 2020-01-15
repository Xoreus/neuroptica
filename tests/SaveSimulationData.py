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

def saveSimData(FOLDER, dataset_name, ii, N, X, y, Xt, yt):
    axes = psm.plot_scatter_matrix(X, y)
    plt.savefig(f'{FOLDER}/Datasets/{dataset_name}_Dataset#{ii}.png')
    plt.clf()
    mpl.rcParams.update(mpl.rcParamsDefault)

    np.savetxt(f'{FOLDER}/Datasets/{dataset_name}_X_{N}Features_Dataset#{ii}.txt',X, delimiter=',',fmt='%.3f')
    np.savetxt(f'{FOLDER}/Datasets/{dataset_name}_y_{N}Features_Dataset#{ii}.txt',y, delimiter=',',fmt='%.3f')
    np.savetxt(f'{FOLDER}/Datasets/{dataset_name}_Xt_{N}Features_Dataset#{ii}.txt',Xt, delimiter=',',fmt='%.3f')
    np.savetxt(f'{FOLDER}/Datasets/{dataset_name}_yt_{N}Features_Dataset#{ii}.txt',yt, delimiter=',',fmt='%.3f')

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
