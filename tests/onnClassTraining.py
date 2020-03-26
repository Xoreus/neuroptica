"""
singleLossAnalysis.py - trains topology and tests with diff losses/phase uncerts (phi and theta)
Saves all required files for plotting in matlab (matlab is way better an making nice graphs...)

Author: Simon Geoffroy-Gagnon
Edit: 11.02.2020
"""
import sys
import random
import numpy as np
import pickle
import matplotlib as mpl
mpl.use('Agg')
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
import time
import os
from copy import deepcopy

import ONN_Setups
import create_datasets as cd
import calculate_accuracy as calc_acc
import ONN_Simulation_Class as ONN_Cls
sys.path.append(r'C:\Users\sgeoff1\Documents\neuroptica')
sys.path.append('/home/simon/Documents/neuroptica')
import neuroptica as neu

def change_dataset_shape(onn):
    if 'C' in onn.topo and 'Q' in onn.topo:
        X = np.array([list(np.zeros(int((onn.N-2)))) + list(samples) for samples in onn.X])
        Xt = np.array([list(np.zeros(int((onn.N-2)))) + list(samples) for samples in onn.Xt])
    elif 'C' in onn.topo and 'W' in onn.topo:
        X = (np.array([list(np.zeros(int((onn.N-2)/2))) + list(samples) +
            list(np.zeros(int(np.ceil((onn.N-2)/2)))) for samples in onn.X]))
        Xt = (np.array([list(np.zeros(int((onn.N-2)/2))) + list(samples) +
            list(np.zeros(int(np.ceil((onn.N-2)/2)))) for samples in onn.Xt]))
    else:
        X, Xt = onn.X, onn.Xt

    return X, onn.y, Xt, onn.yt

def retrain_ONN(ONN, rng_range):
    if len(rng_range) != 0: print('Starting different rng simulations')
    for ONN.rng in rng_range:
        ONN_Training(ONN, create_dataset_flag=False)

def train_single_onn(onn):
    X, y, Xt, yt = onn.normalize_dataset()
    t = time.time()
    print(f'\nmodel: {onn.topo}, Loss/MZI = {onn.loss_dB[0]:.2f} dB, Loss diff = {onn.loss_diff}, Phase Uncert = {onn.phase_uncert_theta[0]:.2f} Rad, dataset = {onn.dataset_name}, rng = {onn.rng}, N = {onn.N}')

    model = ONN_Setups.ONN_creation(onn)

    X, y, Xt, yt = change_dataset_shape(onn)

    # initialize the ADAM optimizer and fit the ONN to the training data
    optimizer = neu.InSituAdam(model, neu.MeanSquaredError, step_size=onn.STEP_SIZE)
    onn.losses, onn.trn_accuracy, onn.val_accuracy, onn.phases, onn.best_trf_matrix = optimizer.fit(X.T, y.T, Xt.T, yt.T, epochs=onn.EPOCHS, batch_size=onn.BATCH_SIZE, show_progress=True)
    print(f'time spent for current training: {(time.time() - t)/60:.2f} minutes')
    return onn, model

def ONN_Training(ONN, digits=[1,3,6,7], create_dataset_flag=True, zeta=0):
    ONN_Classes = []
    if create_dataset_flag: create_dataset(ONN, digits=digits)
    for onn in ONN.ONN_setup:
        ONN_Classes.append(deepcopy(ONN))
        ONN_Classes[-1].topo = onn
        ONN_Classes[-1].get_topology_name()
    for onn in ONN_Classes:
        onn, model = train_single_onn(onn)
        onn.accuracy = calc_acc.get_accuracy(onn, model, onn.Xt, onn.yt, loss_diff=onn.loss_diff)
        create_folder(onn)
        onn.saveAll(model)

    folder = os.path.split(onn.FOLDER)[-1]
    print('\n' + folder + '\n')
    ONN.FOLDER = onn.FOLDER
    ONN.get_all_topologies()
    ONN.X, ONN.Xt, ONN.y, ONN.yt = onn.X, onn.Xt, onn.y, onn.yt
    with open(onn.FOLDER + '/ONN_Pickled_Class.P', 'wb') as f:
        pickle.dump(ONN, f)
    ONN.saveSelf()

