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
import setupSimulation as setSim
import calculate_accuracy as calc_acc
import ONN_Simulation_Class as ONN_Cls
sys.path.append(r'C:\Users\sgeoff1\Documents\neuroptica')
sys.path.append('/home/simon/Documents/neuroptica')
import neuroptica as neu

def create_dataset(onn, digits=[1,3,6,7]):
    if onn.dataset_name == 'MNIST' and onn.N == 4:
        onn.X, onn.y, onn.Xt, onn.yt = cd.MNIST_dataset(digits, nsamples=onn.SAMPLES)
    elif onn.dataset_name == 'MNIST':
        onn.X, onn.y, onn.Xt, onn.yt = cd.MNIST_dataset(N=onn.N, nsamples=onn.SAMPLES)
    elif onn.dataset_name == 'Gaussian':
        onn.X, onn.y, onn.Xt, onn.yt = cd.gaussian_dataset(targets=int(onn.N), features=int(onn.N), nsamples=onn.SAMPLES, rng=onn.rng)
    elif onn.dataset_name == 'Iris':
        onn.X, onn.y, onn.Xt, onn.yt = cd.iris_dataset(nsamples=int(onn.SAMPLES))

def create_folder(onn):
    ROOT_FOLDER = r'Analysis/'
    FUNCTION = 'single_loss_linsep/'
    FOLDER = f'{onn.dataset_name}_N={onn.N}_loss-diff={onn.loss_diff}_rng{onn.rng}'
    onn.FOLDER = ROOT_FOLDER + FUNCTION + FOLDER
    setSim.createFOLDER(onn.FOLDER)

def retrain_ONN(ONN, rng_range):
    if len(rng_range) != 0: print('Starting different rng simulations')
    for ONN.rng in rng_range:
        ONN_Training(ONN, create_dataset_flag=False)

def train_single_onn(onn):
    random.seed(onn.rng)
    X, y, Xt, yt = onn.normalize_dataset()
    t = time.time()
    print(f'model: {onn.onn_topo}, Loss/MZI = {onn.loss_dB[0]:.2f} dB, Loss diff = {onn.loss_diff}, Phase Uncert = {onn.phase_uncert_theta[0]:.2f} Rad, dataset = {onn.dataset_name}, rng = {onn.rng}, N={onn.N}')
    model = ONN_Setups.ONN_creation(onn)

    if 'C' in onn.onn_topo and 'Q' in onn.onn_topo:
        X = np.array([list(np.zeros(int((onn.N-2)))) + list(samples) for samples in onn.X])
        Xt = np.array([list(np.zeros(int((onn.N-2)))) + list(samples) for samples in onn.Xt])
    elif 'C' in onn.onn_topo and 'W' in onn.onn_topo:
        X = (np.array([list(np.zeros(int((onn.N-2)/2))) + list(samples) +
            list(np.zeros(int(np.ceil((onn.N-2)/2)))) for samples in onn.X]))
        Xt = (np.array([list(np.zeros(int((onn.N-2)/2))) + list(samples) +
            list(np.zeros(int(np.ceil((onn.N-2)/2)))) for samples in onn.Xt]))

    # initialize the ADAM optimizer and fit the ONN to the training data
    optimizer = neu.InSituAdam(model, neu.MeanSquaredError, step_size=onn.STEP_SIZE)
    onn.losses, onn.trn_accuracy, onn.val_accuracy, onn.phases, onn.best_trf_matrix = optimizer.fit(X.T, y.T, Xt.T, yt.T, epochs=onn.EPOCHS, batch_size=onn.BATCH_SIZE, show_progress=True)
    print(f'time spent for current training and testing all loss/phase uncert: {(time.time() - t)/60:.2f} minutes')
    return model, onn

def ONN_Training(ONN, digits=[1,3,6,7], create_dataset_flag=True, zeta=0):
    ONN_Classes = []
    if create_dataset_flag: create_dataset(ONN, digits=digits)
    for onn in ONN.ONN_setup:
        ONN_Classes.append(deepcopy(ONN))
        ONN_Classes[-1].onn_topo = onn
        ONN_Classes[-1].get_topology_name()
    for onn in ONN_Classes:
        model, onn = train_single_onn(onn)
        onn.accuracy = calc_acc.get_accuracy(onn, model, onn.Xt, onn.yt, loss_diff=onn.loss_diff, zeta=onn.zeta)
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

if __name__ == '__main__':
    ONN = ONN_Cls.ONN_Simulation()
    ONN.N = 4
    ONN.BATCH_SIZE = 2**6
    ONN.EPOCHS = 1500
    ONN.STEP_SIZE = 0.0005
    ONN.SAMPLES = 1000
    ONN.ITERATIONS = 40 # number of times to retry same loss/PhaseUncert
    ONN.loss_diff = 0 # \sigma dB
    ONN.loss_dB = np.linspace(0, 2, 11)
    ONN.phase_uncert_theta = np.linspace(0., 2.5, 21)
    ONN.phase_uncert_phi = np.linspace(0., 2.5, 21)
    ONN.same_phase_uncert = False
    ONN.rng = 1
    x = 32
    reps = 0

    ONN.dataset_name = 'MNIST'
    # ONN.dataset_name = 'Gaussian'
    # ONN.dataset_name = 'Iris'

    # ONN.ONN_setup = np.array(['R_I_P', 'R_P', 'E_P', 'R_D_I_P', 'R_D_P'])
    # ONN.ONN_setup = np.array(['R_I_P', 'R_D_I_P'])
    ONN.ONN_setup = np.array(['R_D_P', 'R_D_I_P', 'R_I_P', 'C_Q_P', 'E_P'])

    for ONN.rng in range(x, x + reps + 1):
        ONN_Training(ONN, digits=[2,4,5,6], zeta=0)

    print('Starting different rng simulations')
    retrain_ONN(ONN, range(0))
