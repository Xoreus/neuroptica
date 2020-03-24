''' phase_uncert_thetar simulating Optical Neural Network
using Neuroptica and linearly separable datasets
Now goes over every topology types with N = 96 and trains them ONLY

Author: Simon Geoffroy-Gagnon
Edit: 2020.03.09
'''
import numpy as np
import calculate_accuracy as calc_acc
import ONN_Simulation_Class as ONN_Cls
import onnClassTraining
import acc_colormap
import digital_NN_main as dnn
import create_datasets as cd
import random
import os
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

def get_dataset(ONN, folder, rng, lim=99, SAMPLES=100, EPOCHS=30):
    while True:
        print(f'RNG = {rng}, N = {ONN.N}')
        X, y, Xt, yt = cd.gaussian_dataset(targets=int(ONN.N), features=int(ONN.N), nsamples=SAMPLES*ONN.N, rng=rng)
        random.seed(rng)

        X = (X - np.min(X))/(np.max(X) - np.min(X))
        Xt = (Xt - np.min(Xt))/(np.max(Xt) - np.min(Xt))

        net, weights = dnn.create_train_dnn(X, y, Xt, yt, folder, EPOCHS)
        print('Validation Accuracy: {:.1f}%'.format(dnn.get_current_accuracy(Xt, yt, net)*100))
        rng += 1
        if dnn.get_current_accuracy(Xt, yt, net)*100 > lim:
            if not os.path.isdir(folder):
                os.makedirs(folder)

            ONN.X = X
            ONN.y = y
            ONN.Xt = Xt
            ONN.yt = yt
            print('This dataset works!\n')
            return ONN, rng

def test_onn(folder, ONN, lim=98.5):
    ONN.get_topology_name()
    ONN.X = np.loadtxt(folder + f'/X.txt', delimiter=',')
    ONN.y = np.loadtxt(folder + f'/y.txt', delimiter=',')
    ONN.Xt = np.loadtxt(folder + f'/Xt.txt', delimiter=',')
    ONN.yt = np.loadtxt(folder + f'/yt.txt', delimiter=',')
    for ONN.rng in range(10):
        ONN.phases = []
        if max(ONN.val_accuracy) > lim:
            ONN.same_phase_uncert = False
            print('Different Phase Uncert')
            ONN.accuracy_PT = calc_acc.get_accuracy(ONN, model, ONN.Xt, ONN.yt, loss_diff=ONN.loss_diff)
            ONN.PT_FoM = np.sum((np.array(ONN.accuracy_PT) > ONN.zeta*np.max(ONN.accuracy_PT))*ONN.PT_Area)
            print(ONN.topology, ONN.PT_FoM, 'Rad^2')
            ONN.same_phase_uncert = True
            print('Same Phase Uncert')
            ONN.accuracy_LPU = calc_acc.get_accuracy(ONN, model, ONN.Xt, ONN.yt, loss_diff=ONN.loss_diff)
            ONN.LPU_FoM = np.sum((np.array(ONN.accuracy_LPU) >  ONN.zeta*np.max(ONN.accuracy_LPU))*ONN.LPU_Area)
            print(ONN.topology, ONN.LPU_FoM, 'Rad*dB')

            ONN.createFOLDER()
            acc_colormap.colormap_me(ONN)
            ONN.saveAll(model)

            return ONN, model
    return ONN, 0

if __name__ == '__main__':
    ONN = ONN_Cls.ONN_Simulation()
    ONN.BATCH_SIZE = 2**6
    ONN.EPOCHS = 300
    ONN.STEP_SIZE = 0.005
    ONN.ITERATIONS = 5 # number of times to retry same loss/PhaseUncert
    ONN.loss_diff = 0 # \sigma dB
    ONN.loss_dB = np.linspace(0, 1, 41)
    ONN.phase_uncert_theta = np.linspace(0., .5, 41)
    ONN.phase_uncert_phi = np.linspace(0., .5, 41)
    ONN.rng = 2
    ONN.zeta = 0.75
    ONN.PT_Area = (ONN.phase_uncert_phi[1] - ONN.phase_uncert_phi[0])**2
    ONN.LPU_Area = (ONN.loss_dB[1] - ONN.loss_dB[0])*(ONN.phase_uncert_phi[1] - ONN.phase_uncert_phi[0])

    onn_topo = ['R_P', 'C_Q_P', 'E_P', 'R_I_P']

    rng = 192
    ONN.N = 6 
    FoM = {}
    for ii in range(1):
        while True:
            ONN.FOLDER = f'Analysis/N={ONN.N}/N={ONN.N}_{ii}'
            folder = f'../better-linsep-datasets/N={ONN.N}'
            ONN, rng = get_dataset(ONN, folder, rng)
            PT_FoM = {}
            for ONN.topo in onn_topo:
                ONN.get_topology_name()
                ONN, model = onnClassTraining.train_single_onn(ONN)
                ONN.createFOLDER()
                ONN.pickle_save()
                ONN.saveAll(model)
            np.savetxt(f'{ONN.FOLDER}/all_topologies.txt', onn_topo, fmt='%s')  
            break
