''' phase_uncert_thetar simulating Optical Neural Network
using Neuroptica and linearly separable datasets
Now goes over every topology types with N = 4-32

Author: Simon Geoffroy-Gagnon
Edit: 2020.03.09
'''
import numpy as np
import calculate_accuracy as calc_acc
import ONN_Simulation_Class as ONN_Cls
import onnClassTraining
import acc_colormap
import ONN_Setups
import digital_NN_main as dnn
import create_datasets as cd
import random
import os
import matplotlib
import N_Finder


matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'


def test_onn(ONN, lim=98.5):
    ONN.get_topology_name()
    for ONN.rng in range(10):
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
ONN.val_accuracy = 0

onn_topo = ['R_P', 'C_Q_P', 'E_P', 'R_I_P']

ONN.N = 6 
for ii in range(1):
    ONN.FOLDER = f'Analysis/N={ONN.N}/N={ONN.N}_{ii}'
    for ONN.topo in onn_topo:
        print('\n',ONN.topo)
        ONN.get_topology_name()
        ONN = ONN.pickle_load()
        model = ONN_Setups.ONN_creation(ONN)
        model.set_all_phases_uncerts_losses(ONN.phases, 0, 0, 0, 0)

        ONN, model = test_onn(ONN, lim=50)
        ONN.createFOLDER()
        ONN.pickle_save()
        ONN.saveAll(model)
    np.savetxt(f'{ONN.FOLDER}/all_topologies.txt', onn_topo, fmt='%s')  

