''' phase_uncert_thetar simulating Optical Neural Network
using Neuroptica and linearly separable datasets
Simply short fcns to test trained models with Loss/MZI and Phase Uncert.

Author: Simon Geoffroy-Gagnon
Edit: 2020.03.28
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
import training_onn

if 1:
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    matplotlib.rcParams['mathtext.fontset'] = 'custom'
    matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

def test_PT(ONN, model):
    ONN.same_phase_uncert = False
    print('Different Phase Uncert')
    ONN.accuracy_PT = calc_acc.get_accuracy(ONN, model, ONN.Xt, ONN.yt, loss_diff=ONN.loss_diff)
    ONN.PT_FoM = np.sum((np.array(ONN.accuracy_PT) > ONN.zeta*np.max(ONN.accuracy_PT))*ONN.PT_Area)
    # print(ONN.topology, ONN.PT_FoM, 'Rad^2')
    return ONN, model

def test_LPU(ONN, model):
    ONN.same_phase_uncert = True
    print('Same Phase Uncert')
    ONN.accuracy_LPU = calc_acc.get_accuracy(ONN, model, ONN.Xt, ONN.yt, loss_diff=ONN.loss_diff)
    ONN.LPU_FoM = np.sum((np.array(ONN.accuracy_LPU) >  ONN.zeta*np.max(ONN.accuracy_LPU))*ONN.LPU_Area)
    # print(ONN.topology, ONN.LPU_FoM, 'Rad*dB')
    return ONN, model

def colormap_me(ONN):
    acc_colormap.colormap_me(ONN)

def test_onn(ONN, model):
    ONN, model = test_PT(ONN, model)
    ONN, model = test_LPU(ONN, model)
    acc_colormap.colormap_me(ONN)
    return ONN, model


if __name__ == '__main__':
    ONN = ONN_Cls.ONN_Simulation()
    ONN.BATCH_SIZE = 2**6
    ONN.EPOCHS = 300
    ONN.STEP_SIZE = 0.005
    ONN.ITERATIONS = 5 # number of times to retry same loss/PhaseUncert
    ONN.loss_diff = 0 # \sigma dB
    ONN.rng = 2
    ONN.zeta = 0.75
    ONN.PT_Area = (ONN.phase_uncert_phi[1] - ONN.phase_uncert_phi[0])**2
    ONN.LPU_Area = (ONN.loss_dB[1] - ONN.loss_dB[0])*(ONN.phase_uncert_phi[1] - ONN.phase_uncert_phi[0])
    ONN.val_accuracy = 0

    onn_topo = ['R_P', 'C_Q_P', 'E_P', 'R_I_P']
    onn_topo = ['R_P']

    ONN.N = 4 
    for ii in range(1):
        ONN.FOLDER = f'Analysis/N={ONN.N}/N={ONN.N}_{ii}'
        ONN.FOLDER = f'/home/simon/Documents/neuroptica/tests/Analysis/N=4/N=4_2_100.000'
        for ONN.topo in onn_topo:
            print('\n',ONN.topo)
            ONN.get_topology_name()
            ONN = ONN.pickle_load()

            ONN.loss_dB = np.linspace(0, 1, 4)
            ONN.phase_uncert_theta = np.linspace(0., .5, 4)
            ONN.phase_uncert_phi = np.linspace(0., .5, 4)

            model = ONN_Setups.ONN_creation(ONN)
            model.set_all_phases_uncerts_losses(ONN.phases, 0, 0, 0, 0)

            ONN, model = test_onn(ONN, lim=50)
            ONN.createFOLDER()
            ONN.pickle_save()
            ONN.saveAll(model)
        np.savetxt(f'{ONN.FOLDER}/all_topologies.txt', onn_topo, fmt='%s')  

