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
import glob
if 1:
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    matplotlib.rcParams['mathtext.fontset'] = 'custom'
    matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

def test_PT(ONN, model):
    ONN.same_phase_uncert = False
    print('PT')
    ONN.accuracy_PT = calc_acc.get_accuracy(ONN, model, ONN.Xt, ONN.yt, loss_diff=ONN.loss_diff)
    ONN.PT_FoM = np.sum((np.array(ONN.accuracy_PT) > ONN.zeta*np.max(ONN.accuracy_PT))*ONN.PT_Area)
    return ONN, model

def test_LPU(ONN, model):
    ONN.same_phase_uncert = True
    print('LPU')
    ONN.accuracy_LPU = calc_acc.get_accuracy(ONN, model, ONN.Xt, ONN.yt, loss_diff=ONN.loss_diff)
    ONN.LPU_FoM = np.sum((np.array(ONN.accuracy_LPU) >  ONN.zeta*np.max(ONN.accuracy_LPU))*ONN.LPU_Area)
    return ONN, model

def colormap_me(ONN):
    acc_colormap.colormap_me(ONN)

def test_onn(ONN, model):
    ONN, model = test_PT(ONN, model)
    ONN, model = test_LPU(ONN, model)
    acc_colormap.colormap_me(ONN)
    return ONN, model

