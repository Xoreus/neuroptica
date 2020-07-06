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

def test_PT(ONN, model, show_progress=True):
    onn.same_phase_uncert = False
    print('Phi + Theta')
    onn.accuracy_PT = calc_acc.get_accuracy(ONN, model, onn.Xt, onn.yt, loss_diff=onn.loss_diff, show_progress=show_progress)
    onn.PT_FoM = np.sum((np.array(onn.accuracy_PT) > onn.zeta*np.max(onn.accuracy_PT))*onn.PT_Area)
    return ONN, model

def test_LPU(ONN, model, show_progress=True):
    onn.same_phase_uncert = True
    print('Loss + Phase Uncertainty')
    onn.accuracy_LPU = calc_acc.get_accuracy(ONN, model, onn.Xt, onn.yt, loss_diff=onn.loss_diff, show_progress=show_progress)
    onn.LPU_FoM = np.sum((np.array(onn.accuracy_LPU) >  onn.zeta*np.max(onn.accuracy_LPU))*onn.LPU_Area)
    return ONN, model

def colormap_me(ONN):
    acc_colormap.colormap_me(ONN)

def test_onn(ONN, model, show_progress=True):
    ONN, model = test_PT(ONN, model, show_progress=show_progress)
    ONN, model = test_LPU(ONN, model, show_progress=show_progress)
    acc_colormap.colormap_me(ONN)
    return ONN, model

