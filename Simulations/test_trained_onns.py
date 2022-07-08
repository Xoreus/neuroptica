''' phase_uncert_thetar simulating Optical Neural Network
using Neuroptica and linearly separable datasets
Simply short fcns to test trained models with Loss/MZI and Phase Uncert.

Author: Simon Geoffroy-Gagnon
Edit: 2020.03.28
'''
import numpy as np
import calculate_accuracy as calc_acc
import ONN_Simulation_Class as ONN_Cls
import ONN_Setups
import digital_NN_main as dnn
import create_datasets as cd
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

def test_PT(onn, X, y, model, show_progress=True):
    onn.same_phase_uncert = False
    print('\nPhi + Theta')
    onn.accuracy_PT = calc_acc.get_accuracy(onn, model, X, y, loss_diff=onn.loss_diff, show_progress=show_progress)
    onn.PT_FoM = np.sum((np.array(onn.accuracy_PT) > onn.zeta*np.max(onn.accuracy_PT))*onn.PT_Area)
    print(f"-----------------------------------------PT_Accuracy: {np.shape(onn.accuracy_PT)}\n", onn.accuracy_PT)
    return onn, model

def test_LPU(onn, X, y, model, show_progress=True):
    onn.same_phase_uncert = True
    print('\nLoss + Phase Uncertainty')
    onn.accuracy_LPU = calc_acc.get_accuracy(onn, model, X, y, loss_diff=onn.loss_diff, show_progress=show_progress)
    print(f"----------------------------------------LPU_Accuracy: {np.shape(onn.accuracy_LPU)}\n", onn.accuracy_LPU)
    onn.LPU_FoM = np.sum((np.array(onn.accuracy_LPU) >  onn.zeta*np.max(onn.accuracy_LPU))*onn.LPU_Area)
    return onn, model

def test_SLPU(onn, X, y, model, show_progress=True): #Only tests Loss/MZI at 0 dB Phase 
    onn.same_phase_uncert = True
    print('\nLoss + Phase Uncertainty')
    onn.accuracy_LPU = calc_acc.get_accuracy_SLPU(onn, model, X, y, loss_diff=onn.loss_diff, show_progress=show_progress)
    print(f"---------------------------------------SLPU_Accuracy:{np.shape(onn.accuracy_LPU)}\n", onn.accuracy_LPU)
    return onn, model

def test_onn(onn, model, show_progress=True):
    onn, model = test_PT(onn, onn.Xt, onn.yt, model, show_progress=show_progress)
    onn, model = test_LPU(onn, onn.Xt, onn.yt, model, show_progress=show_progress)
    return onn, model

