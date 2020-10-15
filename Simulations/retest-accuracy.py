''' 
Script to retest accuracy while changing dataset range
Author: Simon Geoffroy-Gagnon
Edit: 2020.07.11
'''
import numpy as np
import calculate_accuracy as calc_acc
import ONN_Simulation_Class as ONN_Cls
import sys
sys.path.append('../')
import neuroptica as neu


onn = ONN_Cls.ONN_Simulation()
onn = onn.pickle_load(f'Simulations/Analysis/3l/N=16')
model = onn.model

onn.range_dB = 10

onn.loss_dB = np.linspace(0, 2, 2) # set loss/MZI range
onn.phase_uncert_theta = np.linspace(0., 0.75, 2) # set theta phase uncert range
onn.phase_uncert_phi = np.linspace(0., 0.75, 2) # set phi phase uncert range

Xtn = 10*np.log10(np.abs(onn.Xt)**2+sorted(set(np.abs(onn.Xt).reshape(-1)))[1])
Xtn = ((Xtn - np.min(Xtn))/(np.max(Xtn) - np.min(Xtn)) - 1)*onn.range_dB
Xtn = 10**(Xtn/10)
# Xtn = onn.Xt

print(Xtn)
print('Test Accuracy of new power inputs = {:.2f}%'.format(calc_acc.accuracy(onn, model, Xtn, onn.yt)))

