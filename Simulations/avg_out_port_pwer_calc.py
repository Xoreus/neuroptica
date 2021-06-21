''' phase_uncert_thetar simulating Optical Neural Network
using Neuroptica and linearly separable datasets
Now goes over every topology types with N = 4-32
Gets the average power of each output ports

Author: Simon Geoffroy-Gagnon
Edit: 2020.03.09
'''

import numpy as np
import calculate_accuracy as calc_acc
import ONN_Simulation_Class as ONN_Cls
import onnClassTraining
import N_Finder
from collections import defaultdict


ONN = ONN_Cls.ONN_Simulation()
ONN.BATCH_SIZE = 2**6
ONN.EPOCHS = 300
ONN.STEP_SIZE = 0.005
ONN.ITERATIONS = 5 # number of times to retry same loss/PhaseUncert
ONN.loss_diff = 0 # \sigma dB
ONN.zeta = 0.75

ONN.PT_Area = (ONN.phase_uncert_phi[1] - ONN.phase_uncert_phi[0])**2
ONN.LPU_Area = (ONN.loss_dB[1] - ONN.loss_dB[0])*(ONN.phase_uncert_phi[1] - ONN.phase_uncert_phi[0])
ONN.loss_dB = np.linspace(0, 1, 2)
ONN.phase_uncert_theta = np.linspace(0., 0.4, 2)
ONN.phase_uncert_phi = np.linspace(0., 0.4, 2)

ONN.rng = 2

onn_topo = ['R_D_I_P','R_P','C_Q_P','R_D_P','E_P', 'R_I_P']
onn_topo = ['R_P','C_Q_P']
output_pwer = defaultdict(list)
input_pwer = defaultdict(list)
ONN.Ns = [4, 6, 8, 10] 


for ii in [3]:
    for ONN.N in ONN.Ns:
        ONN.FOLDER = f'/home/edwar/Documents/Github_Projects/neuroptica/tests/Analysis/outPorts_mean_pow/N={ONN.N}_{ii}'
        for ONN.topo in onn_topo:
            ONN.get_topology_name()
            ONN.pickle_load()

            output_pwer[ONN.topo].append(np.mean(np.sum(ONN.out, axis=1)))
            input_pow = [[x**2 for x in sample] for sample in X]
            ONN.out_pwer = out
            ONN.in_pwer = input_pow
