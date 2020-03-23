''' phase_uncert_thetar simulating Optical Neural Network
using Neuroptica and linearly separable datasets
Now goes over every topology types with N = 4-32
Gets the average of power at each ports, this time recalculating for lossy MZIs

Author: Simon Geoffroy-Gagnon
Edit: 2020.03.22
'''

import numpy as np
import calculate_accuracy as calc_acc
import ONN_Simulation_Class as ONN_Cls
import onnClassTraining
import acc_colormap
import N_Finder
import ONN_Setups
from collections import defaultdict


ONN = ONN_Cls.ONN_Simulation()

onn_topo = ['R_D_I_P','R_P','C_Q_P','R_D_P','E_P','R_I_P']
# onn_topo = ['R_D_I_P']
output_pwer = defaultdict(list)
input_pwer = defaultdict(list)
rng = 3
ONN.Ns = [4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32] 
ONN.Ns = [4, 8, 6, 10, 12] 
for ii in range(10):
    for ONN.N in ONN.Ns:
        for ONN.topo in onn_topo:
            ONN.get_topology_name()
            print(ONN.N, ONN.topo, ii)
            ONN.FOLDER = f'/home/simon/Documents/neuroptica/tests/Analysis/outPorts_mean_pow/N={ONN.N}_{ii}'
            ONN = ONN.pickle_load()

            ONN.FOLDER = f'/home/simon/Documents/neuroptica/tests/Analysis/outPorts_mean_pow_Lossy_half_dB_loss/N={ONN.N}_{ii}'

            model = ONN_Setups.ONN_creation(ONN)
            model.set_all_phases_uncerts_losses(ONN.phases, 0, 0, 0.5, 0)
            X, _, Xt, _ = onnClassTraining.change_dataset_shape(ONN)
            out = []
            for x in X:
                out.append(model.forward_pass(np.array([x]).T))
            output_pwer[ONN.topo].append(np.mean(np.sum(out, axis=1)))
            input_pow = [[x**2 for x in sample] for sample in X]
            ONN.out_pwer = out
            ONN.in_pwer = input_pow
            ONN.createFOLDER()
            ONN.saveAll(model)
            ONN.pickle_save()


