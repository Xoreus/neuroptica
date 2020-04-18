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
import ONN_Setups
from collections import defaultdict
import glob

ONN = ONN_Cls.ONN_Simulation()

onn_topo = ['R_D_I_P','R_P','C_Q_P','R_D_P','E_P','R_I_P']
onn_topo = ['R_P','C_Q_P','E_P','R_I_P']

output_pwer = defaultdict(list)
input_pwer = defaultdict(list)
rng = 3

loss = 0.5
sigma = 0
Ns = [4, 6, 8, 10, 12, 16, 32, 64] 
for N in Ns:
    fileList = glob.glob(f'/home/simon/Documents/neuroptica/tests/Analysis/{0:.2f}_sigma{0:.2f}_outPorts_mean_pow/N={N}*')
    for ii, FOLDER in enumerate(fileList):
        for ONN.topo in onn_topo:
            ONN.FOLDER = FOLDER
            if len(glob.glob(ONN.FOLDER + '/' + ONN.topo + '*')):
                ONN.get_topology_name()
                print(N, ONN.topo, ii, loss)
                ONN = ONN.pickle_load()
                ONN.loss_diff = 0.5

                ONN.FOLDER = f'/home/simon/Documents/neuroptica/tests/Analysis/{loss:.2f}_sigma{ONN.loss_diff:.2f}_outPorts_mean_pow/N={N}_{ii}'

                model = ONN_Setups.ONN_creation(ONN)
                model.set_all_phases_uncerts_losses(ONN.phases, 0, 0, loss, 0)
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
