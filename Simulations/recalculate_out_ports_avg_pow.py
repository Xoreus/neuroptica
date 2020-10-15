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
import training_onn as train
import test_trained_onns as test
import ONN_Setups
import create_datasets
import training_onn as train
import test_trained_onns as test
from collections import defaultdict
import glob

onn = ONN_Cls.ONN_Simulation()

onn_topo = ['R_P','B_C_Q_P','E_P','R_I_P']

output_pwer = defaultdict(list)
input_pwer = defaultdict(list)
rng = 3

loss = 0.5
sigma = 0
Ns = [4, 8, 16, 32, 64] 
Ns = [16, 32] 
# Ns = [64] 
for N in Ns:
    fileList = glob.glob(f'Analysis/outPorts_mean_pow/N={N}*')
    for ii, FOLDER in enumerate(fileList):
        for onn.topo in onn_topo:
            onn.FOLDER = FOLDER
            if len(glob.glob(onn.FOLDER + '/' + onn.topo + '*')):
                onn.get_topology_name()
                print(f'{N},\t {ii},\t {loss}, \t {onn.topo}')
                onn = onn.pickle_load()
                onn.loss_diff = 0.5

                onn.FOLDER = f'Analysis/outPorts_mean_pow_{loss}/N={N}_{ii}'

                model = ONN_Setups.ONN_creation(onn)
                model.set_all_phases_uncerts_losses(onn.phases, 0, 0, loss, 0)
                out = []
                for x in onn.X:
                    out.append(model.forward_pass(np.array([x]).T))
                output_pwer[onn.topo].append(np.mean(np.sum(out, axis=1)))
                input_pow = [[x**2 for x in sample] for sample in onn.X]
                onn.out_pwer = out
                onn.in_pwer = input_pow
                onn.createFOLDER()
                onn.saveAll(model)
                onn.pickle_save()
