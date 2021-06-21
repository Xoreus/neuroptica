''' phase_uncert_thetar simulating Optical Neural Network
using Neuroptica and linearly separable datasets
Now goes over every topology types with N = 4-32
Gets the average of the sum of input and output powers for all topologies

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

onn_topo = ['R_D_I_P','R_P','C_Q_P','R_D_P','E_P','R_I_P']
output_pwer = defaultdict(list)
input_pwer = defaultdict(list)
rng = 3
ONN.Ns = [4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32] 
for ii in [4]:
    for ONN.N in ONN.Ns:
        folder = f'/home/edwar/Documents/Github_Projects/neuroptica/linsep-datasets/N={ONN.N}'
        ONN.FOLDER = f'/home/edwar/Documents/Github_Projects/neuroptica/tests/Analysis/pwer_out/N={ONN.N}_{ii}'
        rng = N_Finder.get_dataset(folder, ONN.N, rng, EPOCHS=50)
        for ONN.topo in onn_topo:
            ONN.get_topology_name()
            ONN, model = N_Finder.test_onn(folder, ONN, lim=80)

            model.set_all_phases_uncerts_losses(ONN.phases, 0, 0, 0, 0)
            X, _, Xt, _ = onnClassTraining.change_dataset_shape(ONN)
            out = []
            if model != 0:
                for x in X:
                    out.append(model.forward_pass(np.array([x]).T))
                output_pwer[ONN.topo].append(np.mean(np.sum(out, axis=1)))
                input_pow = [[x**2 for x in sample] for sample in X]
                ONN.out_pwer = output_pwer
                ONN.in_pwer = input_pow

                print(output_pwer)

        input_pwer[ONN.N].append(np.mean(np.sum(input_pow, axis=1)))
        print(input_pwer)

        in_pwer = np.mean(np.sum(ONN.X, axis=1))
        ONN.out_pwer = output_pwer
        ONN.in_pwer  = input_pwer
        ONN.saveSelf()

