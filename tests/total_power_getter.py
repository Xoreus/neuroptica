''' phase_uncert_thetar simulating Optical Neural Network
using Neuroptica and linearly separable datasets
Now goes over every topology types with N = 4-32

Author: Simon Geoffroy-Gagnon
Edit: 2020.03.20
'''
import numpy as np
import calculate_accuracy as calc_acc
import ONN_Simulation_Class as ONN_Cls
import onnClassTraining
import acc_colormap
import ONN_Setups
import N_Finder
from collections import defaultdict
import scipy.io

ONN = ONN_Cls.ONN_Simulation()
ONN.BATCH_SIZE = 2**6
ONN.EPOCHS = 300
ONN.STEP_SIZE = 0.005
ONN.ITERATIONS = 5 # number of times to retry same loss/PhaseUncert
ONN.loss_diff = 0 # \sigma dB
ONN.zeta = 0.75

ONN.PT_Area = (ONN.phase_uncert_phi[1] - ONN.phase_uncert_phi[0])**2
ONN.LPU_Area = (ONN.loss_dB[1] - ONN.loss_dB[0])*(ONN.phase_uncert_phi[1] - ONN.phase_uncert_phi[0])
ONN.loss_dB = np.linspace(0, 1, 5)
ONN.phase_uncert_theta = np.linspace(0., 0.4, 5)
ONN.phase_uncert_phi = np.linspace(0., 0.4, 5)

ONN.rng = 2

onn_topo = ['R_D_I_P', 'R_P', 'C_Q_P','R_D_P', 'E_P']
output_pwer = defaultdict(list)
input_pwer = defaultdict(list)
for ii in range(1):
    for ONN.N in [4, 6, 8, 10, 14, 16, 20, 24, 28, 32]:
        ONN.FOLDER = f'/home/simon/Documents/neuroptica/tests/Analysis/pwer_out/N={ONN.N}_{ii}'
        rng = N_Finder.get_dataset(folder, ONN.N, rng, EPOCHS=50)
        for ONN.topo in onn_topo:
            ONN.get_topology_name()
            model = ONN_Setups.ONN_creation(onn)

            

            X, _, Xt, _ = onnClassTraining.change_dataset_shape(ONN)
            out = []
            if model != 0:
                for x in X:
                    out.append(model.forward_pass(np.array([x]).T))
                output_pwer[ONN.topo].append(np.mean(out))

        input_pwer[ONN.N].append(np.mean(X))

        print(output_pwer)
        print(np.mean(ONN.X))
        ONN.out_pwer = output_pwer
        ONN.in_pwer  = input_pwer
        ONN.saveSelf()


