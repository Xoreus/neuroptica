''' Main function for simulating Optical Neural Network
using Neuroptica and the MNIST dataset

Author: Simon Geoffroy-Gagnon
Edit: 19.02.2020
'''
import numpy as np
import calculate_accuracy as calc_acc
import ONN_Simulation_Class as ONN_Cls
import onnClassTraining
import acc_colormap

onn = ONN_Cls.ONN_Simulation()
onn.BATCH_SIZE = 2**6
onn.EPOCHS = 200
onn.STEP_SIZE = 0.005
onn.ITERATIONS = 10 # number of times to retry same loss/PhaseUncert
onn.loss_diff = 0 # \sigma dB
onn.loss_dB = np.linspace(0, 1, 41)
onn.phase_uncert_theta = np.linspace(0, .5, 41)
onn.phase_uncert_phi = np.linspace(0., .5, 41)
onn.dataset_name = 'MNIST'
onn.SAMPLES = 3000

onn.rng = 4
onn.zeta = 0

onn_topo = ['R_P', 'C_Q_P', 'E_P']
onn_topo = ['R_I_P','R_D_I_P']
onn_topo = ['R_P','I_P']
onn_topo = ['R_P','R_I_P','R_D_I_P', 'R_D_P', 'C_Q_P','E_P']
# onn_topo = ['R_P']
for N in [10]:
    for onn.topo in onn_topo:
        onn.get_topology_name()
        onn.N = N
        onnClassTraining.create_dataset(onn) 
        for onn.rng in range(22):
            onn.phases = []
            model, *_ =  onnClassTraining.train_single_onn(onn)
            if max(onn.val_accuracy) > 10:
                # onn.loss_diff = 0 # \sigma dB
                # onn.accuracy = calc_acc.get_accuracy(onn, model, onn.Xt, onn.yt, loss_diff=onn.loss_diff)
                onn.FOLDER = f'Analysis/MNIST/N={N}'

                onn.createFOLDER()

                onn.same_phase_uncert = False
                print('Different Phase Uncert')
                onn.accuracy_PT = calc_acc.get_accuracy(onn, model, onn.Xt, onn.yt, loss_diff=onn.loss_diff)

                onn.same_phase_uncert = True
                print('Same Phase Uncert')
                onn.accuracy_LPU = calc_acc.get_accuracy(onn, model, onn.Xt, onn.yt, loss_diff=onn.loss_diff)

                # acc_colormap.colormap_me(onn)

                onn.saveAll(model)
                np.savetxt(f'{onn.FOLDER}/all_topologies.txt', onn_topo, fmt='%s')
                break
