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
onn.EPOCHS = 600
onn.STEP_SIZE = 0.005
onn.ITERATIONS = 5 # number of times to retry same loss/PhaseUncert
onn.loss_diff = 0 # \sigma dB
onn.loss_dB = np.linspace(0, 1.5, 4)
onn.phase_uncert_theta = np.linspace(0.05, 1.5, 4)
onn.phase_uncert_phi = np.linspace(0., 1.5, 4)
onn.dataset_name = 'MNIST'

onn.rng = 4
onn.zeta = 0

onn_topo = ['R_P']
for N in [4]:
    for onn.onn_topo in onn_topo:
        onn.get_topology_name()
        onn.N = N
        onnClassTraining.create_dataset(onn) 
        for onn.rng in range(20):
            onn.phases = []
            model, *_ =  onnClassTraining.train_single_onn(onn)
            if max(onn.val_accuracy) > 10:
                onn.accuracy = calc_acc.get_accuracy(onn, model, onn.Xt, onn.yt, loss_diff=onn.loss_diff)
                onn.FOLDER = f'Analysis/MNIST_AddedPhaseNoise/N={N}_{onn.phase_uncert_theta[0]}'
                onn.createFOLDER()

                onn.same_phase_uncert = False
                print('Different Phase Uncert')
                onn.accuracy_PT = calc_acc.get_accuracy(onn, model, onn.Xt, onn.yt, loss_diff=onn.loss_diff)

                onn.same_phase_uncert = True
                print('Same Phase Uncert')
                onn.accuracy_LPU = calc_acc.get_accuracy(onn, model, onn.Xt, onn.yt, loss_diff=onn.loss_diff)

                acc_colormap.colormap_me(onn)

                onn.saveAll(model)
                np.savetxt(f'{onn.FOLDER}/all_topologies.txt', onn_topo, fmt='%s')
                break
