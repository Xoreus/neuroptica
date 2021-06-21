''' changes the loss and phase uncert limits for testing different topology sizes
using Neuroptica and linearly separable datasets
Limits (loss_dB and phase_uncert_*) decrease as the mesh size increases! No good 
way to see by how much yet though.

Author: Simon Geoffroy-Gagnon
Edit: 2020.07.11
'''
import numpy as np
import calculate_accuracy as calc_acc
import ONN_Simulation_Class as ONN_Cls
import test_trained_onns as test
import ONN_Setups 

if __name__ == '__main__':
    onn = ONN_Cls.ONN_Simulation()
    onn_topo = ['B_C_Q_P', 'E_P', 'R_P']
    # onn_topo = ['C_Q_P', 'E_P', 'R_P']
    for onn.topo in onn_topo:
        onn.FOLDER = f'/home/edwar/Documents/Github_Projects/neuroptica/tests/Analysis/N=8_lossy'
        # onn.FOLDER = f'/home/edwar/Documents/Github_Projects/neuroptica/tests/Analysis/outPorts_mean_pow/N=32_4'
        # onn.FOLDER = f'/home/edwar/Documents/Github_Projects/neuroptica/tests/Analysis/N=96_1'
        onn = onn.pickle_load()
        print(f'N={onn.N}, topo={onn.topo}')
        onn.range_dB = 5
        model = ONN_Setups.ONN_creation(onn)
        onn.model = model

        onn.FOLDER = f'Analysis/N={onn.N}_lossy'
        onn.createFOLDER()
        onn.ITERATIONS = 55 # number of times to retry same loss/PhaseUncert
        onn.loss_diff = 0.0 # \sigma dB

        onn.loss_dB = np.linspace(0.5, 0.8, 2)
        onn.phase_uncert_theta = np.linspace(0., 1, 41)
        onn.phase_uncert_phi = np.linspace(0., 1, 41)

        model.set_all_phases_uncerts_losses(onn.phases, 0, 0, 0, 0)
        onn, model = test.test_onn(onn, model)

        onn.saveAll(model)
        onn.saveSimDataset()
        onn.saveAccuracyData()
        onn.pickle_save()

