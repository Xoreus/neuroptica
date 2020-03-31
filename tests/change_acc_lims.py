''' changes the loss and phase uncert limits for testing different topology sizes
using Neuroptica and linearly separable datasets

Author: Simon Geoffroy-Gagnon
Edit: 20.02.2020
'''
import numpy as np
import calculate_accuracy as calc_acc
import ONN_Simulation_Class as ONN_Cls
import test_trained_onns as test
import ONN_Setups 
import acc_colormap

if __name__ == '__main__':
    ONN = ONN_Cls.ONN_Simulation()


    ONN.zeta = 0
    topos = ['R_I_P','R_P', 'C_Q_P', 'E_P']
    topos = ['R_I_P']

    for ONN.N in [8]:
        for ii in range(32):
            for ONN.topo in topos:
                ONN.get_topology_name()
                print(f'N={ONN.N}, topo={ONN.topo}, ii={ii}')
                ONN.FOLDER = '/home/simon/Documents/neuroptica/tests/Analysis/Lossy_Training/N=8_0_old'
                ONN = ONN.pickle_load()
                model = ONN_Setups.ONN_creation(ONN)

                ONN.ITERATIONS = 50 # number of times to retry same loss/PhaseUncert
                ONN.loss_diff = 0 # \sigma dB
                ONN.loss_dB = np.linspace(0, 1.5, 41)
                ONN.phase_uncert_theta = np.linspace(0., 0.4, 41)
                ONN.phase_uncert_phi = np.linspace(0., 0.4, 41)

                model.set_all_phases_uncerts_losses(ONN.phases, 0, 0, 0, 0)
                ONN, model = test.test_onn(ONN, model)

                ONN.createFOLDER()
                ONN.saveSelf()
                acc_colormap.colormap_me(ONN)
                ONN.saveSimDataset()
                ONN.saveAccuracyData()
                ONN.pickle_save()
