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
    topos = ['R_P', 'C_Q_P', 'E_P', 'R_I_P']
    for ONN.N in [8]:
        for ONN.topo in topos:
            print(f'N={ONN.N}, topo={ONN.topo}')
            ONN.FOLDER = f'/home/simon/Documents/neuroptica/tests/Analysis/N=96_1'
            ONN = ONN.pickle_load()
            model = ONN_Setups.ONN_creation(ONN)

            ONN.FOLDER = f'/home/simon/Documents/neuroptica/tests/Analysis/N=96_3'
            ONN.createFOLDER()
            ONN.ITERATIONS = 10 # number of times to retry same loss/PhaseUncert
            ONN.loss_diff = 0.0 # \sigma dB

            ONN.loss_dB = np.linspace(0, 0.4, 41)

            ONN.phase_uncert_theta = np.linspace(0., 0.2, 41)
            ONN.phase_uncert_phi = np.linspace(0., 0.2, 41)


            model.set_all_phases_uncerts_losses(ONN.phases, 0, 0, 0, 0)
            ONN, model = test.test_onn(ONN, model)

            ONN.saveAll(model)
            
            acc_colormap.colormap_me(ONN)
            ONN.saveSimDataset()
            ONN.saveAccuracyData()
            ONN.pickle_save()

