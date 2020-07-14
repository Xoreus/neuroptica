''' changes the loss and phase uncert limits for testing different topology sizes
using Neuroptica and linearly separable datasets

Author: Simon Geoffroy-Gagnon
Edit: 2020.07.11
'''
import numpy as np
import calculate_accuracy as calc_acc
import ONN_Simulation_Class as ONN_Cls
import test_trained_onns as test
import ONN_Setups 
import acc_colormap

if __name__ == '__main__':
    ONN = ONN_Cls.ONN_Simulation()
    topos = ['E_P']
    for ONN.N in [8]:
        for ONN.topo in topos:
            print(f'N={ONN.N}, topo={ONN.topo}')
            ONN.FOLDER = f'/home/simon/Documents/neuroptica/tests/Analysis/PerfectTraining/N=8/N=8_minLoss=0dB_lossStdDev=0dB'
            ONN = ONN.pickle_load()
            ONN.range_dB = 5
            model = ONN_Setups.ONN_creation(ONN)
            ONN.model = model

            ONN.FOLDER = f'/home/simon/Documents/neuroptica/tests/Analysis/N=8'
            ONN.FOLDER = f'/home/simon/Documents/neuroptica/tests/Analysis/PerfectTraining/N=8/N=8_minLoss=0dB_lossStdDev=0dB'
            ONN.createFOLDER()
            ONN.ITERATIONS = 20 # number of times to retry same loss/PhaseUncert
            ONN.loss_diff = 0.0 # \sigma dB

            ONN.loss_dB = np.linspace(0, 2, 41)

            ONN.phase_uncert_theta = np.linspace(0., 0.5, 41)
            ONN.phase_uncert_phi = np.linspace(0., 0.5, 41)


            model.set_all_phases_uncerts_losses(ONN.phases, 0, 0, 0, 0)
            ONN, model = test.test_onn(ONN, model)

            ONN.saveAll(model)
            
            acc_colormap.colormap_me(ONN)
            ONN.saveSimDataset()
            ONN.saveAccuracyData()
            ONN.pickle_save()

