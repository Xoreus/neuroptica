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
        for ii in range(1):
            for ONN.topo in topos:
                ONN.get_topology_name()
                print(f'N={ONN.N}, topo={ONN.topo}, ii={ii}')
                ONN.FOLDER = '/storage/Research/OE_2020/N=4/N=4_new'
                ONN.FOLDER = f'/storage/Research/OE_2020/N={ONN.N}/N={ONN.N}_LPU'
                ONN.FOLDER = f'/home/simon/Documents/neuroptica/tests/Analysis/N=16/N=16_2'
                ONN.FOLDER = f'/home/simon/Documents/neuroptica/tests/Analysis/N={ONN.N}/N={ONN.N}_{ii}'
                ONN = ONN.pickle_load()
                model = ONN_Setups.ONN_creation(ONN)

                # ONN.FOLDER = f'/home/simon/Documents/neuroptica/tests/Analysis/N=16/N=16_2-new'
                ONN.createFOLDER()
                ONN.ITERATIONS = 50 # number of times to retry same loss/PhaseUncert
                ONN.loss_diff = 0 # \sigma dB

                ONN.loss_dB = np.linspace(0, 1, 41)
                ONN.phase_uncert_theta = np.linspace(0., 0.5, 41)
                ONN.phase_uncert_phi = np.linspace(0., 0.5, 41)

                model.set_all_phases_uncerts_losses(ONN.phases, 0, 0, 0, 0)
                ONN, model = test.test_onn(ONN, model)

                ONN.saveAll(model)
                
                acc_colormap.colormap_me(ONN)
                ONN.saveSimDataset()
                ONN.saveAccuracyData()
                ONN.pickle_save()
