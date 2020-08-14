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
    onn = ONN_Cls.ONN_Simulation()
    onn_topo = ['B_C_Q_P', 'E_P', 'R_I_P', 'R_P']
    onn_topo = ['E_P', 'R_P']
    for onn.N in [8]:
        for onn.topo in onn_topo:
            print(f'N={onn.N}, topo={onn.topo}')
            onn.FOLDER = f'Analysis/IPC/N=8'
            onn = onn.pickle_load()
            onn.range_dB = 5
            model = ONN_Setups.ONN_creation(onn)
            onn.model = model

            onn.FOLDER = f'Analysis/N={onn.N}'
            onn.createFOLDER()
            onn.ITERATIONS = 50 # number of times to retry same loss/PhaseUncert
            onn.loss_diff = 0.0 # \sigma dB

            onn.loss_dB = np.linspace(0, 2.5, 41)

            onn.phase_uncert_theta = np.linspace(0., 0.5, 41)
            onn.phase_uncert_phi = np.linspace(0., 0.5, 41)


            model.set_all_phases_uncerts_losses(onn.phases, 0, 0, 0, 0)
            onn, model = test.test_onn(onn, model)

            onn.saveAll(model)
            
            acc_colormap.colormap_me(onn)
            onn.saveSimDataset()
            onn.saveAccuracyData()
            onn.pickle_save()

