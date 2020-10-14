''' replots required, trying to get good contour plotting

Author: Simon Geoffroy-Gagnon
Edit: 2020.09.24
'''
import numpy as np
import calculate_accuracy as calc_acc
import ONN_Simulation_Class as ONN_Cls
import test_trained_onns as test
import ONN_Setups 
import acc_colormap

if __name__ == '__main__':
    onn = ONN_Cls.ONN_Simulation()
    onn_topo = ['B_C_Q_P', 'E_P', 'R_P']
    FOLDERS = ['/home/simon/Documents/neuroptica/tests/Analysis/N=32',
            '/home/simon/Documents/neuroptica/tests/Analysis/N=96_v2',
            '/home/simon/Documents/neuroptica/tests/Analysis/N=8_v2',
            '/home/simon/Documents/neuroptica/tests/Analysis/N=8_lossy']
    # FOLDERS = ['/home/simon/Documents/neuroptica/tests/Analysis/N=32']
    for onn.FOLDER in FOLDERS:
        for onn.topo in onn_topo:
            onn = onn.pickle_load()
            onn.zeta = 0.75
            print(f'N={onn.N}, topo={onn.topo}, zeta = {onn.zeta}')
            model = ONN_Setups.ONN_creation(onn)
            onn.model = model

            onn.createFOLDER()
            onn.saveAll(model, cmap='gist_heat')
            
            acc_colormap.colormap_me(onn)
            onn.saveSimDataset()
            onn.saveAccuracyData()
            onn.pickle_save()
