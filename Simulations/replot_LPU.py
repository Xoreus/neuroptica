from math import floor
import sys
# from neuroptica.components import MZI
sys.path.append('../')
import pickle
import numpy as np
import calculate_accuracy as calc_acc
import ONN_Simulation_Class as ONN_Cls
import test_trained_onns as test
import ONN_Setups
import neuroptica as neu
from main import save_onn

'''
A python script that takes a trained ONN instance (*.pkl file) and replot the PT and LPU graphs
using stored accuracy or re-simulated accuracy.

Make sure the model created below matches the trained model stored in .pkl file.
(i.e same no. of layers, same topology)

if want to use new accuracies to plot colormaps instead of the ones stored in .csv,
uncomment "save_onn(onn, model)" to re-test with testing set,
this can show effect of random phase errors on the FoM size (not much).

Author: Bokun Zhao
Edit: 2024.05.17 by Bokun Zhao (bokun.zhao@mail.mcgill.ca)
'''


if __name__ == '__main__':
    topology = "custom" # or "clement"
    for seed in [13, 21, 47, 50, 91]:
        pkl_path = f'Analysis/iris_augment/10x2_{topology}_VOA_rng_{seed}_auto/{topology}_VOA_rng_{seed}_auto.pkl'
        with open(pkl_path, 'rb') as p:
            onn = pickle.load(p)
            print(onn.FOLDER)
            print(onn.topo)
        # reset fixed seed so that different phase uncertainties are sampled at each execution
        # np.random.seed()
        
        # onn.createFOLDER()
        # onn.plotBackprop(backprop_legend_location=0)
        save_onn(onn, onn.model) # if want to plot with new accuracies, uncomment this
        # onn.saveAll(onn.model, cmap='hsv')