''' phase_uncert_thetar simulating Optical Neural Network

using Neuroptica and linearly separable datasets
Code that runs a Reck + DMM + Inverted Reck topology and saves the phases
Farhad, you need to download all the required Python packages before
using pip install [package] on the terminal.

Author: Simon Geoffroy-Gagnon
Edit: 2020.05.05
'''
import numpy as np
import calculate_accuracy as calc_acc
import ONN_Simulation_Class as ONN_Cls
import acc_colormap
import training_onn as train
import test_trained_onns as test
from collections import defaultdict

ONN = ONN_Cls.ONN_Simulation() # Creates a ONN simulation class
ONN.BATCH_SIZE = 2**6 # Batch size
ONN.EPOCHS = 200
ONN.STEP_SIZE = 0.005

ONN.ITERATIONS = 10 # number of times to retry same loss/PhaseUncert when simulating the ONN
rng = 1

onn_topo = ['R_D_I_P']
for ii in range(1):
    for ONN.N in [4]:
        ONN, rng = train.get_dataset(ONN, rng, SAMPLES=50) # Gets a linearly separable gaussian dataset
        ONN.FOLDER = f'Farhads_SVD_FOLDER' # Where to save the files 
        ONN.createFOLDER() # Create folder using above path
        ONN.saveSimDataset() # Saves the dataset

        for ONN.topo in onn_topo:
            ONN.loss_diff = 0 # sigma of the insertion loss
            ONN.get_topology_name() # go from R_D_I_P to Reck + DMM + Inverted Reck
            for ONN.rng in range(1):
                ONN.phases = [] # Reset phases
                ONN.loss_dB = [0] # We want no loss when training
                ONN, model = train.train_single_onn(ONN) # trains a single ONN Topology ONN.topo
                if max(ONN.val_accuracy) > 95: # only simulate ONN if max accuracy > 95%

                    # Simulation of ONN with loss/mzi and phase uncert
                    ONN.loss_dB = np.linspace(0, 2, 41)
                    ONN.phase_uncert_theta = np.linspace(0., 1., 41)
                    ONN.phase_uncert_phi = np.linspace(0., 1., 41)
                    ONN.loss_diff = 0
                    test.test_onn(ONN, model)
                    acc_colormap.colormap_me(ONN)
                    ONN.saveAll(model)
                    ONN.pickle_save()
                    break

