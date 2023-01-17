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
Edit: 2022.10.13 by Bokun Zhao (bokun.zhao@mail.mcgill.ca)
'''

pkl_path = 'Analysis/iris_augment/Gaussian_Bokun.pkl'

if __name__ == '__main__':
    with open(pkl_path, 'rb') as p:
        onn = pickle.load(p)
        onn.zeta = 0.75
    
    # reset fixed seed so that different phase uncertainties are sampled at each execution
    # np.random.seed()

    features = onn.features
    nlaf = neu.cReLU(onn.features)
    classes = features

    # reck model (single/double layer)
    # model = neu.Sequential([
    #     # neu.ReckLayer(features),
    #     # neu.Activation(nlaf), # first layer ends here
    #     neu.ReckLayer(features),
    #     neu.Activation(neu.AbsSquared(features)), # photodetector measurement
    #     neu.DropMask(features, keep_ports=range(classes)) # Drops the unwanted ports
    # ])

    # clement model (single/double layer)
    # model = neu.Sequential([
    #     # neu.ClementsLayer(features),
    #     # neu.Activation(nlaf), # first layer ends here
    #     neu.ClementsLayer(features),
    #     neu.Activation(neu.AbsSquared(features)), # photodetector measurement
    #     neu.DropMask(features, keep_ports=range(classes))
    # ])

    # bottom diamond model (single/double layer)
    # model = neu.Sequential([
        # neu.AddMaskDiamond(features),
        # neu.DiamondLayer(features),
        # neu.DropMask(2*features - 2, keep_ports=range(features - 2, 2*features - 2)), # Bottom Diamond Topology
        # neu.Activation(nlaf), # first layer ends here
    #     neu.AddMaskDiamond(features),
    #     neu.DiamondLayer(features),
    #     neu.DropMask(2*features - 2, keep_ports=range(features - 2, 2*features - 2)), # Bottom Diamond Topology
    #     neu.Activation(neu.AbsSquared(features)), # photodetector measurement
    #     neu.DropMask(features, keep_ports=range(classes)),
    # ])

    # center diamond model (single/double layer)
    model = neu.Sequential([
        # neu.AddMaskDiamond(features),
        # neu.DiamondLayer(features),
        # neu.DropMask(2*features - 2, keep_ports=range(features//2-1, floor(features*1.5)-1)), # Middle Diamond Topology
        # neu.Activation(nlaf), # first layer ends here
        neu.AddMaskDiamond(features),
        neu.DiamondLayer(features),
        neu.DropMask(2*features - 2, keep_ports=range(features//2-1, floor(features*1.5)-1)), # Middle Diamond Topology
        neu.Activation(neu.AbsSquared(features)), # photodetector measurement
        neu.DropMask(features, keep_ports=range(classes)),
    ])

    onn.model = model
    onn.createFOLDER()
    onn.plotBackprop(backprop_legend_location=0)
    # save_onn(onn, model) # if want to plot with new accuracies, uncomment this
    onn.saveAll(model, cmap='hsv')

    onn.saveForwardPropagation(model)
    # current_phases = model.get_all_phases()
    # model.set_all_phases_uncerts_losses(current_phases, phase_uncert_theta=0.181, phase_uncert_phi=0.181, loss_dB=0.00, loss_diff=0.0)
    # onn.save_correct_classified_samples(model)
    # model.set_all_phases_uncerts_losses(current_phases, phase_uncert_theta=0.0, phase_uncert_phi=0.0, loss_dB=0.00, loss_diff=0.0)
    # onn.save_correct_classified_samples(model)
    # onn.save_correct_classified_samples(model, zeta=onn.zeta)
    # onn.save_correct_classified_samples(model, zeta=2*onn.zeta)