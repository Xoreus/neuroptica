import os
import sys
from typing import List
import wave

# from neuroptica.components import MZI
sys.path.append('../')
# import neuroptica.nonlinearities as neuNon
import neuroptica as neu
from sklearn.decomposition import PCA
import random
import gzip
from urllib.request import urlretrieve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import pickle
import test_trained_onns as test
from neuroptica.component_layers import MZILayer, OpticalMesh, PhaseShifterLayer

def load_onn_pkl():
    with open(f'Analysis/iris_augment/8x8_test/DIAMOND With 2pi inital phases, varying uncert..pkl', 'rb') as p:
        loaded_onn = pickle.load(p)
        loaded_onn.topo = loaded_onn.topo + "replot"
        return loaded_onn 

def save_onn(onn, model, lossDiff=0):
    onn.loss_diff = lossDiff # Set loss_diff
    # For simulation purposes, defines range of loss and phase uncert
    onn.loss_dB = np.linspace(0., 1, 40) # set loss/MZI range
    onn.phase_uncert_theta = np.linspace(0., 1, 40) # set theta phase uncert range
    onn.phase_uncert_phi = np.linspace(0., 1, 40) # set phi phase uncert range

    onn, model = test.test_PT(onn, onn.Xt, onn.yt, model, show_progress=True) # test Phi Theta phase uncertainty accurracy
    onn, model = test.test_LPU(onn, onn.Xt, onn.yt, model, show_progress=True) # test Loss/MZI + Phase uncert accuracy
    onn.saveAll(model) # Save best model information
    onn.plotAll() # plot training and tests
    onn.plotBackprop(backprop_legend_location=0)
    ''' Backprop Legend Location Codes:
    'best' 	        0
    'upper right' 	1
    'upper left' 	2
    'lower left' 	3
    'lower right' 	4
    'right'        	5
    'center left' 	6
    'center right' 	7
    'lower center' 	8
    'upper center' 	9
    'center'    	10
    '''
    onn.pickle_save() # save pickled version of the onn class

def mzi_uncertainties(p_waveguide_indices, p_ref_value_theta, p_ref_value_phi, p_i):
    '''
    Called in component_layers.py, MZILayer() class, in method from_waveguide_indices()
    Method to apply different uncertainties based on how isolated a MZI is, for diamond/reck mesh.
    e.g. for diamond mesh, the assumption is made such that MZIs at top or bottom can be well calibrated, thus having low phase uncertainties,
    uncertainties increases as we go towards the middle.
    :param p_waveguide_indices: list of port indices in this layer, len(waveguide_indices)/2 gives no. of MZIs in this layer.
    :param p_ref_value_theta/phi (unit: rad): reference sigma value, uncertainties can be scaled using this value as a basis, e.g. 0.5*ref_value
    :param p_i: = 0, 2, 4, ..., caller function's for loop index, indicate which MZI within a MZILayer we're creating.
    :return: tuple (sigma_theta, sigma_phi) representing the phase uncertainties (unit: rad) for a single MZI
    '''
    network_dim = p_waveguide_indices[int(len(p_waveguide_indices)/2)] + 1 # ONN size / no. of input ports in use / transformation matrix dimesion, e.g. 8.
    mid = network_dim - 1.5 # "theoretical" average of wavegude_indices array
    # e.g. for  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], mid = 6.5

    assert mid == 6.5, f"mid is {mid}, expected 6.5" # debug purpose, when running 8x8 diamond
    assert network_dim == 8, f"network_dim is {network_dim}, expected 8" # debug purpose

    # a metric to indicate how much this mzi is away from center row
    # e.g. for 8x8 diamond, center row is 6.5, which has highest uncertainty
    # e.g. for 8x8 diamond, top row is 12.5, which has lowest uncertainty
    # e.g. for 8x8 diamond, bottom row is 0.5, which has lowest uncertainty
    dist_to_center = abs(p_waveguide_indices[p_i] + 0.5 - mid) # range 0 - 6 for 8x8 diamond
    theta_uncert = mid - dist_to_center #takes values of (0.5, 1.5, 2.5, ..., 6.5) for 8x8 diamond, the larger the dist, the smaller the uncert

    theta_uncert  = (theta_uncert - 0.5) / 2 + 0.5 # takes scaling values of (0.5, 1, 1.5, 2, 2.5, 3, 3.5) for 8x8 diamond
    phi_uncert  = theta_uncert + 0.00 # takes scaling values of (0.5 + 0.00, 1 + 0.00,..., 3.5 + 0.00) for 8x8 diamond
    # based on heuristic, the phi shifter is believed to have more error than theta shifter within a MZI
    # so a positive offset can be added to the scaling for phi_uncert if wished to do so, but for 2D PT_Plot, this is unnecessary.

    # apply the above scaling to p_ref_value
    max_scale = (mid - 0.5) / 2 + 0.5 + 0.00 # = 3.75 for 8x8 diamond
    assert max_scale == 3.5, f"max_scale is {max_scale}, expected 3.75" # debug purpose
    sigma_theta = theta_uncert/max_scale * p_ref_value_theta
    sigma_phi = phi_uncert/max_scale * p_ref_value_phi

    # return (sigma_theta, sigma_phi) # unit: rad
    return (theta_uncert/max_scale, phi_uncert/max_scale) # unit: rad

def mzi_uncertainties_revised_reck(p_layerCount, p_N, p_waveguide_indices, p_ref_value_theta, p_ref_value_phi, p_i):
    '''
    ===revised so that the uncertainties are now based on how direct we can access a MZI's input===
    Called in component_layers.py, MZILayer() class, in method from_waveguide_indices()
    :param p_layerCount: layer index, indicating which MZIlayer is being created
    :param p_N: ONN size / no. of input ports in use / transformation matrix dimesion, e.g. 8
    :param p_waveguide_indices: list of port indices in this layer, len(waveguide_indices)/2 gives no. of MZIs in this layer.
    :param p_ref_value_theta/phi (unit: rad): reference sigma value, uncertainties can be scaled using this value as a basis, e.g. 0.5*ref_value
    :param p_i: = 0, 2, 4, ..., caller function's for loop index, indicate which MZI within a MZILayer we're creating.
    :return: tuple (sigma_theta, sigma_phi) representing the phase uncertainties (unit: rad) for a single MZI
    '''
    MZIcount = len(p_waveguide_indices)//2 # no. of MZIs in this MZILayer
    scale = 0.2 # ref_value x scale = actual uncertainty in radian
    maxScale = (0.1 + 0.1*(p_layerCount//2)) * 2 # maximum scaling in this layer
    theta_uncert = p_ref_value_theta * (maxScale - (p_i//2) * scale) # MZIs at lower indices have high uncertainty
    phi_uncert = p_ref_value_phi * (maxScale - (p_i//2) * scale)
    return (theta_uncert, phi_uncert) # unit: rad

def mzi_uncertainties_revised_clement(p_layerCount, p_ref_value_theta, p_ref_value_phi):
    scale = ((p_layerCount+1) // 2 / 10 + 0.1) * 2
    theta_uncert = p_ref_value_theta * scale
    phi_uncert = p_ref_value_phi * scale
    return (theta_uncert, phi_uncert) # unit: rad

def mzi_uncertainties_revised_diamond(p_layerCount, p_N, p_waveguide_indices, p_ref_value_theta, p_ref_value_phi, p_i):
    MZIcount = len(p_waveguide_indices)//2 # no. of MZIs in this MZILayer
    centerLayer = (p_N + 2)/2 - 2 # = 6
    steps = (MZIcount - 1) // 2 # no. of increments/decrement of uncertainty scale
    maxScale = (0.1 + 0.1*(p_layerCount//2)) * 2 # maximum scaling in this layer
    scale = float()
    if p_layerCount <= 6: # first half + middle MZILayers, starts at 0.1 rad uncertainties (top MZI within each MZILayer)
        scale = 0.2
    else:
        scale = 0.2 + 0.2 * (p_layerCount - centerLayer) # second half of the MZILayers
    
    if p_i//2 < MZIcount/2.0: # top half of MZIs (+ the middle one), increasing with p_i
        scale += p_i / 10.0
    else: # bottom half, decreasing from maxScale, with p_i
        if MZIcount % 2 == 0: # even layers
            scale = maxScale - 0.2*(p_i//2 - MZIcount//2)
        else: # odd layers
            scale = maxScale - 0.2*(p_i//2 - MZIcount//2)
    theta_uncert = p_ref_value_theta * scale # MZIs at lower indices have high uncertainty
    theta_uncert = p_ref_value_theta * scale
    phi_uncert = p_ref_value_phi * scale
    return (theta_uncert, phi_uncert) # unit: rad

##########below are methods to help understand the code, run them to see output


def replot_trained_model():
    onn = load_onn_pkl()
    onn.loss_dB = np.linspace(0., 1, 40) # set loss/MZI range
    onn.phase_uncert_theta = np.linspace(0., 1, 40) # set theta phase uncert range
    onn.phase_uncert_phi = np.linspace(0., 1, 40) # set phi phase uncert range

    model = neu.Sequential([
        neu.AddMaskDiamond(8),
        neu.DiamondLayer(8),
        neu.DropMask(2*8 - 2, keep_ports=range(8 - 2, 2*8 - 2)), # Bottom Diamond Topology
        neu.Activation(neu.AbsSquared(8)), # photodetector measurement
        neu.DropMask(8, keep_ports=range(8)),
    ])

    save_onn(onn, model)
    onn.saveForwardPropagation(model)
    current_phases = model.get_all_phases()
    model.set_all_phases_uncerts_losses(current_phases)
    onn.save_correct_classified_samples(model)
    onn.save_correct_classified_samples(model, zeta=onn.zeta)
    onn.save_correct_classified_samples(model, zeta=2*onn.zeta)
    
def see_component_layer():
    # test_waveguides = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    # for i in range(0, len(test_waveguides), 2):
    #     result = mzi_uncertainties(test_waveguides, p_ref_value_theta=0.5, p_ref_value_phi=0.76, p_i=i)
    #     print(f"({result[0]:.2f}, {result[1]:.2f})")

    N = 8 # waveguides to be used in a mesh input 
    S = 2*N - 2 # for diamond: total physical ports (include those not used)

    mzi_limits_lower = list(range(S - N, -1, -1)) + list(range(1, N - 1))
    mzi_limits_upper = list(range(N - 1, S)) + list(range(S - 2, N - 2, -1))
    print(f"diamond_mzi_limits_lower:\n{mzi_limits_lower}\n")
    print(f"diamond_mzi_limits_upper:\n{mzi_limits_upper}\n")

    # mzi_limits_upper = [i for i in range(1, N)] + [i for i in range(N - 2, 1 - 1, -1)]
    # mzi_limits_lower = [(i + 1) % 2 for i in mzi_limits_upper]
    # print(f"reck_mzi_limits_lower:\n{mzi_limits_lower}\n")
    # print(f"reck_mzi_limits_upper:\n{mzi_limits_upper}\n")

    if N % 2 == 0:
        mzi_limits_lower = [(i) % 2 for i in range(N)]
        mzi_limits_upper = [N - 1 - (i) % 2 for i in range(N)]
    else:
        mzi_limits_lower = [(i) % 2 for i in range(N)]
        mzi_limits_upper = [N - 1 - (i + 1) % 2 for i in range(N)]
    print(f"clement_mzi_limits_lower:\n{mzi_limits_lower}\n")
    print(f"clement_mzi_limits_upper:\n{mzi_limits_upper}\n")
    
    mzi_nums = [int(len(range(start, end+1))/2) for start, end in zip(mzi_limits_lower, mzi_limits_upper)] # get the number of MZIs in this component layer
    print(f"mzi_nums: {mzi_nums}\n")

    phases = [(None, None) for _ in range(sum(mzi_nums))]
    print(f"phases:\n{phases}\n")

    phases_mzi_layer = []
    idx = 0
    for ii in mzi_nums:
        phases_layer = []
        for jj in range(ii):
            phases_layer.append(phases[idx])
            idx += 1
        phases_mzi_layer.append(phases_layer)

    print(f"phases_mzi_layer:\n{phases_mzi_layer}\n")

    print(f"{len(mzi_limits_lower)}, {len(mzi_limits_upper)}, {len(phases_mzi_layer)}")
    print(f"\n----------------------------------for loop----------------------------------\n")
    layers = []
    layerCount = 0 # keep track of which layer we are creating
    for start, end, phases in zip(mzi_limits_lower, mzi_limits_upper, phases_mzi_layer): # for each MZILayer...
        thetas = [phase[0] for phase in phases]
        phis = [phase[1] for phase in phases]
        print(f"thetas:\t {thetas}")
        print(f"phis:\t {phis}")
        waveguide_indices =list(range(start, end + 1))
        print(f"waveguide_indices:\t {waveguide_indices}\n")
        for i in range(0, len(waveguide_indices), 2): # ...create uncertainties for EACH MZI within that MZILayer
            # print(mzi_uncertainties_revised_diamond(p_layerCount=layerCount, p_N=S ,p_waveguide_indices=waveguide_indices, p_ref_value_theta=0.5, p_ref_value_phi=0.5, p_i=i), end=' ')
            print(mzi_uncertainties_revised_clement(p_layerCount=layerCount, p_ref_value_theta=0.5, p_ref_value_phi=0.5))
        layerCount += 1
        print('\n\n----------------------------------------------------------------------\n')
        # layers.append(MZILayer.from_waveguide_indices(S, list(range(start, end + 1)), thetas=thetas, phis=phis, phase_uncert_theta=phase_uncert_theta, phase_uncert_phi=phase_uncert_phi, loss_dB=loss_dB, loss_diff=loss_diff))

if __name__ == '__main__':
    # B = np.zeros_like(X, dtype=NP_COMPLEX)
    zeros_numpy = np.zeros((5, 5))
    one_numpy = np.ones((5, 5))
    empty_numpy = np.empty((5, 5))
    sequence_numpy = np.arange(0, 5, 0.5, dtype=int)
    random_numpy = np.random.randint(0, 10, (5, 5))

    print(f"zeros_numpy: {zeros_numpy}")
    print(f"one_numpy: {one_numpy}")
    print(f"empty_numpy: {empty_numpy}")
    print(f"sequence_numpy: {sequence_numpy}")
    print(f"random_numpy: {random_numpy}")

    dot_product = np.dot(sequence_numpy, random_numpy)
    