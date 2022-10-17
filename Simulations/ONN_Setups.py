"""
This allows for the quick creation of different ONN layers, such as Reck (R) or inverse Reck (I) layers. Also creates any non-linearity layer.

Author: Simon Geoffroy-Gagnon
Edit: 05.02.2020
"""
import os
cwd = os.getcwd()
import sys
sys.path.append('python_packages/')
sys.path.append('../')
import neuroptica as neu
import numpy as np

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
def ONN_creation(onn, Nonlinearity=neu.nonlinearities.Sigmoid(4), phases=[(None, None)]):
    """  Create the Topology based on the layers and N provided. R = Reck, I = Inverted Reck, A = add mask(2N), M = DMM layer, D = Drop Mask, N = Nonlinearity, P = Photodetector, B = sqrt(Photodetector), C = diamond layer, Q = Drop mask, keep bottom ports of diamond, ect... Dict keys do not follow any good distribution unfortunately """
    layers = onn.topo.replace('_', '')
    layers = ''.join(char if char != 'D' else 'AMD' for char in layers) # D really means AddMask, DMM, DropMask
    layer_dict = {
            'R':neu.ReckLayer(onn.N, include_phase_shifter_layer=False, loss_dB=onn.loss_dB[0], loss_diff=onn.loss_diff, phase_uncert=onn.phase_uncert_theta[0], phases=phases),
            # 'I':neu.flipped_ReckLayer(onn.N, include_phase_shifter_layer=False, loss_dB=onn.loss_dB[0], loss_diff=onn.loss_diff, phase_uncert=onn.phase_uncert_theta[0], phases=phases),# Flipped Reck layer with N*(N-1)/2 MZIs in an upside down triangle

            'A':neu.AddMask(2*onn.N),# Add 0s between every channel to account for empty tapered input waveguides the DMM sections
            'M':neu.DMM_layer(2*onn.N, loss_dB=onn.loss_dB[0], loss_diff=onn.loss_diff, phase_uncert=onn.phase_uncert_theta[0], phases=phases),
            'D':neu.DropMask(N=2*onn.N, keep_ports=range(0, 2*onn.N, 2)),# Drop every other channel from the output of the DMM section

            'N':neu.Activation(Nonlinearity), # Nonlinearity can be neu.cReLU(), ect..
            'P':neu.Activation(neu.AbsSquared(onn.N)), # Photodetector
            
            'B':neu.AddMaskDiamond(onn.N), # Adds 0s to the top half of the Diamond input
            'C':neu.DiamondLayer(onn.N, include_phase_shifter_layer=False, loss_dB=onn.loss_dB[0], loss_diff=onn.loss_diff, phase_uncert=onn.phase_uncert_theta[0], phases=phases), # Diamond Mesh
            'Q':neu.DropMask(2*onn.N - 2, keep_ports=range(onn.N - 2, 2*onn.N - 2)), # Bottom Diamond Topology
            # 'W':neu.DropMask(2*onn.N - 2, drop_ports=[0, onn.N+1]), # Central Diamond Topology

            'E':neu.ClementsLayer(onn.N, include_phase_shifter_layer=False, loss_dB=onn.loss_dB[0], loss_diff=onn.loss_diff, phase_uncert=onn.phase_uncert_theta[0], phases=phases), # Clements layer, N*(N-1)/2 MZIs setup in a rectangular topology
            }

    Model = neu.Sequential([layer_dict[layer] for layer in layers])
    return Model

if __name__ == '__main__':
    Model = ONN_creation('E_P', N=4)
    print(len(Model.get_all_phases()[0]))

    tf = Model.get_transformation_matrix()
    print(tf)
