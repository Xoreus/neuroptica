"""
This allows for the quick creation of different ONN layers, such as Reck (R) or inverse Reck (I) layers. Also creates any non-linearity layer. 

Author: Simon Geoffroy-Gagnon
Edit: 05.02.2020
"""
import sys
sys.path.append('/home/simon/Documents/neuroptica')
import neuroptica as neu
import numpy as np

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

def ONN_creation(layers='R', N=4, loss_dB=0, loss_diff=0, phase_uncert=0, Nonlinearity=neu.Sigmoid(4), phases=[(None, None)]):
    """ 
    Create the Topology based on the layers and N provided. R = Reck, I = Inverted Reck, A = add mask(2N), M = DMM layer, D = Drop Mask, N = Nonlinearity, P = Photodetector, B = sqrt(Photodetector), C = diamond layer, Q = Drop mask, keep bottom ports of diamond, W = drop mask, keep ports in middle of diamond
    """
    layers = layers.replace('_', '') 
    layers = ''.join(char if char != 'D' else 'AMD' for char in layers) # D really means AddMask, DMM, DropMask
    layer_dict = {
            'R':neu.ReckLayer(N, include_phase_shifter_layer=False, loss_dB=loss_dB, loss_diff=loss_diff, phase_uncert=phase_uncert, phases=phases),
            'I':neu.flipped_ReckLayer(N, include_phase_shifter_layer=False, loss_dB=loss_dB, loss_diff=loss_diff, phase_uncert=phase_uncert, phases=phases), 

            'A':neu.AddMask(2*N), 
            'M':neu.DMM_layer(2*N, loss_dB=loss_dB, loss_diff=loss_diff, phase_uncert=phase_uncert, phases=phases),
            'D':neu.DropMask(N=2*N, keep_ports=range(0, 2*N, 2)), 

            'N':neu.Activation(Nonlinearity), 
            'P':neu.Activation(neu.AbsSquared(N)),
            'B':neu.Activation(neu.Abs(N)),

            'C':neu.DiamondLayer(N, include_phase_shifter_layer=False, loss_dB=loss_dB, loss_diff=loss_diff, phase_uncert=phase_uncert, phases=phases), # Diamond Mesh
            'Q':neu.DropMask(2*N - 2, keep_ports=range(N - 2, 2*N - 2)), # Bottom Diamond Topology
            'W':neu.DropMask(2*N - 2, drop_ports=[0, 5]), # Central Diamond Topology

            'E':neu.ClementsLayer(N, include_phase_shifter_layer=False, loss_dB=loss_dB, loss_diff=loss_diff, phase_uncert=phase_uncert, phases=phases),
            }

    Model = neu.Sequential([layer_dict[layer] for layer in layers])
    return Model

if __name__ == '__main__':
    Model = ONN_creation('E_P', N=4)
    print(len(Model.get_all_phases()[0]))

    tf = Model.get_transformation_matrix()
    print(tf)
