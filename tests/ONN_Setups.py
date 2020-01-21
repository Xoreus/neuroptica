"""
This allows for the quick creation of different ONN layers, such as Reck (R) or inverse Reck (I) layers. Also creates any non-linearity layer. 
"""

import sys
sys.path.append('/home/simon/Documents/neuroptica')
import neuroptica as neu
import numpy as np


def ONN_creation(layers='R', N=4, loss=0, phase_uncert=0, Nonlinearity=neu.Sigmoid(4), Phases=[(None, None)]):

    layers = layers.replace('_', '') 
    layers = ''.join(char if char != 'D' else 'AMD' for char in layers) # D really means AddMask, DMM, DropMask
    # print(layers)
    layer_dict = {
            'R':neu.ReckLayer(N, include_phase_shifter_layer=False, loss=loss, phase_uncert=phase_uncert),
            'I':neu.flipped_ReckLayer(N, include_phase_shifter_layer=False, loss=loss,phase_uncert=phase_uncert), 

            'A':neu.AddMask(2*N), 
            'M':neu.DMM_layer(2*N, loss=loss, phase_uncert=phase_uncert),
            'D':neu.DropMask(N=2*N, keep_ports=range(0, 2*N, 2)), 

            'N':neu.Activation(Nonlinearity), 
            'P':neu.Activation(neu.AbsSquared(N)),
            'B':neu.Activation(neu.Abs(N)),

            'C':neu.DiamondLayer(N, loss=loss, phase_uncert=phase_uncert, include_phase_shifter_layer=False),
            'Q':neu.DropMask(2*N - 2, keep_ports=range(N - 2, 2*N - 2)),
            'W':neu.DropMask(2*N - 2, keep_ports=range(N - 1, 2*N - 1))

            }

    Model = neu.Sequential([layer_dict[layer] for layer in layers])
    return Model

if __name__ == '__main__':
    Model = ONN_creation('C_Q_I_P')

    # phases = []
    # pi = np.pi
    # phases.append((100*pi/180, 5*pi/180))   
    # phases.append((90*pi/180, 10*pi/180))  
    # phases.append((80*pi/180, 15*pi/180))   
    # phases.append((70*pi/180, 20*pi/180))
    # phases.append((60*pi/180, 25*pi/180))
    # phases.append((50*pi/180, 30*pi/180)) 
    # phases = [phases]

    # Model.set_all_phases_uncerts_losses(phases, 0, 0)

    # Phases = Model.get_all_phases()

    # trf_matrix = Model.get_transformation_matrix()
    # np.set_printoptions(precision=4)
    # np.set_printoptions(suppress=True)
    # print("Transformation Matrix:")
    # print(trf_matrix)
    # np.set_printoptions(precision=3)
    # np.set_printoptions(suppress=True)
    # print("Phases:")
    # print(Phases)

    trf_matrix = np.array(Model.get_transformation_matrix())
    flat_trf_matrix = []
    # print(trf_matrix)
    # print([len(trf) for trf in trf_matrix])
    for elem in trf_matrix:
        for row in elem:
            flat_trf_matrix.append(row)
    # print(flat_trf_matrix)
    # np.savetxt('TransformationMatrix.txt', np.array(flat_trf_matrix), fmt = '%.4f+%.4fj')

    with open('TransformationMatrix.txt', "a") as myfile:
        for trf in trf_matrix:
            np.savetxt(myfile, trf, fmt='%.4f%+.4fj, '*len(trf[0]), delimiter=', ')
            myfile.write('\n')

