"""
This allows for the quick creation of different ONN layers, such as Reck (R) or inverse Reck (I) layers. Also creates any non-linearity layer. 
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
    from numpy import pi
    import pandas as pd
    from functools import reduce
    from neuro_to_farhad_phases import list_to_df, to_neuro_phases, to_farhad_phases

    theta1 = 100*pi/180;    
    theta2 = 90*pi/180;  
    theta3 = 80*pi/180;    theta4 = 70*pi/180;
    theta5 = 60*pi/180;    theta6 = 50*pi/180; 
    theta7 = 40*pi/180;    theta8 = 30*pi/180;     theta9 = 20*pi/180;   
    phi1 = 5*pi/180;    
    phi2 = 10*pi/180;    phi3 = 15*pi/180;    phi4 = 20*pi/180;
    phi5 = 25*pi/180;    phi6 = 30*pi/180;   
    phi7 = 35*pi/180;    phi8 = 40*pi/180;    phi9 = 45*pi/180; 

    farhad_phases = [(theta1, phi1), (theta2, phi2), (theta3, phi3), (theta4, phi4), (theta5, phi5), (theta6, phi6), (theta7, phi7), (theta8, phi8), (theta9, phi9)]
    df = pd.DataFrame(farhad_phases, columns=['Theta','Phi'])
    print("Farhad's original phases:")
    print(df)

    # phases = to_neuro_phases(farhad_phases, 'diamond')
    phases = to_neuro_phases(farhad_phases, 'reck')

    # Model = ONN_creation(layers='C_Q_P', phases=phases)
    Model = ONN_creation(layers='R_D_P', phases=phases)
    phases = Model.get_all_phases()
    df = list_to_df(phases) 
    print("\nNeuroptica's original phases:")
    print(df)
    df_far = to_farhad_phases(df, 'reck')
    print("\nNeuroptica's phases converted to Farhad phasing:")
    print(df_far)

    # Model = ONN_creation(layers='R_P', phases=phases[0])

    tf = Model.get_transformation_matrix()
    print(tf)

    # trf_matrix = np.array(Model.get_transformation_matrix_diamond())

    # wf1 = [np.array(trf_matrix[0][0][0]), np.array(trf_matrix[0][1][1]), np.array(trf_matrix[0][2][2])]
    # wf2 = [np.array(trf_matrix[0][1][0]), np.array(trf_matrix[0][2][1]), np.array(trf_matrix[0][3][1])]
    # wf3 = [np.array(trf_matrix[0][2][0]), np.array(trf_matrix[0][3][0]), np.array(trf_matrix[0][4][0])]
    # wf_total = wf1 + wf2 + wf3
    # # for tf in wf_total:
    #     # for elem in tf:
    #         # print((elem))
    #     # print('\n')

    # tf_matrix = reduce(np.dot, wf_total)
    # # print(tf_matrix)
    # # print(tf)
    # with open('transformationmatrix.txt', "w") as myfile:
    #     for elem in tf:
    #         np.savetxt(myfile, elem, fmt='%.2f%+.2fj, '*len(tf[0]), delimiter=', ')
    #         myfile.write('\n')
    # with open('wfTotal.txt', "w") as myfile:
    #     for tf in wf_total:
    #         for elem in tf:
    #             np.savetxt(myfile, elem, fmt='%.2f%+.2fj, '*len(list(elem)), delimiter=', ')
    #             myfile.write('\n')


    # last_phases = Model.get_all_phases()
    # last_phases_flat = [item for sublist in last_phases for item in sublist]
    # df = pd.DataFrame(last_phases_flat, columns=['Theta','Phi'])

    # df.to_csv(f'phases.txt',  float_format='%.3f')
