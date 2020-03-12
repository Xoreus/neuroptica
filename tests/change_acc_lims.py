''' changes the loss and phase uncert limits for testing different topology sizes
using Neuroptica and linearly separable datasets

Author: Simon Geoffroy-Gagnon
Edit: 20.02.2020
'''
import numpy as np
import calculate_accuracy as calc_acc
import ONN_Simulation_Class as ONN_Cls
import onnClassTraining
import ONN_Setups 
import acc_colormap

def set_phases(onn, phases):
    sep_phases = [[],[]]
    for l in onn.topo:
        if l == '_':
            continue
        elif l == 'C' or l == 'E':
            for phase in phases:
                sep_phases[0].append(phase)
        elif l == 'R' or l == 'I':
            for idx, phase in enumerate(phases):
                if idx < onn.N*(onn.N-1)/2:
                    sep_phases[0].append((phase[0], phase[1]))
                else:
                    sep_phases[1].append((phase[0], phase[1]))
    return sep_phases

if __name__ == '__main__':
    ONN = ONN_Cls.ONN_Simulation()

    ONN.ITERATIONS = 10 # number of times to retry same loss/PhaseUncert
    ONN.loss_diff = 0.0 # \sigma dB
    ONN.loss_dB = np.linspace(0, 1.5, 41)
    ONN.phase_uncert_theta = np.linspace(0., 0.4, 41)
    ONN.phase_uncert_phi = np.linspace(0., 0.4, 41)

    ONN.zeta = 0
    topos = ['R_P', 'C_Q_P']
    # topos = ['R_I_P', 'R_D_I_P', 'R_P']

    for N in [16]:
        for ONN.topo in topos:
            ONN.get_topology_name()
            print(f'N={N}, topo={ONN.topo}')
            FOLDER = f'/home/simon/Documents/neuroptica/tests/Analysis/N=16-PhiTheta/N=16_0.0052'
            ONN.FOLDER = f'/home/simon/Documents/neuroptica/tests/Analysis/N=16-PhiTheta/N=16_0.0052'

            ONN.X = np.loadtxt(FOLDER + '/Datasets/X.txt', delimiter=',')
            ONN.y = np.loadtxt(FOLDER + '/Datasets/y.txt', delimiter=',')
            ONN.Xt = np.loadtxt(FOLDER + '/Datasets/Xt.txt', delimiter=',')
            ONN.yt = np.loadtxt(FOLDER + '/Datasets/yt.txt', delimiter=',')
            ONN.N = N
            phases = np.loadtxt(f'{FOLDER}/Phases/Phases_Best_{ONN.topo}.txt', skiprows=1, usecols=(1,2), delimiter=',')
            phases = [(t, p) for t, p in phases]

            sep_phases = set_phases(ONN, phases)
            ONN.phases = sep_phases
            model = ONN_Setups.ONN_creation(ONN)

            ONN.same_phase_uncert = False
            print('Different Phase Uncert')
            ONN.accuracy_PT = calc_acc.get_accuracy(ONN, model, ONN.Xt, ONN.yt, loss_diff=ONN.loss_diff)

            ONN.same_phase_uncert = True
            print('Same Phase Uncert')
            ONN.accuracy_LPU = calc_acc.get_accuracy(ONN, model, ONN.Xt, ONN.yt, loss_diff=ONN.loss_diff)

            # break
            ONN.createFOLDER()
            ONN.saveSelf()
            acc_colormap.colormap_me(ONN)
            ONN.saveSimDataset()
            ONN.saveAccuracyData()
        np.savetxt(f'{ONN.FOLDER}/all_topologies.txt', topos, fmt='%s')

