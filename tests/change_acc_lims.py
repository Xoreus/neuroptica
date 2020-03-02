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

def set_phases(onn, phases):
    sep_phases = [[],[]]
    for l in onn.onn_topo:
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
    ONN.loss_diff = 0 # \sigma dB
    ONN.loss_dB = np.linspace(0, 1, 41)
    # ONN.loss_dB = [0]
    ONN.phase_uncert_theta = np.linspace(0.,0.5, 41)
    ONN.phase_uncert_phi = np.linspace(0., 0.5, 41)

    # ONN.same_phase_uncert = True
    ONN.same_phase_uncert = False
    ONN.loss_dB = [0]

    ONN.zeta = 0
    topos = ['R_P', 'C_Q_P']
    # topos = ['E_P']

    for N in [8*2*2]:
        for ONN.onn_topo in topos:
            ONN.get_topology_name()
            print(f'N={N}, topo={ONN.onn_topo}')
            FOLDER = f'/home/simon/Documents/neuroptica/tests/Analysis/linsep/N=32_0'
            ONN.FOLDER = f'Analysis/linsep/N={N}_0-Loss+PU_FINAL'
            ONN.FOLDER = f'Analysis/linsep/N={N}_0-PhiTheta_FINAL'
            ONN.X = np.loadtxt(FOLDER + '/Datasets/X.txt', delimiter=',')
            ONN.y = np.loadtxt(FOLDER + '/Datasets/y.txt', delimiter=',')
            ONN.Xt = np.loadtxt(FOLDER + '/Datasets/Xt.txt', delimiter=',')
            ONN.yt = np.loadtxt(FOLDER + '/Datasets/yt.txt', delimiter=',')
            ONN.N = N
            phases = np.loadtxt(f'{FOLDER}/Phases/Phases_Best_{ONN.onn_topo}.txt', skiprows=1, usecols=(1,2), delimiter=',')
            phases = [(t, p) for t, p in phases]

            sep_phases = set_phases(ONN, phases)
            ONN.phases = sep_phases
            model = ONN_Setups.ONN_creation(ONN)
            ONN.accuracy = calc_acc.get_accuracy(ONN, model, ONN.Xt, ONN.yt, loss_diff=ONN.loss_diff)

            # break
            ONN.createFOLDER()
            ONN.saveSelf()
            ONN.saveSimDataset()
            ONN.saveAccuracyData()
    np.savetxt(f'{ONN.FOLDER}/all_topologies.txt', topos, fmt='%s')

