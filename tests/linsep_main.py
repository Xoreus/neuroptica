''' phase_uncert_thetar simulating Optical Neural Network

using Neuroptica and linearly separable datasets
Now goes over every topology types with N = 4-32

Author: Simon Geoffroy-Gagnon
Edit: 2020.03.28
'''
import numpy as np
import calculate_accuracy as calc_acc
import ONN_Simulation_Class as ONN_Cls
import acc_colormap
import training_onn as train
import test_trained_onns as test

ONN = ONN_Cls.ONN_Simulation()
ONN.BATCH_SIZE = 2**6
ONN.EPOCHS = 300
ONN.STEP_SIZE = 0.005

ONN.ITERATIONS = 20 # number of times to retry same loss/PhaseUncert
rng_og = 16
max_rng = 15 # how many repititions 
# onn_topo = ['R_P', 'C_Q_P', 'E_P', 'R_I_P']
onn_topo = ['R_D_I_P']
# onn_topo = ['R_I_D_P','E_D_P', 'R_D_P', 'C_Q_P']
for ONN.N in [4]:
    loss_diff = [0]
    loss_var = [0]

    for ld in loss_diff:
        for lt in loss_var:
            rng = rng_og
            np.random.seed(rng)
            ONN, _ = train.get_dataset(ONN, rng, SAMPLES=40, EPOCHS=60)
            ONN.FOLDER = f'Analysis/N={ONN.N}'
            ONN.createFOLDER()
            ONN.saveSimDataset()

            for ONN.topo in onn_topo:
                max_acc = 0
                ONN.loss_diff = ld
                ONN.loss_dB = [lt]
                ONN.get_topology_name()
                for ONN.rng in range(max_rng):
                    ONN.phases = []
                    ONN, model = train.train_single_onn(ONN, loss_function='mse')

                    if max(ONN.val_accuracy) > max_acc:
                        best_model = model
                        max_acc = max(ONN.val_accuracy) 

                    if max(ONN.val_accuracy) > 95 or ONN.rng == max_rng-1:
                        print(f'max accuracy for {ONN.topo}: {max_acc:.3f}%')
                        ONN.loss_diff = ld
                        ONN.loss_dB = np.linspace(0, 3, 31)
                        ONN.phase_uncert_theta = np.linspace(0., 0.75, 21)
                        ONN.phase_uncert_phi = np.linspace(0., 0.75, 21)
                        test.test_PT(ONN, best_model)
                        test.test_LPU(ONN, best_model)
                        ONN.saveAll(best_model)
                        ONN.pickle_save()
                        break

