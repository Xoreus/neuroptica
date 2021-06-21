''' phase_uncert_thetar simulating Optical Neural Network
using Neuroptica and linearly separable datasets
Now goes over every topology types with N = 4-32
Gets the average of power at each ports

Author: Simon Geoffroy-Gagnon
Edit: 2020.09.03
'''

import numpy as np
import calculate_accuracy as calc_acc
import ONN_Simulation_Class as ONN_Cls
import training_onn as train
import test_trained_onns as test
import ONN_Setups
import create_datasets
import training_onn as train
import test_trained_onns as test
from collections import defaultdict

onn = ONN_Cls.ONN_Simulation()
onn.BATCH_SIZE = 2**6
onn.EPOCHS = 250
onn.STEP_SIZE = 0.005
onn.ITERATIONS = 5 # number of times to retry same loss/PhaseUncert
onn.loss_diff = 0 # \sigma dB
onn.SAMPLES = 20
onn.zeta = 0.75
onn.max_accuracy_req = 90 # (%) Will stop retrying after accuracy above this is reached
onn.max_number_of_tests = 10 # Max number of retries for a single model's training (keeps maximum accuracy model)

onn_topo = ['R_P','B_C_Q_P','E_P','R_I_P']

output_pwer = defaultdict(list)
input_pwer = defaultdict(list)
onn.rng = 23234
onn.Ns = [16] 
# for ii in range(5, 10):
for ii in range(20, 40):
    for onn.N in onn.Ns:
        onn.features = onn.N
        onn.classes = onn.N
        onn.FOLDER = f'/home/edwar/Documents/Github_Projects/neuroptica/tests/Analysis/outPorts_mean_pow/N={onn.N}_{ii}'
        onn, onn.rng = train.get_dataset(onn, onn.rng, SAMPLES=onn.SAMPLES, EPOCHS=60, linear_sep_acc_limit=95)

        onn.createFOLDER()
        onn.saveSimDataset()

        for onn.topo in onn_topo:
            onn.get_topology_name()
            print(onn.topo)
            max_acc = 0
            for _ in range(10):
                onn.loss_dB = [0]
                model = ONN_Setups.ONN_creation(onn)
                onn, model = train.train_single_onn(onn, model, loss_function='mse') # 'cce' for complex models, 'mse' for simple single layer ONNs
                # Save best model
                if max(onn.val_accuracy) > max_acc:
                    best_model = model
                    onn.model = model
                    best_onn = onn
                    max_acc = max(onn.val_accuracy) 

                if (max(onn.val_accuracy) > onn.max_accuracy_req or
                        onn.rng == onn.max_number_of_tests-1):
                    onn.loss_dB = np.linspace(0, 1, 11) # set loss/MZI range
                    onn.phase_uncert_theta = np.linspace(0., 1, 11) # set theta phase uncert range
                    onn.phase_uncert_phi = np.linspace(0., 1, 11) # set phi phase uncert range
                    onn.PT_Area = (onn.phase_uncert_phi[1] - onn.phase_uncert_phi[0])**2
                    onn.LPU_Area = (onn.loss_dB[1] - onn.loss_dB[0])*(onn.phase_uncert_phi[1] - onn.phase_uncert_phi[0])

                    print('Test Accuracy of validation dataset = {:.2f}%'.format(calc_acc.accuracy(onn, model, onn.Xt, onn.yt)))

                    test.test_PT(onn, onn.Xt, onn.yt, best_model, show_progress=True) # test Phi Theta phase uncertainty accurracy
                    test.test_LPU(onn, onn.Xt, onn.yt, best_model, show_progress=True) # test Loss/MZI + Phase uncert accuracy
                    onn.saveAll(best_model) # Save best model information

                    model.set_all_phases_uncerts_losses(onn.phases, 0, 0, 0, 0)
                    # X, _, Xt, _ = onnClassTraining.change_dataset_shape(onn)
                    out = []

                    for x in onn.X:
                        out.append(model.forward_pass(np.array([x]).T))
                    output_pwer[onn.topo].append(np.mean(np.sum(out, axis=1)))
                    input_pow = [[x**2 for x in sample] for sample in onn.X]
                    onn.out_pwer = out
                    onn.in_pwer = input_pow
                    onn.saveAll(model)
                    onn.pickle_save()
                    break


