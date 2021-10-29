''' phase_uncert_thetar simulating Optical Neural Network

using Neuroptica and linearly separable datasets
Now goes over every topology types with N = 4-32

Author: Simon Geoffroy-Gagnon
Edit: 2020.03.28
'''
import numpy as np
import calculate_accuracy as calc_acc
import ONN_Simulation_Class as ONN_Cls
import training_onn as train
import test_trained_onns as test
import ONN_Setups
import create_datasets

onn = ONN_Cls.ONN_Simulation()
onn.BATCH_SIZE = 2**6
onn.EPOCHS = 200
onn.STEP_SIZE = 0.005
onn.ITERATIONS = 50 # number of times to retry same loss/PhaseUncert
onn.rng = 1 # starting RNG value
onn.max_accuracy_req = 98 # (%) Will stop retrying after accuracy above this is reached
onn.max_number_of_tests = 5 # Max number of retries for a single model's training (keeps maximum accuracy model)
onn.range_dB = 10
onn.range_linear = 20

onn.ITERATIONS = 20 # number of times to retry same loss/PhaseUncert
onn_topo = ['B_C_Q_P', 'E_P', 'R_P']

dataset = 'Gauss'
# dataset = 'MNIST'

np.random.seed(onn.rng)
for onn.N in [8]:
    onn.features = onn.N
    onn.classes = onn.N
    loss_diff = [0]
    training_loss = [0]

    for lossDiff in loss_diff:
        for trainLoss in training_loss:
            if dataset == 'Gauss':
                onn, _ = train.get_dataset(onn, onn.rng, SAMPLES=400, EPOCHS=60)
            elif dataset == 'MNIST':
                onn.X, onn.y, onn.Xt, onn.yt = create_datasets.MNIST_dataset(N=onn.N, nsamples=1000)
                

            onn.FOLDER = f'Analysis/N={onn.N}'
            onn.createFOLDER()
            onn.saveSimDataset()

            for onn.topo in onn_topo:
                print(f"Training {onn.topo}")
                max_acc = 0
                onn.loss_diff = lossDiff
                onn.loss_dB = [trainLoss]
                onn.get_topology_name()
                for _ in range(onn.max_number_of_tests):
                    onn.phases = []

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
                        onn.loss_diff = lossDiff # Set loss_diff
                        onn.loss_dB = np.linspace(0, 1.5, 76) # set loss/MZI range
                        onn.phase_uncert_theta = np.linspace(0., 1, 3) # set theta phase uncert range
                        onn.phase_uncert_phi = np.linspace(0., 1, 3) # set phi phase uncert range

                        print('Test Accuracy of validation dataset = {:.2f}%'.format(calc_acc.accuracy(onn, model, onn.Xt, onn.yt)))

                        test.test_PT(onn, onn.Xt, onn.yt, best_model, show_progress=True) # test Phi Theta phase uncertainty accurracy
                        test.test_LPU(onn, onn.Xt, onn.yt, best_model, show_progress=True) # test Loss/MZI + Phase uncert accuracy
                        onn.saveAll(best_model) # Save best model information
                        onn.plotAll() # plot training and tests
                        onn.pickle_save() # save pickled version of the onn class
                        break

