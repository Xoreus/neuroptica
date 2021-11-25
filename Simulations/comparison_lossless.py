''' phase_uncert_thetar simulating Optical Neural Network

using Neuroptica and linearly separable datasets
Now goes over every topology types with N = 4-32

Author: Simon Geoffroy-Gagnon
Edit: 2020.03.28
'''
import numpy as np
from sklearn.preprocessing import MinMaxScaler as mms
import calculate_accuracy as calc_acc
import ONN_Simulation_Class as ONN_Cls
import training_onn as train
import test_trained_onns as test
import ONN_Setups
import create_datasets
from copy import deepcopy
from plot_scatter_matrix import plot_scatter_matrix
import matplotlib.pyplot as plt
import neuroptica as neu

onn = ONN_Cls.ONN_Simulation()

onn.BATCH_SIZE = 400
onn.EPOCHS = 200
onn.STEP_SIZE = 0.005
onn.SAMPLES = 400

onn.ITERATIONS = 50 # number of times to retry same loss/PhaseUncert
onn.rng = 1 # starting RNG value
onn.max_number_of_tests = 10 # Max number of retries for a single model's training (keeps maximum accuracy model)
onn.max_accuracy_req = 99.9 # (%) Will stop retrying after accuracy above this is reached

onn.features = 8 # How many features? max for MNIST = 784 
onn.classes = 8 # How many classes? max for MNIST = 10
onn.N = onn.features # number of ports in device

#onn.MinMaxScaling = (0.5623, 1.7783) # For power = [-5 dB, +5 dB]
#onn.MinMaxScaling = (np.sqrt(0.1), np.sqrt(10)) # For power = [-10 dB, +10 dB]
onn.range_linear = 20
onn.range_dB = 10

onn_topo = ['Diamond', 'Clements', 'Reck']
# onn_topo = ['B_C_Q_P', 'E_P', 'R_P'] #Diamond, Clements, Reck See ONN_Simulation_Class
#onn_topo = ['B_C_Q_P']

def create_model(features, classes, topo):
    ''' create ONN model based on neuroptica layer '''
    eo_settings = {'alpha': 0.1, 'g':0.5 * np.pi, 'phi_b': -1 * np.pi} # If Electro-Optic Nonlinear Activation is used

    # Some nonlinearities, to be used withing neu.Activation()
    eo_activation = neu.ElectroOpticActivation(features, **eo_settings)
    cReLU = neu.cReLU(features)
    zReLU = neu.zReLU(features)
    bpReLU = neu.bpReLU(features, cutoff=1, alpha=0.1)
    modReLU = neu.modReLU(features, cutoff=1)
    sigmoid = neu.Sigmoid(features)
    
    nlaf = cReLU # Pick the Non Linear Activation Function

    # If you want multi-layer Diamond Topology
    if topo == 'Diamond':
        model = neu.Sequential([
            # neu.AddMaskDiamond(onn.N), # Adds 0s to the top half of the Diamond input
            # neu.DiamondLayer(onn.N, include_phase_shifter_layer=False, loss_dB=onn.loss_dB[0], loss_diff=onn.loss_diff, phase_uncert=onn.phase_uncert_theta[0], phases=[(None, None)]), # Diamond Mesh
            # neu.DropMask(2*onn.N - 2, keep_ports=range(onn.N - 2, 2*onn.N - 2)), # Bottom Diamond Topology
            # neu.Activation(nlaf),
            # neu.AddMaskDiamond(onn.N), # Adds 0s to the top half of the Diamond input
            # neu.DiamondLayer(onn.N, include_phase_shifter_layer=False, loss_dB=onn.loss_dB[0], loss_diff=onn.loss_diff, phase_uncert=onn.phase_uncert_theta[0], phases=[(None, None)]), # Diamond Mesh
            # neu.DropMask(2*onn.N - 2, keep_ports=range(onn.N - 2, 2*onn.N - 2)), # Bottom Diamond Topology
            # neu.Activation(nlaf),
            # neu.AddMaskDiamond(onn.N), # Adds 0s to the top half of the Diamond input
            # neu.DiamondLayer(onn.N, include_phase_shifter_layer=False, loss_dB=onn.loss_dB[0], loss_diff=onn.loss_diff, phase_uncert=onn.phase_uncert_theta[0], phases=[(None, None)]), # Diamond Mesh
            # neu.DropMask(2*onn.N - 2, keep_ports=range(onn.N - 2, 2*onn.N - 2)), # Bottom Diamond Topology
            # neu.Activation(nlaf),
            # neu.AddMaskDiamond(onn.N), # Adds 0s to the top half of the Diamond input
            # neu.DiamondLayer(onn.N, include_phase_shifter_layer=False, loss_dB=onn.loss_dB[0], loss_diff=onn.loss_diff, phase_uncert=onn.phase_uncert_theta[0], phases=[(None, None)]), # Diamond Mesh
            # neu.DropMask(2*onn.N - 2, keep_ports=range(onn.N - 2, 2*onn.N - 2)), # Bottom Diamond Topology
            # neu.Activation(nlaf),
            neu.AddMaskDiamond(onn.N), # Adds 0s to the top half of the Diamond input
            neu.DiamondLayer(onn.N, include_phase_shifter_layer=False, loss_dB=onn.loss_dB[0], loss_diff=onn.loss_diff, phase_uncert=onn.phase_uncert_theta[0], phases=[(None, None)]), # Diamond Mesh
            neu.DropMask(2*onn.N - 2, keep_ports=range(onn.N - 2, 2*onn.N - 2)), # Bottom Diamond Topology
            neu.Activation(neu.AbsSquared(features)), # photodetector measurement
            #neu.DropMask(features, keep_ports=range(classes)),
        ])
    elif topo == 'Clements':
    # If you want regular Clements (multi-layer) topology
        model = neu.Sequential([
            # neu.ClementsLayer(onn.N, include_phase_shifter_layer=False, loss_dB=onn.loss_dB[0], loss_diff=onn.loss_diff, phase_uncert=onn.phase_uncert_theta[0], phases=[(None, None)]),
            # neu.Activation(nlaf),
            # neu.ClementsLayer(onn.N, include_phase_shifter_layer=False, loss_dB=onn.loss_dB[0], loss_diff=onn.loss_diff, phase_uncert=onn.phase_uncert_theta[0], phases=[(None, None)]),
            # neu.Activation(nlaf),
            # neu.ClementsLayer(onn.N, include_phase_shifter_layer=False, loss_dB=onn.loss_dB[0], loss_diff=onn.loss_diff, phase_uncert=onn.phase_uncert_theta[0], phases=[(None, None)]),
            # neu.Activation(nlaf),
            # neu.ClementsLayer(onn.N, include_phase_shifter_layer=False, loss_dB=onn.loss_dB[0], loss_diff=onn.loss_diff, phase_uncert=onn.phase_uncert_theta[0], phases=[(None, None)]),
            # neu.Activation(nlaf),
            neu.ClementsLayer(onn.N, include_phase_shifter_layer=False, loss_dB=onn.loss_dB[0], loss_diff=onn.loss_diff, phase_uncert=onn.phase_uncert_theta[0], phases=[(None, None)]),
            neu.Activation(neu.AbsSquared(features)), # photodetector measurement
            #neu.DropMask(features, keep_ports=range(classes))
        ])
    elif topo == 'Reck':
    # If you want regular Reck (single-layer) topology
        model = neu.Sequential([
            # neu.ReckLayer(onn.N, include_phase_shifter_layer=False, loss_dB=onn.loss_dB[0], loss_diff=onn.loss_diff, phase_uncert=onn.phase_uncert_theta[0], phases=[(None, None)]),
            # neu.Activation(nlaf),
            # neu.ReckLayer(onn.N, include_phase_shifter_layer=False, loss_dB=onn.loss_dB[0], loss_diff=onn.loss_diff, phase_uncert=onn.phase_uncert_theta[0], phases=[(None, None)]),
            # neu.Activation(nlaf),
            # neu.ReckLayer(onn.N, include_phase_shifter_layer=False, loss_dB=onn.loss_dB[0], loss_diff=onn.loss_diff, phase_uncert=onn.phase_uncert_theta[0], phases=[(None, None)]),
            # neu.Activation(nlaf),
            # neu.ReckLayer(onn.N, include_phase_shifter_layer=False, loss_dB=onn.loss_dB[0], loss_diff=onn.loss_diff, phase_uncert=onn.phase_uncert_theta[0], phases=[(None, None)]),
            # neu.Activation(nlaf),
            neu.ReckLayer(onn.N, include_phase_shifter_layer=False, loss_dB=onn.loss_dB[0], loss_diff=onn.loss_diff, phase_uncert=onn.phase_uncert_theta[0], phases=[(None, None)]),
            neu.Activation(neu.AbsSquared(features)), # photodetector measurement
            #neu.DropMask(features, keep_ports=range(classes)) # Drops the unwanted ports
        ])
    print(model)
    return model

accuracy_dict = []

dataset = 'Gauss'
#dataset = 'MNIST'

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
                onn.X, onn.y, onn.Xt, onn.yt = create_datasets.MNIST_dataset(classes=onn.classes, features=onn.features, nsamples=onn.SAMPLES)
                

            onn.FOLDER = f'Analysis/N={onn.N}'
            onn.createFOLDER()
            onn.saveSimDataset()

            for onn.topo in onn_topo:
                print(f"Training {onn.topo}")
                print("Loss Diff", lossDiff)
                print("Training Loss", trainLoss)
                max_acc = 0
                onn.loss_diff = lossDiff
                onn.loss_dB = [trainLoss]
                onn.get_topology_name()
                for test_number in range(onn.max_number_of_tests):
                    onn.phases = []

                    # model = ONN_Setups.ONN_creation(onn)
                    model = create_model(onn.features, onn.classes, onn.topo)
                    current_phases = model.get_all_phases()
                    # print("Starting Phases Get All Phases")
                    # print(current_phases)
                    current_phases = [[(None, None) for _ in layer] for layer in current_phases]
                    # print("Setting Random Phases")
                    # print(current_phases)
                    model.set_all_phases_uncerts_losses(current_phases, 0, 0, trainLoss, lossDiff)
                    # print("Setting Phases")
                    # print(model.get_all_phases())
                    onn, model = train.train_single_onn(onn, model, loss_function='mse') # 'cce' for complex models, 'mse' for simple single layer ONNs
                    #print("Ending Phases")
                    #print(model.get_all_phases())
                    # Save best model
                    if max(onn.val_accuracy) > max_acc:
                        best_model = deepcopy(model)
                        best_onn = deepcopy(onn)
                        max_acc = max(onn.val_accuracy) 
                        onn.pickle_save() # save pickled version of the onn class
                        current_phases = best_model.get_all_phases()
                        best_model.set_all_phases_uncerts_losses(current_phases, 0, 0, trainLoss, lossDiff)
                    if (max(onn.val_accuracy) > onn.max_accuracy_req or
                            test_number == onn.max_number_of_tests-1):
                        print(f'\nBest Accuracy: {max_acc:.2f}%. Using this model for simulations.')
                        best_onn.loss_diff = lossDiff # Set loss_diff
                        best_onn.loss_dB = np.linspace(0, 1, 51) # set loss/MZI range
                        print(best_onn.loss_dB)
                        best_onn.phase_uncert_theta = np.linspace(0., 1, 3) # set theta phase uncert range
                        best_onn.phase_uncert_phi = np.linspace(0., 1, 3) # set phi phase uncert range

                        print('Test Accuracy of validation dataset = {:.2f}%'.format(calc_acc.accuracy(best_onn, best_model, best_onn.Xt, best_onn.yt)))

                        test.test_SLPU(best_onn, best_onn.Xt, best_onn.yt, best_model, show_progress=True)
                        accuracy_dict.append(best_onn.accuracy_LPU)
                        #onn.saveAll(best_model) # Save best model information
                        #onn.plotAll(trainingLoss=trainLoss) # plot training and tests
                        best_onn.plotBackprop(backprop_legend_location=0)
                        best_onn.saveForwardPropagation(best_model)
                        current_phases = best_model.get_all_phases()
                        axes = plot_scatter_matrix(best_onn.X, best_onn.y,  figsize=(15, 15), label='X', start_at=0, fontsz=54)
                        plt.savefig(best_onn.FOLDER + '/scatterplot.pdf')
                        plt.clf()
                        break

labels_size = 20
legend_size = 16
tick_size = 14
color = 'tab:blue'
fig, ax = plt.subplots(figsize=(8.27, 8.27), dpi=100) #11.69, 8.27
ax.tick_params(axis='both', which='major', labelsize=tick_size)
ax.tick_params(axis='both', which='minor', labelsize=tick_size)
ax.set_xlabel('Loss/MZI (dB)', fontsize=labels_size)
ax.set_ylabel("Validation Accuracy (\%)", fontsize=labels_size)
lns0 = ax.plot(best_onn.loss_dB, accuracy_dict[0], color='#edb120', label=onn_topo[0])
lns1 = ax.plot(best_onn.loss_dB, accuracy_dict[1], color='#d95319', label=onn_topo[1])
lns2 = ax.plot(best_onn.loss_dB, accuracy_dict[2], color='#0072bd', label=onn_topo[2])
ax.set_ylim([0, 100])
lns = lns0+lns1+lns2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0, fontsize=legend_size)
fig.tight_layout() 
plt.savefig(best_onn.FOLDER + '/comparison.pdf')
plt.clf()
