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

onn.BATCH_SIZE = 512
onn.EPOCHS = 400
onn.STEP_SIZE = 0.005
onn.SAMPLES = 512

onn.ITERATIONS = 200 # number of times to retry same loss/PhaseUncert
onn.rng = 1 # starting RNG value
onn.max_number_of_tests = 5 # Max number of retries for a single model's training (keeps maximum accuracy model)
onn.max_accuracy_req = 99.9 # (%) Will stop retrying after accuracy above this is reached

onn.features = 16 # How many features? max for MNIST = 784 
onn.classes = 10 # How many classes? max for MNIST = 10
onn.N = onn.features # number of ports in device

#onn.MinMaxScaling = (0.5623, 1.7783) # For power = [-5 dB, +5 dB]
#onn.MinMaxScaling = (np.sqrt(0.1), np.sqrt(10)) # For power = [-10 dB, +10 dB]
onn.range_linear = 20
onn.range_dB = 10

#onn_topo = ['Diamond', 'Clements', 'Reck']
onn_topo = ['Reck']
# onn_topo = ['B_C_Q_P', 'E_P', 'R_P'] #Diamond, Clements, Reck See ONN_Simulation_Class
#onn_topo = ['B_C_Q_P']

def dataset(onn, dataset='MNIST', half_square_length=2):
    ''' Create a dataset for training/testing the ONN
        Choices: Gauss, for a multivariate gaussian dataset
             MNIST, for a PCE'ed MNIST dataset
             FFT MNIST, for the central square of the FFT'ed MNIST
             FFT+PCA,  for the FFT'ed MNIST with PCA'''
    if dataset == 'Gauss':
        onn, onn.rng = train.get_dataset(onn, onn.rng, SAMPLES=onn.SAMPLES, EPOCHS=60, linear_sep_acc_limit=95) # EPOCHS here refers to the number of epochs for digital NN to see if linearly separable
    elif dataset == 'MNIST':
        onn.X, onn.y, onn.Xt, onn.yt = create_datasets.MNIST_dataset(classes=onn.classes, features=onn.features, nsamples=onn.SAMPLES) # this gives real valued vectors as input samples 
    elif dataset == 'FFT_MNIST':
        onn.X, onn.y, onn.Xt, onn.yt = create_datasets.FFT_MNIST(half_square_length=half_square_length, nsamples=onn.SAMPLES) # this gives complex valued vectors
        onn.features = (2*half_square_length)**2
        onn.N = (2*half_square_length)**2
    elif dataset == 'FFT_PCA':
        onn.X, onn.y, onn.Xt, onn.yt = create_datasets.FFT_MNIST_PCA(classes=onn.classes, features=onn.features, nsamples=onn.SAMPLES) # this gives real valued vectors as input samples
    elif dataset == 'Iris': # Gives real valued vectors, 4 features 3 classes
        onn.X, onn.y, onn.Xt, onn.yt = create_datasets.iris_dataset(nsamples=onn.SAMPLES)
        onn.classes = 3
        onn.features = 4
        onn.N = 4
    elif dataset == 'Iris_augment':# Gives real valued vectors, 4 features 4 classes
        onn.X, onn.y, onn.Xt, onn.yt = create_datasets.iris_dataset_augment(divide_mean=1.25, nsamples=onn.SAMPLES)
        onn.classes = 4
        onn.features = 4
        onn.N = 4
    else: 
        print("\nDataset not understood. Use 'Gauss', 'MNIST', 'FFT_MNIST', 'FFT_PCA', 'Iris', or 'Iris_augment'.\n")
    return onn

def normalize_dataset(onn, normalization='MinMaxScaling', experimental=False):
    ''' Constant_Power: Sends extra power to extra channel
        Normalize_Power: Normalize input power --> [0, onn.range_linear]
        MinMaxScaling: Normalize input power to [Min, Max]
    '''
    print(f'Original Dataset range: [{np.min(onn.X):.3f}, {np.max(onn.X):.3f}]')
    if normalization == 'Absolute':
        onn.X = np.abs(onn.X)
        onn.Xt = np.abs(onn.Xt)

    if normalization == 'MinMaxScaling':
        # print(f"Min Max Scaling with range: [{onn.MinMaxScaling[0]:.3f}, {onn.MinMaxScaling[1]:.3f}].")

        # Testing purposes
        # onn.MinMaxScaling = (np.min(onn.X), np.max(onn.X))

        scaler = mms(feature_range=onn.MinMaxScaling)
        onn.X = scaler.fit_transform(onn.X)
        onn.Xt = scaler.fit_transform(onn.Xt)

    if normalization == 'Constant_Power':
        print("Going with 1 extra channel and normalizing power")
        # add an extra channel (+1 ports) and normalize power #
        onn.N += 1

        onn.features += 1
        onn.X = normalize_inputs(onn.X, onn.N)
        # print(onn.X)
        onn.Xt = normalize_inputs(onn.Xt, onn.N)
        print(f"Number of channels: {onn.features}")
    elif normalization == 'Normalized':              
        # To Simply Normalize input power from [0-1]*onn.range_linear #
        if not experimental:
            onn.X = (onn.X - np.min(onn.X))/(np.max(onn.X) - np.min(onn.X))*onn.range_linear
            onn.Xt = (onn.Xt - np.min(onn.Xt))/(np.max(onn.Xt) - np.min(onn.Xt))*onn.range_linear

        # To shift them to useable experimental samples...
        onn.range_dB = 10
        onn.dB_shift = 0.6
        if experimental:
            # change it to dBs
            onn.X = 10*np.log10(np.abs(onn.X)**2+sorted(set(np.abs(onn.X).reshape(-1)))[1])
            # Then normalize it ([0, 1] - shift)*range_dB
            onn.X = ((onn.X - np.min(onn.X))/(np.max(onn.X) - np.min(onn.X)) - onn.dB_shift)*onn.range_dB
            # Then reconvert it back to linear values
            onn.X = 10**(onn.X/10)
            onn.Xt = 10*np.log10(np.abs(onn.Xt)**2+sorted(set(np.abs(onn.Xt).reshape(-1)))[1])
            onn.Xt = ((onn.Xt - np.min(onn.Xt))/(np.max(onn.Xt) - np.min(onn.Xt)) - onn.dB_shift)*onn.range_dB
            onn.Xt = 10**(onn.Xt/10)
    elif normalization == 'Center':
        onn.X = (onn.X - np.min(onn.X))/(np.max(onn.X) - np.min(onn.X)) - 0.5
        onn.Xt = (onn.Xt - np.min(onn.Xt))/(np.max(onn.Xt) - np.min(onn.Xt)) - 0.5
        
    print(f'Using {normalization} scaling, Dataset range: [{np.min(onn.X):.3f}, {np.max(onn.X):.3f}]')
    print(f'Power range: [{10*np.log10(np.min(onn.X)**2):.5f}, {10*np.log10(np.max(onn.X)**2):.5f}] dB')
    return onn

def normalize_inputs(data, num_inputs, P0=10):
    ''' Reshapes the inputs to fit into the specified mesh size and normalizes input data to
    have the same total power input by injecting extra power to an "unused" input port.
    :param X: the input data
    :param num_inputs: the size of the network (number of waveguides)
    :param P0: the total power to inject with each data input
    '''
    _, input_size = data.shape
    injection_port = input_size
    # data = (data - np.min(data))/(np.max(data) - np.min(data))*10
    data = [x**2 for x in data]
    P0 = max(np.sum([x for x in data], axis=1))
    # print(f"Power Required: {P0:.3f}, or {10*np.log10(P0):.3f} dB")
    data_normalized = np.array(np.pad(data, ((0, 0), (0, num_inputs - input_size)), mode="constant"))
    for i, x in enumerate(data_normalized):
        data_normalized[i][injection_port] = np.sqrt(P0 - np.sum(x))
    return data_normalized

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
            neu.AddMaskDiamond(onn.N), # Adds 0s to the top half of the Diamond input
            neu.DiamondLayer(onn.N, include_phase_shifter_layer=False, loss_dB=onn.loss_dB[0], loss_diff=onn.loss_diff, phase_uncert=onn.phase_uncert_theta[0], phases=[(None, None)]), # Diamond Mesh
            neu.DropMask(2*onn.N - 2, keep_ports=range(onn.N - 2, 2*onn.N - 2)), # Bottom Diamond Topology
            neu.Activation(neu.AbsSquared(features)), # photodetector measurement
            neu.DropMask(features, keep_ports=range(classes)),
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
            neu.ClementsLayer(onn.N, include_phase_shifter_layer=False, loss_dB=onn.loss_dB[0], loss_diff=onn.loss_diff, phase_uncert=onn.phase_uncert_theta[0], phases=[(None, None)]),
            neu.Activation(neu.AbsSquared(features)), # photodetector measurement
            neu.DropMask(features, keep_ports=range(classes))
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
            neu.ReckLayer(features),
            neu.Activation(neu.AbsSquared(features)), # photodetector measurement
            neu.DropMask(features, keep_ports=range(classes)) # Drops the unwanted ports
        ])
    print(model)
    return model


accuracy_dict = []

# onn = dataset(onn, dataset='Iris_augment')
# onn = dataset(onn, dataset='Iris')
# onn = dataset(onn, dataset='Gauss')
onn = dataset(onn, dataset='FFT_MNIST')

onn = normalize_dataset(onn, normalization='None')

np.random.seed(onn.rng)
for onn.N in [16]:
    onn.features = onn.N
    onn.classes = onn.N
    loss_diff = [0]
    training_loss = [0]


    for lossDiff in loss_diff:
        for trainLoss in training_loss:
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
                    current_phases = [[(None, None) for _ in layer] for layer in current_phases]
                    model.set_all_phases_uncerts_losses(current_phases, 0, 0, trainLoss, lossDiff)
                    onn, model = train.train_single_onn(onn, model, loss_function='cce') # 'cce' for complex models, 'mse' for simple single layer ONNs
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
