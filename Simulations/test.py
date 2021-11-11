''' phase_uncert_thetar simulating Optical Neural Network

Using Neuroptica and linearly separable datasets or MNIST

Author: Simon Geoffroy-Gagnon
Edit: 2020.09.04

Additional Edits
- Add MZI Error into Learning Model

Author: Edward Leung
Edit: 2021.05.15
'''
import numpy as np
from sklearn.preprocessing import MinMaxScaler as mms
import ONN_Simulation_Class as ONN_Cls
from plot_scatter_matrix import plot_scatter_matrix
import ONN_Setups
import training_onn as train
import test_trained_onns as test
import create_datasets
from sklearn import preprocessing
import sys
import matplotlib.pyplot as plt
sys.path.append('../')
import neuroptica as neu
from copy import deepcopy

def init_onn_settings():
    ''' Initialize onn settings for training, testing and simulation '''
    onn = ONN_Cls.ONN_Simulation() # Required for containing training/simulation information

    onn.BATCH_SIZE = 512 # # of input samples per batch
    onn.EPOCHS = 400 # Epochs for ONN training
    onn.STEP_SIZE= 0.06 # Learning Rate
    onn.SAMPLES = 512 # # of samples per class 400

    onn.ITERATIONS = 200 # number of times to retry same loss/PhaseUncert
    onn.rng_og = 1 # starting RNG value
    onn.max_number_of_tests = 1 # Max number of retries for a single model's training (keeps maximum accuracy model) #5
    onn.max_accuracy_req = 99.9 # Will stop retrying after accuracy above this is reached

    onn.features = 16  # How many features? max for MNIST = 784 
    onn.classes = 10 # How many classes? max for MNIST = 10
    onn.N = onn.features # number of ports in device

    onn.zeta = 0.70 # Min diff between max (correct) sample and second sample

    # TO SCALE THE FIELD SUCH THAT POWER IS WITHIN A RANGE OF dB #
    # it is important to note that the ONN takes in FIELD, not POWER #
    # As such, we scale it to the sqrt() of the dB power #
    onn.MinMaxScaling = (0.5623, 1.7783) # For power = [-5 dB, +5 dB]
    onn.MinMaxScaling = (np.sqrt(0.1), np.sqrt(10)) # For power = [-10 dB, +10 dB]
    onn.range_linear = 1

    onn.topo = 'ONN' # Name of the model

    return onn

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

def create_model(features, classes):
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

    print("Clement Layer")

    # If you want multi-layer Diamond Topology
    # model = neu.Sequential([
    #         # neu.AddMaskDiamond(features),
    #         # neu.DiamondLayer(features),
    #         # neu.DropMask(2*features - 2, keep_ports=range(features - 2, 2*features - 2)), # Bottom Diamond Topology
    #         # neu.Activation(nlaf),
    #         # neu.AddMaskDiamond(features),
    #         # neu.DiamondLayer(features),
    #         # neu.DropMask(2*features - 2, keep_ports=range(features - 2, 2*features - 2)), # Bottom Diamond Topology
    #         # neu.Activation(nlaf),
    #         # neu.AddMaskDiamond(features),
    #         # neu.DiamondLayer(features),
    #         # neu.DropMask(2*features - 2, keep_ports=range(features - 2, 2*features - 2)), # Bottom Diamond Topology
    #         # neu.Activation(nlaf),
    #         neu.AddMaskDiamond(features),
    #         neu.DiamondLayer(features),
    #         neu.DropMask(2*features - 2, keep_ports=range(features - 2, 2*features - 2)), # Bottom Diamond Topology
    #         neu.Activation(neu.AbsSquared(features)), # photodetector measurement
    #         #neu.DropMask(features, keep_ports=range(classes)),
    # ])

    # If you want regular Clements (multi-layer) topology
    model = neu.Sequential([
        # neu.ClementsLayer(features),
        # neu.Activation(nlaf),
        # neu.ClementsLayer(features),
        # neu.Activation(nlaf),
        # neu.ClementsLayer(features),
        # neu.Activation(nlaf),
        neu.ClementsLayer(features),
        neu.Activation(neu.AbsSquared(features)), # photodetector measurement
        neu.DropMask(features, keep_ports=range(classes))
    ])

    # # If you want regular Reck (single-layer) topology
    # model = neu.Sequential([
    #     # neu.ReckLayer(features),
    #     # neu.Activation(nlaf),
    #     # neu.ReckLayer(features),
    #     # neu.Activation(nlaf),
    #     # neu.ReckLayer(features),
    #     # neu.Activation(nlaf),
    #     neu.ReckLayer(features),
    #     neu.Activation(neu.AbsSquared(features)), # photodetector measurement
    #     neu.DropMask(features, keep_ports=range(classes)) # Drops the unwanted ports
    # ])
    print(model)
    return model

def save_onn(onn, model, lossDiff=0, trainingLoss=0):
    onn.loss_diff = lossDiff # Set loss_diff
    # For simulation purposes, defines range of loss and phase uncert
    onn.loss_dB = np.linspace(0., 1, 11) # set loss/MZI range
    print(onn.loss_dB)
    onn.phase_uncert_theta = np.linspace(0., 1, 3) # set theta phase uncert range
    onn.phase_uncert_phi = np.linspace(0., 1, 3) # set phi phase uncert range
    # print("\nUsing this for test_LPU")
    # print(model.get_all_phases())
    onn, model = test.test_PT(onn, onn.Xt, onn.yt, model, show_progress=True) # test Phi Theta phase uncertainty accurracy
    onn, model = test.test_LPU(onn, onn.Xt, onn.yt, model, show_progress=True) # test Loss/MZI + Phase uncert accuracy
    onn.saveAll(model) # Save best model information
    onn.plotAll(trainingLoss=trainingLoss) # plot training and tests
    onn.plotBackprop(backprop_legend_location=0)
    ''' Backprop Legend Location Codes:
    'best' 	        0
    'upper right' 	1
    'upper left' 	2
    'lower left' 	3
    'lower right' 	4
    'right'         5
    'center left' 	6
    'center right' 	7
    'lower center' 	8
    'upper center' 	9
    'center'    	10
    '''
    onn.pickle_save() # save pickled version of the onn class

def main():
    onn = init_onn_settings()
    np.random.seed(onn.rng)

    # onn = dataset(onn, dataset='Iris_augment')
    # onn = dataset(onn, dataset='Iris')
    # onn = dataset(onn, dataset='Gauss')
    # onn = dataset(onn, dataset='MNIST')
    onn = dataset(onn, dataset='FFT_MNIST')

    # onn = normalize_dataset(onn, normalization='MinMaxScaling') # dataset -> [Min, Max]
    onn = normalize_dataset(onn, normalization='None')

    model = create_model(onn.features, onn.classes)

    loss_diff = [0] # If loss_diff is used in insertion loss/MZI
    training_loss = [0] # loss used during training

    for lossDiff in loss_diff:
        for trainLoss in training_loss:
            print("Loss Diff", lossDiff)
            print("Training Loss", trainLoss)
            onn.FOLDER = f'Analysis/iris_augment/{onn.features}x{onn.classes}_test' # Name the folder to be created
            onn.createFOLDER() # Creates folder to save this ONN training and simulation info
            onn.saveSimDataset() # save the simulation datasets

            max_acc = 0 # Reset maximum accuracy achieved
            onn.loss_diff = lossDiff
            onn.loss_dB = [trainLoss]
            onn.get_topology_name()
            for test_number in range(onn.max_number_of_tests):
                onn.phases = [] # Reset Saved Phases
                
                # Reset the phases to create new model
                current_phases = model.get_all_phases()
                # current_phases = [[(None, None) for _ in layer] for layer in current_phases]
                model.set_all_phases_uncerts_losses(current_phases, 0, 0, trainLoss, lossDiff)
                onn, model = train.train_single_onn(onn, model, loss_function='mse') # 'cce' for complex models, 'mse' for simple single layer ONNs, use CCE for classification
                
                # print("\nPhase of the Model: ")
                # print(model.get_all_phases())
                # if test_number>0:
                #     print("\nPhase of current best model")
                #     print(best_model.get_all_phases())
                # Save best model
                # print("Checking for greater accuracy")
                if max(onn.val_accuracy) > max_acc:
                    # print("New model is better")
                    best_model = deepcopy(model)
                    #best_onn.model = model
                    best_onn = deepcopy(onn)
                    max_acc = max(onn.val_accuracy) 
                    onn.plotBackprop(backprop_legend_location=0)
                    onn.pickle_save() # save pickled version of the onn class
                    current_phases = best_model.get_all_phases()
                    best_model.set_all_phases_uncerts_losses(current_phases, 0, 0, trainLoss, lossDiff)
                    # print("\nNew Best Model!")
                    # print(best_model.get_all_phases())

                if (max(onn.val_accuracy) > onn.max_accuracy_req or
                        test_number == onn.max_number_of_tests-1):
                    # print("\nThis is the best model")
                    # print(best_model.get_all_phases())
                    print(f'\nBest Accuracy: {max_acc:.2f}%. Using this model for simulations.')
                    save_onn(best_onn, best_model, 0, trainLoss)
                    best_onn.saveForwardPropagation(best_model)
                    current_phases = best_model.get_all_phases()

                    # print("Setting loss_dB 0")
                    # best_model.set_all_phases_uncerts_losses(current_phases, 0, 0, 0, lossDiff)
                    # best_onn.save_correct_classified_samples(best_model)
                    # print("Setting loss_dB", trainLoss)
                    # best_model.set_all_phases_uncerts_losses(current_phases, 0, 0, trainLoss, lossDiff)
                    # best_onn.save_correct_classified_samples(best_model)
                    # print("Setting loss_dB 5")
                    # best_model.set_all_phases_uncerts_losses(current_phases, 0, 0, 5, lossDiff)
                    # best_onn.save_correct_classified_samples(best_model)
                    # print("Setting Loss/MZI = ", trainLoss)
                    # best_model.set_all_phases_uncerts_losses(current_phases, 0, 0, trainLoss, lossDiff)
                    # best_onn.save_correct_classified_samples(best_model)
                    # print("Setting Loss/MZI = 0")
                    # best_model.set_all_phases_uncerts_losses(current_phases, 0, 0, 0, lossDiff)
                    # best_onn.save_correct_classified_samples(best_model)
                    # print("Setting Loss/MZI = 1")
                    # best_model.set_all_phases_uncerts_losses(current_phases, 0, 0, 1, lossDiff)
                    # best_onn.save_correct_classified_samples(best_model)
                    
                    # best_onn.save_correct_classified_samples(best_model, zeta=onn.zeta)
                    # best_onn.save_correct_classified_samples(best_model, zeta=2*onn.zeta)

                    # To plot scattermatrix of dataset
                    axes = plot_scatter_matrix(onn.X, onn.y,  figsize=(15, 15), label='X', start_at=0, fontsz=54)
                    plt.savefig(onn.FOLDER + '/scatterplot.pdf')
                    break

if __name__ == '__main__':
     main()
