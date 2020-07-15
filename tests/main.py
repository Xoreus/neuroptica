''' phase_uncert_thetar simulating Optical Neural Network

using Neuroptica and linearly separable datasets or MNIST

Author: Simon Geoffroy-Gagnon
Edit: 2020.07.14
'''
import numpy as np
import calculate_accuracy as calc_acc
import cmath
import ONN_Simulation_Class as ONN_Cls
import ONN_Setups
import acc_colormap
import training_onn as train
import test_trained_onns as test
import create_datasets
from sklearn import preprocessing
import sys
sys.path.append('../')
import neuroptica as neu
import cmath
import copy

def init_onn_settings():
    ''' Initialize onn settings for training, testing and simulation '''
    onn = ONN_Cls.ONN_Simulation() # Required for containing training/simulation information
    onn.BATCH_SIZE = 2**6 
    onn.EPOCHS = 400
    onn.STEP_SIZE = 0.0025 # Learning Rate
    onn.SAMPLES = 80 # Per Class

    onn.ITERATIONS = 30 # number of times to retry same loss/PhaseUncert
    onn.rng_og = 0 # starting RNG value
    onn.max_number_of_tests = 10 # Max number of retries for a single model's training (keeps maximum accuracy model)
    onn.max_accuracy_req = 40 # (%) Will stop retrying after accuracy above this is reached

    onn.features = 16 # How many features? max for MNIST = 784 # Add +1 if using normalize_input()
    onn.classes = 10 # How many classes? max for MNIST = 10
    onn.N = onn.features

    onn.range_dB = 10
    onn.range_linear = 20

    onn.FOLDER = f'Analysis/N={onn.N}' # Name the folder to be created
    onn.topo = 'E_P'

    return onn

def dataset(onn, dataset='MNIST', half_square_length=2):
    ''' Create a dataset for training/testing the ONN
        Choices: Gauss, for a multivariate gaussian dataset
             MNIST, for a PCE'ed MNIST dataset
             FFT MNIST, for the central square of the FFT'ed MNIST
             FFT+PCA,  for the FFT'ed MNIST with PCA'''
    if dataset == 'Gauss':
        onn, _ = train.get_dataset(onn, onn.rng, SAMPLES=onn.SAMPLES, EPOCHS=60)
    elif dataset == 'MNIST':
        onn.X, onn.y, onn.Xt, onn.yt = create_datasets.MNIST_dataset(classes=onn.classes, features=onn.features, nsamples=onn.SAMPLES) # this gives real valued vectors as input samples 
    elif dataset == 'FFT_MNIST':
        onn.X, onn.y, onn.Xt, onn.yt = create_datasets.FFT_MNIST(half_square_length=half_square_length, nsamples=onn.SAMPLES) # this gives complex valued vectors
        onn.features = (2*half_square_length)**2
        onn.N = (2*half_square_length)**2
    elif dataset == 'FFT_PCA':
        onn.X, onn.y, onn.Xt, onn.yt = create_datasets.FFT_MNIST_PCA(classes=onn.classes, features=onn.features, nsamples=onn.SAMPLES) # this gives real valued vectors as input samples
    return onn

def normalize_dataset(onn, normalization='Normalized', experimental=False):
    ''' Constant Power: Sends extra power to extra channel
        Normalized: Normalize input power --> [0, onn.range_linear]
    '''
    if normalization == 'Constant Power':
        # add an extra channel (+1 ports) and normalize power #
        onn.X = normalize_inputs(on.X, onn.N)
        onn.Xt = normalize_inputs(onn.Xt, onn.N)
        onn.features += 1
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
    data = (data - np.min(data))/(np.max(data) - np.min(data))
    data_normalized = np.array(np.pad(data, ((0, 0), (0, num_inputs - input_size)), mode="constant"))
    for i, x in enumerate(data_normalized):
        data_normalized[i][injection_port] = np.sqrt(P0 - np.sum(x**2))
    return data_normalized

def create_model(features, classes):
    ''' create ONN model based on neuroptica layer '''
    eo_settings = {'alpha': 0.1, 'g':0.5 * np.pi, 'phi_b': -1 * np.pi} # If Electro-Optic Nonlinear Activation is used

    # If you want multi-layer Diamond Topology
    # model = neu.Sequential([
    #     neu.AddMaskDiamond(features),
    #     neu.DiamondLayer(features),
    #     neu.DropMask(2*features - 2, keep_ports=range(features - 2, 2*features - 2)), # Bottom Diamond Topology
    #     neu.Activation(neu.cReLU(features)),
    #     neu.AddMaskDiamond(features),
    #     neu.DiamondLayer(features),
    #     neu.DropMask(2*features - 2, keep_ports=range(features - 2, 2*features - 2)), # Bottom Diamond Topology
    #     neu.Activation(neu.AbsSquared(features)), # photodetector measurement
    #     neu.DropMask(features, keep_ports=range(classes)),
    # ])

    # If you want regular Clements (multi-layer) topology
    model = neu.Sequential([
        neu.ClementsLayer(features),
        neu.Activation(neu.cReLU(features)),
        neu.ClementsLayer(features),
        neu.Activation(neu.cReLU(features)),
        neu.ClementsLayer(features),
        neu.Activation(neu.cReLU(features)),
        neu.ClementsLayer(features),
        neu.Activation(neu.AbsSquared(features)), # photodetector measurement
        neu.DropMask(features, keep_ports=range(classes))
    ])

    # If you want regular Clements (single-layer) topology
    # model = neu.Sequential([
    #     neu.ClementsLayer(features),
    #     neu.Activation(neu.AbsSquared(features)), # photodetector measurement
    #     neu.DropMask(features, keep_ports=range(classes))
    # ])
    return model

def main():
    onn = init_onn_settings()
    np.random.seed(onn.rng)
    onn = dataset(onn, dataset='MNIST')
    onn = normalize_dataset(onn, normalization='Normalized', experimental=True)
    model = create_model(onn.features, onn.classes)

    loss_diff = [0] # If loss_diff is used in insertion loss/MZI
    training_loss = [0] # loss used during training

    for lossDiff in loss_diff:
        for trainLoss in training_loss:
            onn.createFOLDER() # Creates folder to save this ONN training and simulation info
            onn.saveSimDataset() # save the simulation datasets

            max_acc = 0 # Reset maximum accuracy achieved
            onn.loss_diff = lossDiff
            onn.loss_dB = [trainLoss]
            for onn.rng in range(onn.max_number_of_tests):
                onn.phases = [] # Reset Saved Phases
                
                # Reset the phases to create new model
                current_phases = model.get_all_phases()
                current_phases = [[(None, None) for _ in layer] for layer in current_phases]
                model.set_all_phases_uncerts_losses(current_phases)

                onn, model = train.train_single_onn(onn, model, loss_function='cce') # 'cce' for complex models, 'mse' for simple single layer ONNs

                # Save best model
                if max(onn.val_accuracy) > max_acc:
                    best_model = model
                    onn.model = model
                    best_onn = onn
                    max_acc = max(onn.val_accuracy) 

                if (max(onn.val_accuracy) > onn.max_accuracy_req or
                        onn.rng == onn.max_number_of_tests-1):
                    onn.loss_diff = lossDiff # Set loss_diff
                    onn.loss_dB = np.linspace(0, 2, 3) # set loss/MZI range
                    onn.phase_uncert_theta = np.linspace(0., 1, 3) # set theta phase uncert range
                    onn.phase_uncert_phi = np.linspace(0., 1, 3) # set phi phase uncert range

                    print('Test Accuracy of validation dataset = {:.2f}%'.format(calc_acc.accuracy(onn, model, onn.Xt, onn.yt)))

                    test.test_PT(onn, onn.Xt, onn.yt, best_model, show_progress=True) # test Phi Theta phase uncertainty accurracy
                    test.test_LPU(onn, onn.Xt, onn.yt, best_model, show_progress=True) # test Loss/MZI + Phase uncert accuracy
                    onn.saveAll(best_model) # Save best model information
                    onn.plotAll() # plot training and tests
                    onn.pickle_save() # save pickled version of the onn class
                    break

if __name__ == '__main__':
     main()
