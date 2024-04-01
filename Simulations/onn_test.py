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
plt.rcParams.update(plt.rcParamsDefault)
sys.path.append('../')
import neuroptica as neu


def see_each_mzi(p_onn):
    '''
    Helper Function to visualize each MZI's (sigma_theta, sigma_phi, loss)

    sigma is the standard deviation of the normal distribution from which the phase error is drawn
    see the method "get_transfer_matrix()" in components.py
    '''
    print("\n------------------------------------------------------------------------------------------------")
    MZImesh = [layer for layer in p_onn.model.layers if isinstance(layer, neu.OpticalMeshNetworkLayer)]
    print(f"There are {len(MZImesh)} MZImesh(es) in the model.")
    for i in range(len(MZImesh)):
        mzi_layers = MZImesh[i].mesh.layers # list of objects <MZILayer>
        print(f"There are {len(mzi_layers)} MZILayers in this {p_onn.N}x{p_onn.N} '{p_onn.topo}' topology, onn layer: {i}")
        for eachMZILayer in mzi_layers:
            for eachMZI in eachMZILayer.mzis:
                # you can also print other information related to each MZI here, such as the theta/phi phases...
                print(f"({eachMZI.theta:.3f}, {eachMZI.phase_uncert_phi:.3f})", end="")
            print("\n")
    print("------------------------------------------------------------------------------------------------")

no_tst_inst = 800 # display first "no_tst_inst" test samples from onn.Xt, max=160

onn = ONN_Cls.ONN_Simulation() # Required for containing training/simulation information
onn.topo = 'MNIST_Double_Reck_full'
# onn.FOLDER = '/home/edwar/Documents/Github_Projects/neuroptica/tests/Analysis/iris_augment/4x3'
onn.FOLDER = '/Users/bokunzhao/Documents/McGill/Winter2024-G1/ECSE 691_697/WIDER_IOU_50_8x8/neuroptica/Simulations/Analysis/iris_augment/10x10_MNIST'
onn = onn.pickle_load()
model = onn.model
model.set_all_phases_uncerts_losses(Phases=onn.phases) # resets effect of uncertainty simulation


mesh_no = 0 # 0, 1, 2... select mesh layer
col_no = 9 # 0, 1, 2... select "MZI column" within a mesh layer
row_no = 1 # 0, 1, 2... select which MZI within a "MZI column"

# W_onn = model.get_transformation_matrix()[mesh_no]
# print(f"ONN phases:\n")
# see_each_mzi(onn)
# print(f"\nOld Transformation matrix:\n{W_onn}\n")

# MZImesh = [layer for layer in model.layers if isinstance(layer, neu.OpticalMeshNetworkLayer)]
# mzi_to_tune = MZImesh[mesh_no].mesh.layers[col_no].mzis[row_no]
# mzi_to_tune.theta = 0.00
# mzi_to_tune.phi = 0.00

# W_onn = model.get_transformation_matrix()[mesh_no]
# print(f"New phases:\n")
# see_each_mzi(onn)
# print(f"\nNew Transformation matrix:\n{W_onn}\n")
# exit()


yhat = model.forward_pass(onn.Xt[:no_tst_inst, :].T)
cls = np.array([np.argmax(yhat) for yhat in yhat.T])
gt = np.array([np.argmax(tru) for tru in onn.yt]) # yt: one-hot encoded vectors; gt: actual scalar labels
# print(onn.model.get_all_phases())
# print(cls)
# print(gt)
print(f"Testing with first {no_tst_inst} test data instances:")
print(f"first {no_tst_inst} Xt:\n{onn.Xt[:no_tst_inst, :].T}") # (10, no_tst_inst) for MNIST dataset
print(f"first {no_tst_inst} yhat:\n{yhat}") # (10, no_tst_inst) for MNIST dataset
print(f"first {no_tst_inst} labels:\n{gt[:no_tst_inst]}")
print(f"\n")
# print(f"first {no_tst_inst} Xt summation:\n{np.sum(onn.Xt[:no_tst_inst, :].T, axis=0)}")
print(f"first {no_tst_inst} Xt squared summation:\n{np.sum(onn.Xt[:no_tst_inst, :].T**2, axis=0)}")
# print(f"first {no_tst_inst} Xt abs. summation:\n{np.sum(np.abs(onn.Xt[:no_tst_inst, :].T), axis=0)}")
print(f"\n")
print(f"first {no_tst_inst} yhat summation:\n{np.sum(yhat, axis=0)}") # (10, no_tst_inst) for MNIST dataset
# print(f"first {no_tst_inst} yhat squared summation:\n{np.sum(yhat**2, axis=0)}") # (10, no_tst_inst) for MNIST dataset
# print(f"first {no_tst_inst} yhat Re. summation:\n{np.sum(np.real(yhat), axis=0)}") # (10, no_tst_inst) for MNIST dataset
# print(f"first {no_tst_inst} yhat Re. squared summation:\n{np.sum(np.real(yhat)**2, axis=0)}") # (10, no_tst_inst) for MNIST dataset
print(f"first {no_tst_inst} yhat abs. squared summation:\n{np.sum(np.abs(yhat)**2, axis=0)}") # (10, no_tst_inst) for MNIST dataset
# print(f"first {no_tst_inst} yhat sqrt. summation:\n{np.sum(yhat**0.5, axis=0)}") # (10, no_tst_inst) for MNIST dataset

# print(yhat.shape)
print(f'Accuracy = {sum(gt[:no_tst_inst] == cls)/len(onn.Xt[:no_tst_inst, :])*100}%')

'''
A sigle MZI represents the following matrix:
        [a+bj c+dj]
        [e+fj h+lj]
The following identity exists due to unitary nature?:
    a^2+b^2+e^2+f^2 = 1 (Identity A)
    c^2+d^2+h^2+l^2 = 1 (Identity B)
    ac+bd+eh+fl = 0 (Identity C)
This is validated.

The following code sweep the theta and phi phase shifters for one MZI in the network
'''

exit()
accuracies = []
step = 40
theta_sweep = np.linspace(-np.pi, np.pi, step)
phi_sweep = np.linspace(-np.pi, np.pi, step)
for phi in phi_sweep:
    for theta in theta_sweep:
        # brute force through 2x2 ONN (single MZI) design space
        new_phases = [[(theta, phi)]]
        model.set_all_phases_uncerts_losses(new_phases)
        W_onn = model.get_transformation_matrix()[0]
        a = np.real(W_onn[0][0])
        b = np.imag(W_onn[0][0])
        c = np.real(W_onn[0][1])
        d = np.imag(W_onn[0][1])
        e = np.real(W_onn[1][0])
        f = np.imag(W_onn[1][0])
        h = np.real(W_onn[1][1])
        l = np.imag(W_onn[1][1])
        try:
            # assert a**2+b**2+e**2+f**2 == 1.0, f"ERROR: identity A"
            assert abs(a**2+b**2+e**2+f**2 - 1.0) <= 1e-9
            assert abs(c**2+d**2+h**2+l**2 - 1.0) <= 1e-9
            assert abs(a*c+b*d+e*h+f*l) <= 1e-9
        except:
            print(f"identity A = {a**2+b**2+e**2+f**2}")
            print(f"identity B = {c**2+d**2+h**2+l**2}")
            print(f"identity C = {a*c+b*d+e*h+f*l}")
            print(f"At phases:\n{model.get_all_phases()}")
            print(f"\nTransformation matrix:\n{W_onn}\n")
        # check accuracy
        yhat = model.forward_pass(onn.Xt[:no_tst_inst, :].T)
        cls = np.array([np.argmax(yhat) for yhat in yhat.T])
        gt = np.array([np.argmax(tru) for tru in onn.yt])
        acc = sum(gt[:no_tst_inst] == cls)/len(onn.Xt[:no_tst_inst, :])*100
        accuracies.append(acc)


accuracies = np.flipud(np.array(accuracies).reshape(int(np.sqrt(len(accuracies))), -1))
plt.imshow(accuracies, cmap='cool', interpolation='nearest',extent=[-np.pi,+np.pi,-np.pi,+np.pi])
plt.title("2x2 ONN (single MZI) binary MNIST (4 & 9) classification accuracy")
# print(ticks)
ticks = np.round(np.linspace(-np.pi, np.pi, 10), decimals=2)
plt.xticks(ticks)
plt.yticks(ticks)
plt.xlabel("Theta value (rad)")
plt.ylabel("Phi value (rad)")
plt.colorbar()  # Add color bar to show the scale
plt.savefig(f"2x2_ONN")
# plt.show()


onn.loss_diff = 0 # Set loss_diff
# For simulation purposes, defines range of loss and phase uncert
onn.loss_dB = np.linspace(0, 2, 3) # set loss/MZI range
onn.phase_uncert_theta = np.linspace(0., 1, 3) # set theta phase uncert range
onn.phase_uncert_phi = np.linspace(0., 1, 3) # set phi phase uncert range
# onn, model = test.test_PT(onn, onn.Xt, onn.yt, model, show_progress=True) # test Phi Theta phase uncertainty accurracy
# onn, model = test.test_LPU(onn, onn.Xt, onn.yt, model, show_progress=True) # test Loss/MZI + Phase uncert accuracy

# onn.saveAll(model) # Save best model information
