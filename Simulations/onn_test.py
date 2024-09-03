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
from main import save_onn


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

# find the nearest number in the array to the value
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

# weight quantization
def get_phase_levels(V_pi = 1.92, V_2pi = 2.73, V_max = 4, b = 8):
    # (Bokun Modified)
    # assuming V_pi, V_2pi and V_max all known, number of bits also known.
    # voltage-phaseShift relation: phaseShift = pi * (V_in^2 / V_pi^2)
    # print(f"Actual     phaseShift at V_2pi: {2*np.pi}")
    # print(f"Calculated phaseShift at V_2pi: {np.pi*(V_2pi**2/V_pi**2)}")
    # V_in can take 2**b values, evenly spaced between [0, V_max] volts:
    V_ins = np.linspace(0, V_max, 2**b)
    # corresponding to these possible phaseShifts:
    phase_levels = np.pi * (V_ins**2 / V_pi**2)
    # however, phaseShifts greater than 2pi is not used:
    phase_levels[phase_levels>2*np.pi] = 0
    return phase_levels

def get_quantized_phases(onn, b = 8):
    phase_levels = get_phase_levels(b = b)
    # "onn.phases" or "model.get_all_phases()" returns a (#mesh, #MZI, 2) array, innermost-dim is tuple (theta, phi)
    original_phases = onn.model.get_all_phases()
    quantized_phases = []
    for each_mesh in original_phases:
        quantized_phases.append([(find_nearest(phase_levels, each_phase_pair[0]), find_nearest(phase_levels, each_phase_pair[1])) 
                                 for each_phase_pair in each_mesh])
    return quantized_phases # set it using "model.set_all_phases_uncerts_losses(Phases=quantized_phases)"


no_tst_inst = 8000 # 10000 for MNIST, 8000 for CIFAR-10
zeta = 0.60
# zeta used as the % of max_accuracy, above which FoM is counted. E.g. max_acc = 78.00%, then points whose acc ≥ (zeta*78.00)% contributes to FoM
# or the as absolute threshold accuracy, e.g. zeta = 0.60 (60%)

for volt_bits in [8]: # [4, 6, 8, 10, 12, 14, 16]
    for N in [8, 16]: # or [8, 16] 
        for beta in [1.2]: # [1.0, 1.2, 1.4]
            for seed in [13, 21, 47, 50, 91]: # or [13, 21, 47, 50, 91]
                for layernum in ['']: # or ['', '_2', '_3']
                    print(f"===================== {volt_bits} voltage bits === N={N} === beta={beta} === seed={seed} === layer{layernum} =======================")
                    onn = ONN_Cls.ONN_Simulation() # Required for containing training/simulation information
                    onn.topo = f'Frontier_BETA_{beta}_N={N}_miniBokun{layernum}_EO_rng={seed}_auto' # .pkl doesn't have "NxN" in the file name
                    onn = onn.pickle_load(onn_folder=f'/Users/bokunzhao/Documents/McGill/Winter2024-G1/ECSE 691_697/WIDER_IOU_50_8x8/neuroptica/Simulations/Analysis/iris_augment/Bokun_BETA_CIFAR/{N}x2_Frontier_BETA_{beta}_N={N}_miniBokun{layernum}_EO_rng={seed}_auto')
                    onn.FOLDER = f'/Users/bokunzhao/Documents/McGill/Winter2024-G1/ECSE 691_697/WIDER_IOU_50_8x8/neuroptica/Simulations/Analysis/iris_augment/Bokun_BETA_CIFAR/{N}x2_Frontier_BETA_{beta}_N={N}_miniBokun{layernum}_EO_rng={seed}_auto'
                    # print(f"{onn.FOLDER}")
                    onn.zeta = zeta
                    model = onn.model
                    model.set_all_phases_uncerts_losses(Phases=onn.phases) # resets effect of uncertainty simulation
                    # print("Phases (model original):")
                    # print(f"phases({np.shape(model.get_all_phases())}):{model.get_all_phases()}")
                    # print("Phases (onn original):") # same as above two lines (but not a copy!)
                    # print(f"phases({np.shape(onn.phases)}):{onn.phases}")
                    quantized_phases = get_quantized_phases(onn, b=volt_bits)
                    model.set_all_phases_uncerts_losses(Phases=quantized_phases)  # sets the model phases
                    onn.phases = quantized_phases # sets the onn phases
                    # print("Phases (model quantized):")
                    # print(f"phases({np.shape(quantized_phases)}):{quantized_phases}")
                    # print("Phases (onn quantized):") # same as above two lines (but not a copy!)
                    # print(f"phases({np.shape(onn.phases)}):{onn.phases}")
                    # exit()
                    
                    save_onn(onn, onn.model) # if want to plot with new accuracies, uncomment this
                    # onn.plotAll(cmap='hsv', trainingLoss=0.00) # replot PT and LPU without resimulation (e.g. after adjusting Figure text size )
                    continue
                    # break


                    # mesh_no = 0 # 0, 1, 2... select mesh layer
                    # col_no = 9 # 0, 1, 2... select "MZI column" within a mesh layer
                    # row_no = 1 # 0, 1, 2... select which MZI within a "MZI column"

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

                    # print(f"Using {no_tst_inst} out of {onn.X_test.shape[0]} test samples")

                    '''====Use forward pass ===='''
                    yhat = model.forward_pass(onn.X_test[:no_tst_inst, :].T) # using forward pass
                    '''====Or use W directly===='''
                    # print(f"UnitaryMat({np.shape(model.get_transformation_matrix())}):\n{model.get_transformation_matrix()}")
                    # W = model.get_transformation_matrix()[0] # The unitary weight matrix represented by the mesh
                    # yhat = np.matmul(W, onn.X_test[:no_tst_inst, :].T)[onn.features//2-1:onn.features//2+1 ,:]
                    # yhat = np.abs(yhat) ** 2 # Tested: same as using forward pass
                    '''========================='''
                    # print(yhat.shape)
                    # exit()
                    cls = np.array([np.argmax(yhat) for yhat in yhat.T])
                    gt = np.array([np.argmax(tru) for tru in onn.y_test]) # y_test: one-hot encoded vectors; gt: actual scalar labels
                    # print(f"Testing with first {no_tst_inst} test data instances:")
                    # print(f"example cls[0:10] {cls.shape}: {cls[0:10]}")
                    # print(f"example gt[0:10] {gt.shape}: {gt[0:10]}")

                    # Calculate True Positives, False Positives, and False Negatives
                    TP = np.sum((cls == 1) & (gt == 1))
                    FP = np.sum((cls == 1) & (gt == 0))
                    TN = np.sum((cls == 0) & (gt == 0))
                    FN = np.sum((cls == 0) & (gt == 1))
                    # Precision and Recall
                    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
                    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
                    # F1 Score
                    if (precision + recall) > 0:
                        f1_score = 2 * (precision * recall) / (precision + recall)
                    else:
                        f1_score = 0
                    P_trigger = (FP + TP)/(TN + FP + FN + TP)

                    # print(f'Accuracy = {sum(gt[:no_tst_inst] == cls)/len(onn.X[:no_tst_inst, :])*100}%')
                    # print(f'Val. Accuracy = {max(onn.val_accuracy)}%')
                    # print(f'F1 Score: {(f1_score*100):.3f}%.')
                    # print(f'False Nagatives: {FN}.')
                    # print(f'P(trigger): {(P_trigger):.5f}.')
                    print(f'(accuracy               F1               FN               P_trigger)')
                    print(f'{sum(gt[:no_tst_inst] == cls)/len(onn.X[:no_tst_inst, :])*100:.5f} {(f1_score*100):.3f} {FN} {(P_trigger):.5f}\n')
                    # print(f"first {no_tst_inst} Xt:\n{onn.Xt[:no_tst_inst, :].T}") # (10, no_tst_inst) for MNIST dataset
                    # print(f"first {no_tst_inst} yhat:\n{yhat}") # (10, no_tst_inst) for MNIST dataset
                    # print(f"first {no_tst_inst} labels:\n{gt[:no_tst_inst]}")
                    # print(f"\n")
                    # print(f"first {no_tst_inst} Xt squared summation:\n{np.sum(onn.Xt[:no_tst_inst, :].T**2, axis=0)}")
                    # print(f"\n")
                    # print(f"first {no_tst_inst} yhat summation:\n{np.sum(yhat, axis=0)}") # (10, no_tst_inst) for MNIST dataset
                    # print(f"first {no_tst_inst} yhat abs. squared summation:\n{np.sum(np.abs(yhat)**2, axis=0)}") # (10, no_tst_inst) for MNIST dataset

                    # print(yhat.shape)

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
