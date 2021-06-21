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
sys.path.append('../')
import neuroptica as neu

onn = ONN_Cls.ONN_Simulation() # Required for containing training/simulation information
onn.topo = 'ONN'
# onn.FOLDER = '/home/edwar/Documents/Github_Projects/neuroptica/tests/Analysis/iris_augment/4x3'
onn.FOLDER = '/home/edwar/Documents/Github_Projects/neuroptica/tests/Analysis/iris_augment/4x3_test'
onn = onn.pickle_load()
model = onn.model
model.set_all_phases_uncerts_losses(Phases=onn.phases)
yhat = model.forward_pass(onn.Xt.T)
cls = np.array([np.argmax(yhat) for yhat in yhat.T])
gt = np.array([np.argmax(tru) for tru in onn.yt])
# print(onn.model.get_all_phases())
# print(cls)
# print(gt)

print(f'Accuracy = {sum(gt == cls)/len(onn.Xt)*100}%')


onn.loss_diff = 0 # Set loss_diff
# For simulation purposes, defines range of loss and phase uncert
onn.loss_dB = np.linspace(0, 2, 3) # set loss/MZI range
onn.phase_uncert_theta = np.linspace(0., 1, 3) # set theta phase uncert range
onn.phase_uncert_phi = np.linspace(0., 1, 3) # set phi phase uncert range
# onn, model = test.test_PT(onn, onn.Xt, onn.yt, model, show_progress=True) # test Phi Theta phase uncertainty accurracy
# onn, model = test.test_LPU(onn, onn.Xt, onn.yt, model, show_progress=True) # test Loss/MZI + Phase uncert accuracy

# onn.saveAll(model) # Save best model information
