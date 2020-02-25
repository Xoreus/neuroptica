'''
Visualize the difference between max and second max average

Author: Simon Geoffroy-Gagnon
Edit: 25.02.2020
'''
import sys
import random
import pandas as pd
import numpy as np
import pickle
import matplotlib as mpl
mpl.use('Agg')
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
sys.path.append(r'C:\Users\sgeoff1\Documents\neuroptica')
sys.path.append('/home/simon/Documents/neuroptica')
import neuroptica as neu
import ONN_Simulation_Class as ONN_Cls
import ONN_Setups
import onnClassTraining as onn_trainer


def delta_getter(model, onn):
    X, y, Xt, yt = onn_trainer.change_dataset_shape(onn)
    
    Y_hat = model.forward_pass(X.T)
    pred = np.array([np.argmax(yhat) for yhat in Y_hat.T])
    gt = np.array([np.argmax(tru) for tru in onn.y])
    correct = np.equal(pred, gt)

    delta = np.array([sorted(yhat)[-1] - sorted(yhat)[-2] for yhat in Y_hat.T])
    data = {'Delta':delta, 'Class':gt, 'Correct':correct}
    df = pd.DataFrame(data)
    df = df[df.Correct != False]
    meanValue = []
    for cls in range(max(gt)+1):
        df_cls = df[df.Class == cls]
        meanValue.append(df_cls.Delta.mean())
    return meanValue

if __name__ == '__main__':
    onn = ONN_Cls.ONN_Simulation()
    
    FOLDER = '/home/simon/Documents/neuroptica/tests/Analysis/Good_Plots/retest-AllTopologies_3DAccMap_Gaussian_N=4_loss-diff=0.5_rng7'

    onn.X = np.loadtxt(f'{FOLDER}/Datasets/Gaussian_X_4Features_4Classes_Samples=560_Dataset.txt', delimiter=',')
    onn.Xt = np.loadtxt(f'{FOLDER}/Datasets/Gaussian_Xt_4Features_4Classes_Samples=560_Dataset.txt', delimiter=',')
    onn.y = np.loadtxt(f'{FOLDER}/Datasets/Gaussian_y_4Features_4Classes_Samples=560_Dataset.txt', delimiter=',')
    onn.yt = np.loadtxt(f'{FOLDER}/Datasets/Gaussian_yt_4Features_4Classes_Samples=560_Dataset.txt', delimiter=',')
    onn.FOLDER = 'Analysis/test'
    onn.onn_topo = 'C_Q_P'
    onn.onn_topo = 'R_P'
    onn.get_topology_name()
    onn.EPOCHS = 1200
    deltas = {}
    for onn.onn_topo in ['C_Q_P','R_P']:
        meanDelta = []
        for _ in range(10):
            best_acc = 0
            print('')
            for onn.rng in range(5):
                random.seed(onn.rng)
                model, onn = onn_trainer.train_single_onn(onn, create_dataset_flag=False) 
                if max(onn.val_accuracy)> best_acc:
                    print(f'Current best accuracy: {max(onn.val_accuracy):.3f}')
                    best_model = model
                    best_acc = max(onn.val_accuracy)

            meanDelta.append(delta_getter(best_model, onn))
        # print(np.mean(meanDelta, axis=0))
        # print(meanDelta)
        deltas.update({onn.onn_topo:np.mean(meanDelta, axis=0)})
        onn.createFOLDER()
        onn.saveAll(best_model)
        print(deltas)
