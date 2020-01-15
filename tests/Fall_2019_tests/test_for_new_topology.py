"""
This i3 a function to test Neuroptica's ONN with the MNIST dataset, using PCA to reduce the features from
784 to 4 and choosing only 4 numbers: [1, 3, 6, 7]
when normalised, the LNN gets best acc of 88% w/ [0.3.6.9]
best ONN Validation accuracy: 66% with [1,3,6,7]

Author: Simon Geoffroy-Gagnon
Edit: 03.09.2019
"""
import pandas as pd
from sklearn.datasets.samples_generator import make_blobs
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import PCA_MNIST as mnist

# Set random seed to always get same data
rng = 7
random.seed(rng)

sys.path.append('/home/simon/Documents/neuroptica')
import neuroptica as neu

def blob_maker(targets=4, features=4, nsamples=10000,
               cluster_std=.1):

    # generate 2d classification dataset
    X, y = make_blobs(n_samples=nsamples, centers=targets,
                      n_features=features,
                      cluster_std=cluster_std,
                      center_box=(0, 1), random_state=rng, shuffle=False)
    ohe_labels = pd.get_dummies(y).values
    x, xt, y, yt = train_test_split(X, ohe_labels, test_size=0.2)

    return x, y, xt, yt


# Number of input features?
N = 4

# Get mnist dataset
acc_tst = []
numbers = []

# X, y, Xt, yt, *_ = mnist.get_data(N=N)#[1,3,6,7], N=N)
calc_phase_name = 'Diamond_test'
X, y, Xt, yt = blob_maker(targets=N, features=N, nsamples=1000)
X = np.array([list(np.zeros(int((N-2)))) + list(samples) for samples in X])
Xt = np.array([list(np.zeros(int((N-2)))) + list(samples) for samples in Xt])

#print(X.shape)
# print(y.shape)
# pb.plot_blobs(X,y)

# Normalize inputs
X = (X - np.min(X))/(np.max(X) - np.min(X))

# rand_ind = random.sample(list(range(len(X))), 100)
# X = X[rand_ind]
# y = y[rand_ind]
# print(np.sum(y, axis=0))
# rand_ind = random.sample(list(range(len(Xt))), 100)
# Xt = Xt[rand_ind]
# yt = yt[rand_ind]

# create ReckMesh-DMM-ConjugateReckMesh
model = neu.Sequential([
    neu.DiamondLayer(N, include_phase_shifter_layer=False, loss=0),
    neu.Activation(neu.AbsSquared(N)),
    neu.DropMask(N, keep_ports=range(int((N-2)/2), N)),
])
setup = 'Diamond'
print(model)
print(X.shape)
print(y.shape)
# print(X)
# Yhat = model.forward_pass([1, 1])
# print(Yhat)
if 1:
    optimizer = neu.InSituAdam(model, neu.MeanSquaredError, step_size=0.001)
    # losses, trn_accuracy, val_accuracy = optimizer.fit(X.T, y.T, Xt.T, yt.T, epochs=700, batch_size=64)
    losses, trn_accuracy, val_accuracy = optimizer.fit(X.T, y.T, Xt.T, yt.T, epochs=700, batch_size=64, show_progress=True)
    acc_tst.append(val_accuracy[-1])

    D = []
    phases = []
    for layer in model.layers:
        if hasattr(layer, 'mesh'):
            D.append(layer.mesh.get_transfer_matrix())
            phases.append([x for x in layer.mesh.all_tunable_params()])

    thetas = [item for sublist in phases for (item, _) in sublist]
    phis = [item for sublist in phases for (_, item) in sublist]
    d = {'Theta':thetas, 'Phis':phis}

    df = pd.DataFrame(d)
    print(df)

    # print(np.array(thetas).T)
    # print(np.array(phis).T)
    # Get final accuracy
    outputs = model.forward_pass(Xt.T).T
    Yhat = np.array([np.argmax(yhat) for yhat in outputs])
    GT = np.array([np.argmax(gt) for gt in yt])
    acc = np.sum(Yhat == GT)/len(Xt)*100
    max_acc = max(trn_accuracy)
    max_iter = np.argmax(trn_accuracy)
    max_v_acc = max(val_accuracy)
    max_v_iter = np.argmax(val_accuracy)
    ax1 = plt.plot()
    plt.plot(losses, color='b')
    plt.xlabel('Epoch')
    plt.ylabel("$\mathcal{L}$", color='b')
    ax2 = plt.gca().twinx()
    ax2.plot(trn_accuracy, color='r')
    ax2.plot(val_accuracy, color='g')

    plt.ylabel('Accuracy', color='r')
    plt.legend(['Training Accuracy', 'Validation Accuracy'])
    plt.title((f'Gradient Descent, Max Validation Accuracy: {max_v_acc:.2f}% at epoch {max_v_iter}\n Max Training Accuracy: '
        f'{max_acc:.2f}% at epoch {max_iter}, '+ setup))
    plt.ylim([0, 100])
    fig = plt.gcf()
    fig.savefig(calc_phase_name + '.png')
    plt.show()

    # pb.plot_blobs(X,y)


if 0:# DMM
    D = []
    phases = []
    for layer in model.layers:
        if hasattr(layer, 'mesh'):
            D.append(layer.mesh.get_transfer_matrix())
            phases.append([x for x in layer.mesh.all_tunable_params()])

    thetas = [item for sublist in phases for (item, _) in sublist]
    # print(len(thetas))

    phis = [item for sublist in phases for (_, item) in sublist]
    # print(len(phis))

    Reck_layer_idx = [f'Reck, MZI_{mzi}' for mzi in range(1, int((N*(N-1))/2+1))]
    # print(Reck_layer_idx)

    DMM_layer_idx = [f'DMM, MZI_{mzi}' for mzi in range(int((N*(N-1))/2+1), int((N*(N-1))/2+1 + N))]
    # print(DMM_layer_idx)

    CC_Reck_layer_idx = [f'CC Reck, MZI_{mzi}' for mzi in range(int((N*(N-1))/2 + N+1 ), int(N +  (N*(N-1)) + 1))]
    # print(CC_Reck_layer_idx)

    df = pd.DataFrame(np.array([thetas, phis]).T, columns=['Theta','Phi'],  index=Reck_layer_idx + DMM_layer_idx + CC_Reck_layer_idx)


    print(df)
    df.to_csv(calc_phase_name + ".csv", index=False)
    np.savetxt(calc_phase_name + "_X.csv", X, delimiter=",")
    np.savetxt(calc_phase_name + "_y.csv", y, delimiter=",")
    np.savetxt(calc_phase_name + "_yt.csv", yt, delimiter=",")
    np.savetxt(calc_phase_name + "_Xt.csv", Xt, delimiter=",")
