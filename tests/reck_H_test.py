
"""
Testing the hermitian transpose reck mesh, single MZI
Author: Simon Geoffroy-Gagnon
Edit: 03.09.2019
"""
import pandas as pd
from sklearn.datasets.samples_generator import make_blobs
# import scatter_matrix_plotter as pb
# import iris_fourth_flower as ir
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import time
# Set random seed to always get same data
rng = 6
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

def printf(format, *args):
    sys.stdout.write(format % args)

# Number of input features?
N = 4
theta =  np.pi/2
phi =  np.pi/2
loss = 0 # dB

# X, y, Xt, yt, *_ = mnist.get_data([1,3,6,7])
# X, y, Xt, yt = blob_maker(targets=N, features=N, nsamples=1000)

X = np.genfromtxt('X.csv', delimiter=',')
Xt = np.genfromtxt('Xt.csv', delimiter=',')

y = np.genfromtxt('y.csv', delimiter=',')
yt = np.genfromtxt('yt.csv', delimiter=',')

# Normalize inputs
X = (X - np.min(X))/(np.max(X) - np.min(X))
# pb.plot_blobs(X,y)
if 0:
    print(len(X))
    rand_ind = random.sample(list(range(len(X))), 8000)
    X = X[rand_ind]
    y = y[rand_ind]
    print(np.sum(y, axis=0))
    rand_ind = random.sample(list(range(len(Xt))), 5000)
    Xt = Xt[rand_ind]
    yt = yt[rand_ind]

losses = [0, .25, .5, .75, 1, 1.25, 1.5] # in dB
iterations = 20 # number of times to retry same loss

D = []

for loss in losses:
    thetas = []
    phis = []
    acc_mean = []
    for iter in range(iterations):
        setup = 'Reck'
        model = neu.Sequential([
            # neu.ReckLayer(N, include_phase_shifter_layer=False, thetas=[None]*int(N*(N-1)/2), phis=[None]*int(N*(N-1)/2), loss=loss),

            # neu.AddMask(2*N),
            # neu.DMM_layer(2*N, thetas=[0]*int(N), phis=[None]*int(N)),
            # neu.DropMask(N=2*N, keep_ports=range(0, 2*N, 2)), # makes the DMM thetas pi
            # neu.DropMask(N=2*N, keep_ports=range(1, 2*N+1, 2)), # makes the DMM thetas 0 or 2*pi

            # neu.flipped_ReckLayer(N, include_phase_shifter_layer=False),
            neu.ReckLayer_H(N, include_phase_shifter_layer=False, thetas=[None]*int(N*(N-1)/2), phis=[None]*int(N*(N-1)/2), loss=loss),

            neu.Activation(neu.AbsSquared(N)), # photodetector measurement
        ])


        optimizer = neu.InSituAdam(model, neu.MeanSquaredError, step_size=0.0005)
        losses, trn_accuracy, val_accuracy, _, _ = optimizer.fit(X.T, y.T, Xt.T, yt.T, epochs=2000, batch_size=2**5, show_progress=False)

        phases = []
        D_curr = []
        for layer in model.layers:
            if hasattr(layer, 'mesh'):
                phases.append([x for x in layer.mesh.all_tunable_params()])

        thetas.append([item for sublist in phases for (item, _) in sublist])
        phis.append([item for sublist in phases for (_, item) in sublist])

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
        plt.savefig(f'loss={loss}_iter={iter}.png')
        plt.close()

        Reck_layer_idx = [f'Reck, MZI_{mzi}' for mzi in range(int(N*(N-1)/2)+N)]
        DMM_layer_idx = [f'DMM, MZI_{mzi}' for mzi in range(int((N*(N-1))/2),int(N*(N-1)/2+N))]
        Reck_H_layer_idx = [f'Reck_H, MZI_{mzi}' for mzi in range(int(N*(N-1)/2+N),int(N*(N-1)+2*N))]
        # df = pd.DataFrame(np.array([thetas, phis]).T, columns=['Theta','Phi'],  index=Reck_layer_idx)
        # df = pd.DataFrame(np.array([thetas, phis]).T, columns=['Theta','Phi'])
        # df = pd.DataFrame(np.array([thetas, phis]).T, columns=['Theta','Phi'], index=Reck_layer_idx+DMM_layer_idx+Reck_H_layer_idx)

        # print(df.round(2))
        acc_mean.append(val_accuracy[-1])
        # print(D)
        np.savetxt("X.csv", X, delimiter=",")
        np.savetxt("y.csv", y, delimiter=",")
        np.savetxt("yt.csv", yt, delimiter=",")
        np.savetxt("Xt.csv", Xt, delimiter=",")
    mean_t = np.array(thetas).mean(axis=0)
    std_t = np.array(thetas).std(axis=0)
    mean_p = np.array(phis).mean(axis=0)
    std_p = np.array(phis).std(axis=0)
    print(f"loss = {loss}")
    print("Mean Accuracy: {}".format(np.array(acc_mean).mean()))
    print("thetas, mean:")
    print(mean_t)
    print("thetas, std:")
    print(std_t)
    print("phis, mean:")
    print(mean_p)
    print("phis, std:")
    print(std_p)
    print('')
# np.savetxt("Phases.txt", phases, delimiter=",")
    # print(D)
